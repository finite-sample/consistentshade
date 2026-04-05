"""High-level BCR training API."""

import itertools
from typing import Callable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler, TensorDataset

from ..config import BCR_CONFIG, CLASSIFICATION_CONFIG, REGRESSION_CONFIG
from ..models import DropMLP
from ..utils import set_seed


class BCRTrainer:
    """
    High-level API for Bootstrap Consistency Regularization training.

    Example:
        trainer = BCRTrainer(d_in=10, task='regression', K=3, lam=0.05)
        trainer.fit(train_dataset, epochs=25)
        preds = trainer.predict(test_x)
    """

    def __init__(
        self,
        d_in: int,
        task: str = "regression",
        K: int = None,
        lam: float = None,
        hidden_dim: int = None,
        dropout: float = None,
        lr: float = None,
        batch_size: int = None,
        seed: int = None,
        model_factory: Optional[Callable[[], nn.Module]] = None,
    ):
        """
        Initialize BCR trainer.

        Args:
            d_in: Input dimension
            task: 'regression' or 'classification'
            K: Number of bootstrap models (default from BCR_CONFIG)
            lam: Regularization strength (default from BCR_CONFIG)
            hidden_dim: Hidden layer size
            dropout: Dropout rate
            lr: Learning rate
            batch_size: Training batch size
            seed: Random seed for reproducibility
            model_factory: Optional custom model factory function
        """
        self.d_in = d_in
        self.task = task
        self.is_classification = task == "classification"

        cfg = CLASSIFICATION_CONFIG if self.is_classification else REGRESSION_CONFIG

        self.K = K if K is not None else BCR_CONFIG.K
        self.lam = lam if lam is not None else BCR_CONFIG.lam
        self.hidden_dim = hidden_dim if hidden_dim is not None else cfg.hidden_dim
        self.dropout = dropout if dropout is not None else cfg.dropout
        self.lr = lr if lr is not None else cfg.lr
        self.batch_size = batch_size if batch_size is not None else cfg.batch_size
        self.seed = seed

        self.out_dim = 2 if self.is_classification else 1
        self.model_factory = model_factory

        self.models = None
        self.optimizer = None
        self._fitted = False

    def _create_models(self):
        """Create K models for BCR training."""
        if self.seed is not None:
            set_seed(self.seed)

        if self.model_factory is not None:
            self.models = nn.ModuleList([self.model_factory() for _ in range(self.K)])
        else:
            self.models = nn.ModuleList(
                [
                    DropMLP(d_in=self.d_in, hid=self.hidden_dim, p=self.dropout, out_dim=self.out_dim)
                    for _ in range(self.K)
                ]
            )

        self.optimizer = torch.optim.Adam(
            itertools.chain(*(m.parameters() for m in self.models)), lr=self.lr
        )

    def fit(
        self,
        train_data: Union[Dataset, TensorDataset],
        epochs: int = None,
        verbose: bool = False,
    ) -> "BCRTrainer":
        """
        Train the BCR ensemble.

        Args:
            train_data: Training dataset (TensorDataset or similar)
            epochs: Number of training epochs
            verbose: Print training progress

        Returns:
            self for method chaining
        """
        self._create_models()

        cfg = CLASSIFICATION_CONFIG if self.is_classification else REGRESSION_CONFIG
        epochs = epochs if epochs is not None else cfg.epochs

        loader = DataLoader(train_data, batch_size=self.batch_size, sampler=RandomSampler(train_data))

        for m in self.models:
            m.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for xb, yb in loader:
                B = len(xb)
                preds = []
                sup_losses = []

                for m in self.models:
                    px = m(xb)
                    preds.append(px)
                    weights = torch.poisson(torch.ones(B, device=xb.device))

                    if self.is_classification:
                        per_sample = F.cross_entropy(px, yb, reduction="none")
                    else:
                        per_sample = (px - yb) ** 2

                    weighted_loss = (weights * per_sample).mean()
                    sup_losses.append(weighted_loss)

                preds = torch.stack(preds)
                sup_loss = torch.stack(sup_losses).mean()

                if self.is_classification:
                    probs = F.softmax(preds, dim=-1)
                    var_pen = probs.var(dim=0).mean()
                else:
                    var_pen = preds.var(dim=0, unbiased=False).mean()

                loss = sup_loss + self.lam * var_pen

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / n_batches:.4f}")

        self._fitted = True
        return self

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make ensemble predictions.

        Args:
            x: Input features (N, d_in)

        Returns:
            Predictions (N,) for regression or (N, n_classes) for classification
        """
        if not self._fitted:
            raise RuntimeError("Trainer must be fitted before making predictions")

        for m in self.models:
            m.eval()

        preds = torch.stack([m(x) for m in self.models])
        return preds.mean(0)

    @torch.no_grad()
    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple:
        """
        Make predictions with uncertainty estimates.

        Args:
            x: Input features (N, d_in)

        Returns:
            Tuple of (mean_predictions, std_predictions)
        """
        if not self._fitted:
            raise RuntimeError("Trainer must be fitted before making predictions")

        for m in self.models:
            m.eval()

        preds = torch.stack([m(x) for m in self.models])
        return preds.mean(0), preds.std(0)

    def evaluate(
        self, test_x: torch.Tensor, test_y: torch.Tensor
    ) -> tuple:
        """
        Evaluate model on test data.

        Args:
            test_x: Test features
            test_y: Test targets

        Returns:
            Tuple of (predictions, metric) where metric is RMSE for regression
            or accuracy for classification
        """
        preds = self.predict(test_x)

        if self.is_classification:
            metric = (preds.argmax(1) == test_y).float().mean().item()
        else:
            metric = torch.sqrt(F.mse_loss(preds, test_y)).item()

        return preds.numpy(), metric
