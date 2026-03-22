# Bootstrap-Consistency Regularization

**Training Neural Networks for Prediction Stability**

When you retrain a neural network on a new bootstrap sample, predictions can change more than you'd expect. BCR directly penalizes this instability during training, reducing prediction variance by 22-50% with minimal accuracy loss.

## Installation

```bash
git clone https://github.com/finite-sample/consistentshade.git
cd consistentshade
pip install -r requirements.txt
```

**Requirements:** Python 3.8+, PyTorch, scikit-learn, pandas, numpy

## Quick Start

```python
import torch
from scripts.trainers import train_bcr_regression, train_bcr_classification

# For regression
predictions, rmse = train_bcr_regression(
    seed=42,
    train_ds=your_train_dataset,  # TensorDataset
    test_x=test_features,          # torch.Tensor
    test_y=test_targets,           # torch.Tensor
    d_in=n_features,
    K=3,      # number of shadow models
    lam=0.05, # variance penalty strength
)

# For classification
predictions, accuracy = train_bcr_classification(
    seed=42,
    train_ds=your_train_dataset,
    test_x=test_features,
    test_y=test_labels,
    d_in=n_features,
    K=3,
    lam=0.05,
)
```

## Running Experiments

### Main experiments (4 datasets, 8 methods, 30 replicates each)
```bash
python scripts/run_main_experiments.py
```

Results saved to `tabs/` with bootstrap standard errors.

### Hyperparameter sensitivity analysis
```bash
python scripts/run_sensitivity.py
```

Sweeps over K, batch size, learning rate, and hidden dimensions.

### Statistical analysis with significance tests
```bash
python scripts/statistical_analysis.py
```

Computes bootstrap CIs, pairwise p-values, and Cohen's d effect sizes.

### Generate LaTeX tables for paper
```bash
python scripts/generate_tables.py
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `K` | 3 | Number of shadow models (more = better approximation, more memory) |
| `lam` | 0.05 | Variance penalty strength (higher = more stable, may reduce accuracy) |
| `bs` | 64 | Batch size |
| `lr` | 1e-3 | Learning rate |
| `hid` | 64 | Hidden layer dimension |

## Results Summary

| Dataset | Stability Reduction | Accuracy Change |
|---------|--------------------:|----------------:|
| German Credit | **50%** | -0.9 pp |
| Adult Income | **32%** | < 0.1 pp |
| California Housing | **24%** | +1.5% RMSE |
| Synthetic | **34%** | +4% RMSE |

BCR outperforms baselines (SAM, R-Drop, weight decay) on classification tasks. For some regression tasks, standard bagging may be equally effective.

## How It Works

1. **Shadow models**: Train K copies of your network simultaneously
2. **Poisson weighting**: Each copy sees different bootstrap weights on the same mini-batch
3. **Variance penalty**: Penalize disagreement between shadow models on identical inputs
4. **Joint optimization**: All models trained together, sharing the variance penalty gradient

```
Loss = (1/K) * sum(weighted_loss_k) + lambda * variance_penalty
```

At inference, use any single model or average predictions.

## Project Structure

```
consistentshade/
├── scripts/
│   ├── trainers/          # Training functions for all methods
│   │   ├── bcr.py         # Bootstrap-Consistency Regularization
│   │   ├── baseline.py    # Standard ERM
│   │   ├── sam.py         # Sharpness-Aware Minimization
│   │   └── ...
│   ├── datasets.py        # Dataset preparation
│   ├── metrics.py         # Stability metrics with bootstrap CIs
│   ├── run_main_experiments.py
│   ├── run_sensitivity.py
│   └── statistical_analysis.py
├── tabs/                  # Results (CSV + LaTeX)
├── figs/                  # Figures
└── ms/                    # Manuscript (LaTeX)
```

## Citation

Click "Cite this repository" on GitHub or use:

```bibtex
@software{sood2026bcr,
  title={Bootstrap-Consistency Regularization: Training Neural Networks for Prediction Stability},
  author={Sood, Gaurav},
  year={2026},
  url={https://github.com/finite-sample/consistentshade}
}
```

## License

MIT
