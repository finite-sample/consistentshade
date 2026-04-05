# Bootstrap-Consistency Regularization

**Training Neural Networks for Prediction Stability**

When you retrain a neural network on a new bootstrap sample, predictions can change more than you'd expect. BCR directly penalizes this instability during training, reducing prediction variance by 22-65% with minimal accuracy loss.

## Installation

```bash
pip install -e .
```

**Requirements:** Python 3.9+, PyTorch 2.0+

## Quick Start

```python
from bcr import BCRTrainer
import torch

# Create trainer
trainer = BCRTrainer(
    d_in=n_features,
    task='regression',  # or 'classification'
    K=3,                # number of shadow models
    lam=0.05            # variance penalty strength
)

# Train on your dataset
trainer.fit(train_dataset, epochs=25)

# Make predictions
predictions = trainer.predict(test_x)

# Or get predictions with uncertainty
mean_pred, std_pred = trainer.predict_with_uncertainty(test_x)
```

For lower-level control:

```python
from bcr import train_bcr_regression, train_bcr_classification

# Regression
predictions, rmse = train_bcr_regression(
    seed=42,
    train_ds=train_dataset,
    test_x=test_features,
    test_y=test_targets,
    d_in=n_features,
    K=3,
    lam=0.05,
)

# Classification
predictions, accuracy = train_bcr_classification(
    seed=42,
    train_ds=train_dataset,
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
python -m experiments.run_main
```

Results saved to `tabs/` with bootstrap standard errors.

### Comprehensive hyperparameter analysis
```bash
python -m experiments.run_comprehensive
```

Runs K/lambda grid search, challenging scenarios, and sample size scaling.

### Statistical analysis with significance tests
```bash
python -m experiments.statistical_analysis
```

Computes bootstrap CIs, pairwise p-values, and Cohen's d effect sizes.

### Generate LaTeX tables for paper
```bash
python -m experiments.generate_tables
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

| Dataset | Stability Improvement | Accuracy Change |
|---------|----------------------:|----------------:|
| German Credit | **47%** | -1.0 pp |
| Adult Income | **32%** | < 0.1 pp |
| California Housing | **24%** | +1.4% RMSE |
| Synthetic | **34%** | +4% RMSE |

BCR outperforms baselines (SAM, R-Drop, weight decay) on classification tasks. For some regression tasks, standard bagging may be equally effective.

### Key Findings from Comprehensive Experiments

**When BCR helps most:**
- Low n, high p scenarios: **45-65%** stability improvement
- Underdetermined problems benefit most from regularization
- Benefits scale inversely with sample size (33% at n=1.5k → 20% at n=15k)

**Robustness:**
- Handles 10-40% label noise: **24-38%** improvement
- Correlated features (ρ=0.5-0.9): **15-31%** improvement

**Optimal hyperparameters:**
- K=3-5 provides good stability-compute tradeoff
- λ=0.05-0.1 balances stability and accuracy
- Diminishing returns beyond K=5

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
bcr/
├── bcr/
│   ├── models/           # DropMLP and other architectures
│   ├── regularizers/     # BCR, IFR regularization implementations
│   ├── training/         # BCRTrainer and training functions
│   ├── metrics/          # Stability metrics with bootstrap CIs
│   ├── optimizers/       # SAM optimizer
│   └── config.py         # Default configurations
├── experiments/
│   ├── run_main.py       # Main comparison experiments
│   ├── run_comprehensive.py  # K/lambda grid, scenarios, scaling
│   ├── run_sensitivity.py    # Hyperparameter sweeps
│   └── generate_tables.py    # LaTeX table generation
├── tabs/                 # Results (CSV + LaTeX)
│   └── comprehensive/    # K/lambda grid, scenarios, scaling results
├── figs/                 # Figures
└── ms/                   # Manuscript (LaTeX)
```

## Citation

Click "Cite this repository" on GitHub or use:

```bibtex
@software{sood2026bcr,
  title={Bootstrap-Consistency Regularization: Training Neural Networks for Prediction Stability},
  author={Sood, Gaurav},
  year={2026},
  url={https://github.com/soodoku/bcr}
}
```

## License

MIT
