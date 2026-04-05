"""Dataset preparation functions."""

import inspect

import numpy as np
import torch
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_california_housing, fetch_openml, make_regression, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import TensorDataset

from bcr import set_seed


def prepare_synthetic_regression(
    n_samples: int = 4000,
    n_features: int = 20,
    n_informative: int = 15,
    noise: float = 20.0,
    seed: int = 0,
):
    """Prepare synthetic regression dataset."""
    set_seed(seed)
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X).astype("float32")
    y = y.astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    return train_ds, torch.tensor(X_te), torch.tensor(y_te), X.shape[1]


def prepare_california_housing():
    """Prepare California housing regression dataset."""
    cal = fetch_california_housing(as_frame=True)
    X = cal.data.values.astype("float32")
    y = cal.target.values.astype("float32")
    X = StandardScaler().fit_transform(X).astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    return train_ds, torch.tensor(X_te), torch.tensor(y_te), X.shape[1]


def _get_onehot_encoder():
    """Get OneHotEncoder with correct API for sklearn version."""
    if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def prepare_adult_income():
    """Prepare Adult Income classification dataset."""
    adult = fetch_openml("adult", version=2, as_frame=True)
    X_df = adult.data
    y_ser = (adult.target == ">50K").astype("int64")

    num_cols = X_df.select_dtypes("number").columns.to_list()
    cat_cols = X_df.select_dtypes("object").columns.to_list()

    preproc = ColumnTransformer(
        [("num", StandardScaler(), num_cols), ("cat", _get_onehot_encoder(), cat_cols)]
    )
    X = preproc.fit_transform(X_df).astype("float32")
    y = y_ser.to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y
    )
    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    return train_ds, torch.tensor(X_te), torch.tensor(y_te), X.shape[1]


def prepare_german_credit():
    """Prepare German Credit classification dataset."""
    credit = fetch_openml("credit-g", version=1, as_frame=True)
    X_df = credit.data
    y_ser = (credit.target == "good").astype("int64")

    num_cols = X_df.select_dtypes("number").columns.to_list()
    cat_cols = X_df.select_dtypes("object").columns.to_list()

    preproc = ColumnTransformer(
        [("num", StandardScaler(), num_cols), ("cat", _get_onehot_encoder(), cat_cols)]
    )
    X = preproc.fit_transform(X_df).astype("float32")
    y = y_ser.to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y
    )
    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    return train_ds, torch.tensor(X_te), torch.tensor(y_te), X.shape[1]


def prepare_diabetes():
    """Prepare Diabetes regression dataset from sklearn."""
    diabetes = load_diabetes()
    X = diabetes.data.astype("float32")
    y = diabetes.target.astype("float32")
    X = StandardScaler().fit_transform(X).astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    return train_ds, torch.tensor(X_te), torch.tensor(y_te), X.shape[1]


def prepare_wine_quality():
    """Prepare Wine Quality regression dataset from UCI (red wine)."""
    wine = fetch_openml(data_id=287, as_frame=True)
    X_df = wine.data
    y_ser = wine.target.astype("float32")
    X = X_df.values.astype("float32")
    y = y_ser.values
    X = StandardScaler().fit_transform(X).astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    return train_ds, torch.tensor(X_te), torch.tensor(y_te), X.shape[1]


def prepare_bank_marketing():
    """Prepare Bank Marketing classification dataset from UCI (data_id=1461)."""
    bank = fetch_openml(data_id=1461, as_frame=True)
    X_df = bank.data
    y_ser = (bank.target == "2").astype("int64")

    num_cols = X_df.select_dtypes("number").columns.to_list()
    cat_cols = X_df.select_dtypes(["object", "category"]).columns.to_list()

    preproc = ColumnTransformer(
        [("num", StandardScaler(), num_cols), ("cat", _get_onehot_encoder(), cat_cols)]
    )
    X = preproc.fit_transform(X_df).astype("float32")
    y = y_ser.to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y
    )
    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    return train_ds, torch.tensor(X_te), torch.tensor(y_te), X.shape[1]


def prepare_credit_card_default():
    """Prepare Credit Card Default classification dataset from UCI (data_id=42477)."""
    credit = fetch_openml(data_id=42477, as_frame=True)
    X_df = credit.data
    y_ser = (credit.target == "1").astype("int64")

    num_cols = X_df.select_dtypes("number").columns.to_list()
    cat_cols = X_df.select_dtypes(["object", "category"]).columns.to_list()

    if cat_cols:
        preproc = ColumnTransformer(
            [("num", StandardScaler(), num_cols), ("cat", _get_onehot_encoder(), cat_cols)]
        )
    else:
        preproc = StandardScaler()
    X = preproc.fit_transform(X_df).astype("float32")
    y = y_ser.to_numpy()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=1, stratify=y
    )
    train_ds = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr))
    return train_ds, torch.tensor(X_te), torch.tensor(y_te), X.shape[1]


def create_challenging_datasets():
    """Create challenging datasets for ablation studies."""
    datasets = {}

    set_seed(42)
    X, y = make_regression(
        n_samples=200, n_features=50, n_informative=10, noise=15.0, random_state=42
    )
    X = StandardScaler().fit_transform(X).astype("float32")
    y = y.astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    datasets["low_n_high_p"] = {
        "train_ds": TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        "test_x": torch.tensor(X_te),
        "test_y": torch.tensor(y_te),
        "d_in": X.shape[1],
        "description": "Low n (200), high p (50)",
    }

    set_seed(43)
    n, p = 2000, 20
    cov = np.eye(p)
    for i in range(p):
        for j in range(p):
            cov[i, j] = 0.9 ** abs(i - j)
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n).astype("float32")
    beta = np.random.randn(p).astype("float32")
    y = (X @ beta + np.random.randn(n) * 5).astype("float32")
    X = StandardScaler().fit_transform(X).astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    datasets["correlated_features"] = {
        "train_ds": TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        "test_x": torch.tensor(X_te),
        "test_y": torch.tensor(y_te),
        "d_in": X.shape[1],
        "description": "Correlated features (AR(1), rho=0.9)",
    }

    set_seed(44)
    X, y = make_regression(
        n_samples=2000, n_features=20, n_informative=15, noise=10.0, random_state=44
    )
    X = StandardScaler().fit_transform(X).astype("float32")
    y = y.astype("float32")
    n_corrupt = int(0.2 * len(y))
    corrupt_idx = np.random.choice(len(y), n_corrupt, replace=False)
    y[corrupt_idx] = np.random.randn(n_corrupt) * y.std() * 2
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    datasets["label_noise"] = {
        "train_ds": TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        "test_x": torch.tensor(X_te),
        "test_y": torch.tensor(y_te),
        "d_in": X.shape[1],
        "description": "20% label noise",
    }

    set_seed(45)
    X, y = make_regression(
        n_samples=100, n_features=10, n_informative=8, noise=10.0, random_state=45
    )
    X = StandardScaler().fit_transform(X).astype("float32")
    y = y.astype("float32")
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=1)
    datasets["overcapacity"] = {
        "train_ds": TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
        "test_x": torch.tensor(X_te),
        "test_y": torch.tensor(y_te),
        "d_in": X.shape[1],
        "description": "Overcapacity (n=100, p=10)",
    }

    return datasets
