# exp_common.py

import random
from typing import Tuple, Optional

import numpy as np
import torch
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_binary_labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y).reshape(-1)
    uniq = np.unique(y)
    if len(uniq) != 2:
        raise ValueError(f"Binary classification only, got labels: {uniq}")
    if set(uniq.tolist()) == {0, 1}:
        return y.astype(np.float32)
    return (y == uniq[1]).astype(np.float32)


class DatasetAwareSplitter:
    """Repeat a fixed holdout split, or fall back to ordinary CV.

    Every real-data benchmark loader attaches a predefined outer split. Plain
    arrays supplied by other callers retain shuffled K-fold or stratified
    K-fold behavior.
    """

    def __init__(self, base_splitter, n_splits: int):
        self.base_splitter = base_splitter
        self.n_splits = n_splits

    def split(self, X, y=None, groups=None):
        predefined = getattr(X, "predefined_split", None)
        if predefined is None:
            yield from self.base_splitter.split(X, y, groups)
            return

        train_indices, test_indices = predefined
        train_indices = np.asarray(train_indices, dtype=np.int64)
        test_indices = np.asarray(test_indices, dtype=np.int64)
        if (
            train_indices.ndim != 1
            or test_indices.ndim != 1
            or len(train_indices) == 0
            or len(test_indices) == 0
            or np.any(train_indices < 0)
            or np.any(test_indices < 0)
            or np.any(train_indices >= len(X))
            or np.any(test_indices >= len(X))
            or np.intersect1d(train_indices, test_indices).size
        ):
            raise ValueError("Invalid predefined train/test indices attached to X")
        for _ in range(self.n_splits):
            # Each caller advances its training seed by the repeat index; the
            # untouched outer test partition is identical across repetitions.
            yield train_indices.copy(), test_indices.copy()

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


def make_cv_splitter(task_type: str, n_splits: int, seed: int):
    """Use repeated holdout when present, otherwise ordinary/stratified CV."""
    kwargs = dict(n_splits=n_splits, shuffle=True, random_state=seed)
    if task_type == "classification":
        base_splitter = StratifiedKFold(**kwargs)
    else:
        base_splitter = KFold(**kwargs)
    return DatasetAwareSplitter(base_splitter, n_splits)


def early_stopping_split_indices(
    y: np.ndarray,
    task_type: str,
    seed: int,
    validation_fraction: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split an outer training partition into fit and early-stopping subsets."""
    indices = np.arange(len(y))
    stratify = ensure_binary_labels(y) if task_type == "classification" else None
    fit_indices, stop_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return fit_indices, stop_indices


def split_development_evaluation_data(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    seed: int,
    development_fraction: float = 0.2,
):
    """Create tuning data without exposing the fixed outer test partition.

    All real-data loaders carry an official or deterministic outer split.
    Tuning draws a development subset only from the outer training partition;
    the complete split-aware array is returned for repeated final evaluation.
    The plain-array fallback remains available to non-benchmark callers.
    """
    predefined = getattr(X, "predefined_split", None)
    X_array = np.asarray(X)
    y = np.asarray(y)
    if len(X_array) != len(y):
        raise ValueError("X and y must contain the same number of samples")
    if not 0.0 < development_fraction < 1.0:
        raise ValueError("development_fraction must lie strictly between 0 and 1")

    if predefined is not None:
        train_indices, _ = predefined
        train_indices = np.asarray(train_indices, dtype=np.int64)
        training_y = y[train_indices]
        local_indices = np.arange(len(train_indices))
        stratify = (
            ensure_binary_labels(training_y)
            if task_type == "classification"
            else None
        )
        development_local, _ = train_test_split(
            local_indices,
            train_size=development_fraction,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
        development_indices = train_indices[development_local]
        return (
            X_array[development_indices], y[development_indices],
            X, y,
        )

    indices = np.arange(len(y))
    stratify = ensure_binary_labels(y) if task_type == "classification" else None
    development_indices, evaluation_indices = train_test_split(
        indices,
        train_size=development_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return (
        X_array[development_indices], y[development_indices],
        X_array[evaluation_indices], y[evaluation_indices],
    )


def fold_minmax_scale_X(X_train: np.ndarray, X_val: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x_min = X_train.min(axis=0)
    x_max = X_train.max(axis=0)
    span = x_max - x_min
    span[span == 0] = 1.0
    return (X_train - x_min) / span, (X_val - x_min) / span


def fold_standardize_y(
    y_train: np.ndarray,
    y_val: np.ndarray,
    task_type: str
):
    """
    Regression: z-score based on TRAIN split only.
    Classification: return unchanged, mean/std = None.
    """
    if task_type != "regression":
        return y_train, y_val, None, None

    y_train = np.asarray(y_train, dtype=np.float32).reshape(-1)
    y_val = np.asarray(y_val, dtype=np.float32).reshape(-1)

    mean = float(np.mean(y_train))
    std = float(np.std(y_train))
    if std == 0:
        std = 1.0

    return (y_train - mean) / std, (y_val - mean) / std, mean, std


def training_target_range(y_train: np.ndarray) -> float:
    """Return the raw target range used to normalize a regression fold's RMSE."""
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    target_range = float(np.ptp(y_train))
    if target_range <= 0.0:
        raise ValueError("NRMSE is undefined when the training-target range is zero.")
    return target_range


@torch.no_grad()
def eval_for_early_stop(
    model,
    loader,
    task_type: str,
    device: torch.device
) -> float:
    """
    Scalar metric for early stopping / Optuna objective.
    Regression: RMSE on standardized y (because y in loader is standardized).
    Classification: error rate.
    """
    model.eval()
    preds, trues = [], []

    for X, y in loader:
        X = X.to(device)
        out = model(X).detach().cpu().numpy().reshape(-1)
        y_np = y.detach().cpu().numpy().reshape(-1)
        preds.append(out)
        trues.append(y_np)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    if task_type == "regression":
        return float(np.sqrt(mean_squared_error(trues, preds)))

    prob = 1.0 / (1.0 + np.exp(-np.clip(preds, -50, 50)))
    y_pred = (prob > 0.5).astype(np.int64)
    return float(1.0 - accuracy_score(trues.astype(np.int64), y_pred))


@torch.no_grad()
def eval_regression_raw_metrics(
    model,
    loader,
    device: torch.device,
    y_mean: float,
    y_std: float,
    y_range: float,
) -> Tuple[float, float, float]:
    """
    Regression final reporting:
    inverse-transform standardized predictions back to RAW y scale,
    compute RMSE, range-normalized RMSE, and MAE. The normalization denominator
    is the raw target range of the corresponding outer training partition,
    supplied explicitly to prevent leakage from the test partition.
    """
    model.eval()
    preds, trues = [], []

    for X, y in loader:
        X = X.to(device)
        out = model(X).detach().cpu().numpy().reshape(-1)
        y_np = y.detach().cpu().numpy().reshape(-1)
        preds.append(out)
        trues.append(y_np)

    preds_std = np.concatenate(preds, axis=0)
    trues_std = np.concatenate(trues, axis=0)

    preds_raw = preds_std * y_std + y_mean
    trues_raw = trues_std * y_std + y_mean

    rmse = float(np.sqrt(mean_squared_error(trues_raw, preds_raw)))
    if y_range <= 0.0:
        raise ValueError("y_range must be positive.")
    nrmse = float(rmse / y_range)
    mae = float(mean_absolute_error(trues_raw, preds_raw))
    return rmse, nrmse, mae


@torch.no_grad()
def eval_classification_metrics(
    model,
    loader,
    device: torch.device,
) -> Tuple[float, float]:
    """Return error rate and AUROC from raw binary-classification logits."""
    model.eval()
    logits, trues = [], []
    for X, y in loader:
        output = model(X.to(device)).detach().cpu().numpy().reshape(-1)
        logits.append(output)
        trues.append(y.detach().cpu().numpy().reshape(-1))

    logits = np.concatenate(logits, axis=0)
    trues = ensure_binary_labels(np.concatenate(trues, axis=0)).astype(np.int64)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -50, 50)))
    predictions = (probabilities > 0.5).astype(np.int64)
    error_rate = float(1.0 - accuracy_score(trues, predictions))
    auroc = float(roc_auc_score(trues, probabilities))
    return error_rate, auroc


def summarize_predictive_metrics(
    task_type: str,
    first_scores,
    second_scores,
) -> Tuple[str, float, float, str, float, float]:
    """Summarize primary and secondary metrics under the shared CSV schema.

    Cross-validation functions return ``(MAE, NRMSE)`` for regression and
    ``(error rate, AUROC)`` for classification. NRMSE and error rate remain the
    primary metrics used by the paper's rank-based statistical analysis.
    """
    first = np.asarray(first_scores, dtype=np.float64)
    second = np.asarray(second_scores, dtype=np.float64)
    if first.size == 0 or second.size == 0:
        raise ValueError("Predictive metric score lists must be non-empty.")
    if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
        raise ValueError("Predictive metric scores must be finite.")

    if task_type == "regression":
        primary_name, primary_values = "NRMSE", second
        secondary_name, secondary_values = "MAE", first
    elif task_type == "classification":
        primary_name, primary_values = "Error Rate", first
        secondary_name, secondary_values = "AUROC", second
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return (
        primary_name,
        float(np.mean(primary_values)),
        float(np.std(primary_values)),
        secondary_name,
        float(np.mean(secondary_values)),
        float(np.std(secondary_values)),
    )
