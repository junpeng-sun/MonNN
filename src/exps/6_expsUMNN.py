# exps_UMNN.py

import argparse
import ast
import csv
import copy
import json
from typing import Callable, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import optuna
from schedulefree import AdamWScheduleFree

from src.UMNNModel import UMNNModel

from dataPreprocessing.loaders import (
    get_benchmark_loaders,
    get_dataset_task_type,
)

from src.utils import (
    write_results_to_csv,
    count_parameters,
    generate_layer_combinations,
    monotonicity_check,
    get_reordered_monotonic_indices
)

from src.exp_common import (
    set_global_seed,
    ensure_binary_labels,
    make_cv_splitter,
    early_stopping_split_indices,
    split_development_evaluation_data,
    fold_minmax_scale_X,
    fold_standardize_y,
    training_target_range,
    eval_for_early_stop,
    eval_regression_raw_metrics,
    eval_classification_metrics,
    summarize_predictive_metrics,
)
from src.result_paths import experiment_result_file

GLOBAL_SEED = 42
SEARCH_EPOCHS = 20
FINAL_EPOCHS = 100
N_TRIALS = 20
N_SPLITS = 5
MAX_MONO_POINTS = 1000
REPORT_MONOTONICITY = False
UMNN_LEARNING_RATES = [1e-4, 5e-4, 1e-3]
UMNN_GRADIENT_CLIP_VALUE = 1.0
UMNN_WEIGHT_DECAY = 1e-2
RESULT_COLUMNS = [
    "Dataset", "Task Type", "Metric Name",
    "Metric Mean", "Metric Std",
    "Secondary Metric Name", "Secondary Metric Mean", "Secondary Metric Std",
    "NumOfParameters", "Best Configuration"
]


class NonFiniteTrainingError(FloatingPointError):
    """Raised when UMNN training produces a non-finite tensor."""



# Task Type
def get_task_type(loader: Callable) -> str:
    return get_dataset_task_type(loader)



# Dataset
def make_tensor_dataset(X: np.ndarray, y: np.ndarray, task_type: str) -> TensorDataset:
    if task_type == "classification":
        y = ensure_binary_labels(y)

    X_t = torch.FloatTensor(np.asarray(X, dtype=np.float32))
    y_t = torch.FloatTensor(np.asarray(y, dtype=np.float32)).reshape(-1, 1)
    return TensorDataset(X_t, y_t)



# Random sampling for monotonicity check
def sample_random_in_domain(X_ref: np.ndarray, n_points: int, seed: int, device: torch.device) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    X_ref = np.asarray(X_ref)

    x_min = np.nanmin(X_ref, axis=0)
    x_max = np.nanmax(X_ref, axis=0)
    span = x_max - x_min
    span[span == 0] = 1.0

    u = rng.rand(n_points, X_ref.shape[1])
    X_rand = x_min + u * span
    return torch.FloatTensor(X_rand).to(device)



# Safe monotonicity check that restores model and optimizer state after evaluation
def safe_monotonicity_check(
    model: nn.Module,
    optimizer,
    data_tensor: torch.Tensor,
    monotonic_indices,
    device: torch.device
) -> float:
    model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    opt_state = copy.deepcopy(optimizer.state_dict())

    try:
        score = monotonicity_check(model, optimizer, data_tensor, monotonic_indices, device)
    finally:
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(opt_state)
        # move optimizer tensors back to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    return float(score)



# UMNN is evaluated on CUDA only so that device failures remain visible and
# timing results are never mixed with silent CPU retries.
def require_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "[UMNN] CUDA is required for this benchmark, but no CUDA device is available."
        )
    return torch.device("cuda")



# Model creation based on config and monotonic indices
def create_model(
    config: Dict[str, Any],
    input_size: int,
    monotonic_indices,
    seed: int,
    device: torch.device
) -> nn.Module:
    torch.manual_seed(seed)

    if len(monotonic_indices) == 0:
        raise ValueError("[UMNN] monotonic_indices is empty. Check your loaders/reorder settings.")

    mono_hidden = config["mono_hidden"]
    if isinstance(mono_hidden, str):
        mono_hidden = ast.literal_eval(mono_hidden)

    model = UMNNModel(
        input_size=input_size,
        monotonic_indices=monotonic_indices,
        mono_hidden_sizes=list(mono_hidden),
        n_integration_steps=int(config["integration_steps"]),
        output_size=1,
    )
    return model.to(device)



# Training
def get_criterion(task_type: str):
    return nn.MSELoss() if task_type == "regression" else nn.BCEWithLogitsLoss()


def train_model(
    model: nn.Module,
    optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    task_type: str,
    device: torch.device
) -> float:
    criterion = get_criterion(task_type)

    best_val = float("inf")
    patience = 10
    counter = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for _ in range(int(config["epochs"])):

        model.train()
        if hasattr(optimizer, "train"):
            optimizer.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            def closure():
                optimizer.zero_grad()
                out = model(X_batch)
                if not torch.isfinite(out).all():
                    raise NonFiniteTrainingError(
                        "UMNN produced a non-finite training prediction."
                    )
                loss = criterion(out, y_batch)
                if not torch.isfinite(loss):
                    raise NonFiniteTrainingError(
                        "UMNN produced a non-finite training loss."
                    )
                loss.backward()
                # Match the numerical safeguard used by the upstream UMNN
                # experiments while retaining the generalized UMNN model's
                # original exponential positive parameterization.
                torch.nn.utils.clip_grad_value_(
                    model.parameters(), float(config["gradient_clip_value"])
                )
                return loss

            optimizer.step(closure)

        if hasattr(optimizer, "eval"):
            optimizer.eval()
        val_metric = eval_for_early_stop(model, val_loader, task_type, device)

        if val_metric < best_val:
            best_val = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        if counter >= patience:
            break

    model.load_state_dict(best_state)
    return float(best_val)


# Optuna objective
def objective(
    trial,
    X_full: np.ndarray,
    y_full: np.ndarray,
    task_type: str,
    monotonic_indices
) -> float:

    hidden_options_mono = generate_layer_combinations(
        min_layers=1, max_layers=2, units=[8, 16, 32, 64]
    )
    config: Dict[str, Any] = {
        "lr": trial.suggest_categorical("lr", UMNN_LEARNING_RATES),
        "integration_steps": trial.suggest_categorical(
            "integration_steps", [20, 30, 40, 50]
        ),
        "mono_hidden": trial.suggest_categorical("mono_hidden", hidden_options_mono),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "epochs": SEARCH_EPOCHS,
        "gradient_clip_value": UMNN_GRADIENT_CLIP_VALUE,
        "weight_decay": UMNN_WEIGHT_DECAY,
    }

    set_global_seed(GLOBAL_SEED)

    if task_type == "classification":
        y_full = ensure_binary_labels(y_full)

    tr_idx, va_idx = early_stopping_split_indices(
        y_full, task_type, GLOBAL_SEED
    )

    X_tr, X_va = X_full[tr_idx], X_full[va_idx]
    y_tr, y_va = y_full[tr_idx], y_full[va_idx]

    X_tr, X_va = fold_minmax_scale_X(X_tr, X_va)
    y_tr, y_va, _, _ = fold_standardize_y(y_tr, y_va, task_type)

    train_ds = make_tensor_dataset(X_tr, y_tr, task_type)
    val_ds = make_tensor_dataset(X_va, y_va, task_type)

    g = torch.Generator().manual_seed(GLOBAL_SEED)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=g
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(config["batch_size"]),
        shuffle=False
    )

    device = require_cuda_device()
    model = create_model(
        config=config,
        input_size=X_full.shape[1],
        monotonic_indices=monotonic_indices,
        seed=GLOBAL_SEED,
        device=device
    )

    optimizer = AdamWScheduleFree(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
        warmup_steps=5
    )

    try:
        value = train_model(
            model, optimizer, train_loader, val_loader, config, task_type, device
        )
    except NonFiniteTrainingError as error:
        raise optuna.TrialPruned(str(error)) from error
    except ValueError as error:
        # sklearn raises ValueError when a validation prediction contains NaN
        # or infinity. Only this numerical-instability case is pruned; unrelated
        # ValueErrors remain visible as programming/data errors.
        message = str(error).lower()
        if "nan" in message or "infinity" in message or "infinite" in message:
            raise optuna.TrialPruned(
                f"UMNN produced a non-finite validation prediction: {error}"
            ) from error
        raise

    if not np.isfinite(value):
        raise optuna.TrialPruned(
            "UMNN produced a non-finite validation metric."
        )
    return float(value)


def optimize_hyperparameters(
    X: np.ndarray,
    y: np.ndarray,
    task_type: str,
    monotonic_indices,
    n_trials: int = N_TRIALS
) -> Dict[str, Any]:

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED)
    )

    study.optimize(
        lambda trial: objective(trial, X, y, task_type, monotonic_indices),
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=True
    )

    best = dict(study.best_params)
    best["epochs"] = FINAL_EPOCHS
    best["gradient_clip_value"] = UMNN_GRADIENT_CLIP_VALUE
    best["weight_decay"] = UMNN_WEIGHT_DECAY
    return best



# Cross validation
def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    best_config: Dict[str, Any],
    task_type: str,
    monotonic_indices,
    n_splits: int = N_SPLITS
) -> Tuple[list, Any, Dict[str, Tuple[float, float]], int]:

    if task_type == "classification":
        y = ensure_binary_labels(y)

    device = require_cuda_device()
    kf = make_cv_splitter(task_type, n_splits, GLOBAL_SEED)

    if task_type == "regression":
        mae_list, nrmse_list = [], []
    else:
        err_list, auroc_list = [], []

    mono_collect = {"random": [], "train": [], "test": []}
    n_params = None

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):

        set_global_seed(GLOBAL_SEED + fold)

        X_train, X_val = X[tr_idx], X[va_idx]
        y_train, y_val = y[tr_idx], y[va_idx]

        X_train, X_val = fold_minmax_scale_X(X_train, X_val)
        y_range = training_target_range(y_train) if task_type == "regression" else None
        y_train, y_val, y_mean, y_std = fold_standardize_y(y_train, y_val, task_type)
        fit_idx, stop_idx = early_stopping_split_indices(
            y_train, task_type, GLOBAL_SEED + fold
        )

        g = torch.Generator().manual_seed(GLOBAL_SEED + fold)

        train_loader = DataLoader(
            make_tensor_dataset(X_train[fit_idx], y_train[fit_idx], task_type),
            batch_size=int(best_config["batch_size"]),
            shuffle=True,
            generator=g
        )
        stop_loader = DataLoader(
            make_tensor_dataset(X_train[stop_idx], y_train[stop_idx], task_type),
            batch_size=int(best_config["batch_size"]),
            shuffle=False
        )
        val_loader = DataLoader(
            make_tensor_dataset(X_val, y_val, task_type),
            batch_size=int(best_config["batch_size"]),
            shuffle=False
        )

        model = create_model(
            config=best_config,
            input_size=X.shape[1],
            monotonic_indices=monotonic_indices,
            seed=GLOBAL_SEED + fold,
            device=device
        )
        optimizer = AdamWScheduleFree(
            model.parameters(),
            lr=float(best_config["lr"]),
            weight_decay=float(best_config["weight_decay"]),
            warmup_steps=5
        )

        if n_params is None:
            n_params = count_parameters(model)

        train_model(model, optimizer, train_loader, stop_loader, best_config, task_type, device)

        # performance
        if task_type == "regression":
            _, nrmse, mae = eval_regression_raw_metrics(
                model, val_loader, device, y_mean, y_std, y_range
            )
            mae_list.append(float(mae))
            nrmse_list.append(float(nrmse))
        else:
            err, auroc = eval_classification_metrics(model, val_loader, device)
            err_list.append(float(err))
            auroc_list.append(float(auroc))

        # Structure-based methods are monotonic by construction; the paper does not
        # report empirical Random/Train/Test audits for this family.
        if not REPORT_MONOTONICITY:
            mono_collect["random"].append(0.0)
            mono_collect["train"].append(0.0)
            mono_collect["test"].append(0.0)
            continue

        # monotonicity
        if not monotonic_indices:
            mono_collect["random"].append(0.0)
            mono_collect["train"].append(0.0)
            mono_collect["test"].append(0.0)
        else:
            n_train_points = min(MAX_MONO_POINTS, len(X_train))
            n_val_points = min(MAX_MONO_POINTS, len(X_val))
            n_random_points = MAX_MONO_POINTS
            n_points = min(n_train_points, n_val_points)

            if n_points <= 1:
                continue

            rng = np.random.RandomState(GLOBAL_SEED + fold)
            tr_s = rng.choice(len(X_train), n_train_points, replace=False)
            va_s = rng.choice(len(X_val), n_val_points, replace=False)

            train_sample = torch.FloatTensor(X_train[tr_s]).to(device)
            val_sample = torch.FloatTensor(X_val[va_s]).to(device)
            rand_sample = sample_random_in_domain(
                X_train, n_random_points, GLOBAL_SEED + fold, device
            )

            mono_collect["random"].append(
                safe_monotonicity_check(model, optimizer, rand_sample, monotonic_indices, device)
            )
            mono_collect["train"].append(
                safe_monotonicity_check(model, optimizer, train_sample, monotonic_indices, device)
            )
            mono_collect["test"].append(
                safe_monotonicity_check(model, optimizer, val_sample, monotonic_indices, device)
            )

    avg_mono = {k: (float(np.mean(v)), float(np.std(v))) if len(v) > 0 else (0.0, 0.0)
                for k, v in mono_collect.items()}

    if task_type == "regression":
        return mae_list, nrmse_list, avg_mono, int(n_params or 0)
    return err_list, auroc_list, avg_mono, int(n_params or 0)


def process_dataset(loader: Callable):
    X, y = loader()
    task_type = get_task_type(loader)
    monotonic_indices = get_reordered_monotonic_indices(loader.__name__)
    X_dev, y_dev, X_eval, y_eval = split_development_evaluation_data(
        X, y, task_type, GLOBAL_SEED
    )

    best_config = optimize_hyperparameters(
        X_dev, y_dev, task_type, monotonic_indices, n_trials=N_TRIALS
    )

    scores, nrmse_scores, mono_metrics, n_params = cross_validate(
        X_eval, y_eval, best_config, task_type, monotonic_indices, n_splits=N_SPLITS
    )

    return scores, nrmse_scores, mono_metrics, best_config, n_params, task_type


def prepare_results_file(results_file, resume: bool) -> set[str]:
    """Initialize a fresh CSV or return dataset names from a valid partial CSV."""
    if resume and results_file.exists() and results_file.stat().st_size > 0:
        with results_file.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != RESULT_COLUMNS:
                raise RuntimeError(
                    "Cannot resume UMNN: the existing CSV header does not match "
                    "the current result schema."
                )
            completed = set()
            for row in reader:
                dataset_name = row.get("Dataset", "").strip()
                if not dataset_name:
                    continue
                try:
                    config = json.loads(row["Best Configuration"])
                except (KeyError, TypeError, json.JSONDecodeError) as error:
                    raise RuntimeError(
                        "Cannot resume UMNN: an existing result row has an "
                        "invalid Best Configuration."
                    ) from error
                if (
                    config.get("gradient_clip_value") != UMNN_GRADIENT_CLIP_VALUE
                    or config.get("weight_decay") != UMNN_WEIGHT_DECAY
                    or config.get("lr") not in UMNN_LEARNING_RATES
                ):
                    raise RuntimeError(
                        "Cannot resume UMNN: existing rows were generated by "
                        "an incompatible training protocol. Run without "
                        "--resume to regenerate all UMNN results."
                    )
                completed.add(dataset_name)
        print(
            f"[UMNN] Resume enabled: preserving {len(completed)} completed "
            f"dataset result(s) in {results_file}."
        )
        return completed

    with results_file.open("w", newline="") as f:
        csv.writer(f).writerow(RESULT_COLUMNS)
    return set()


def parse_args():
    parser = argparse.ArgumentParser(description="Run the UMNN benchmarks.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Preserve the existing UMNN CSV and skip datasets that already "
            "have a result row."
        ),
    )
    return parser.parse_args()


def main(resume: bool = False):
    set_global_seed(GLOBAL_SEED)

    dataset_loaders = get_benchmark_loaders()

    results_file = experiment_result_file(__file__, "main", "exps_UMNN.csv")
    completed_datasets = prepare_results_file(results_file, resume)

    for loader in dataset_loaders:
        if loader.__name__ in completed_datasets:
            print(f"[UMNN] Skipping completed dataset: {loader.__name__}")
            continue
        print(f"\nProcessing dataset: {loader.__name__} with UMNN...")


        scores, nrmse_scores, mono_metrics, best_config, n_params, task_type = process_dataset(loader)


        if task_type == "regression":
            metric_name = "NRMSE"
            final_mean = float(np.mean(nrmse_scores))
            final_std = float(np.std(nrmse_scores))
        else:
            metric_name = "Error Rate"
            final_mean = float(np.mean(scores))
            final_std = float(np.std(scores))


        (
            metric_name,
            final_mean,
            final_std,
            secondary_metric_name,
            secondary_mean,
            secondary_std,
        ) = summarize_predictive_metrics(
            task_type, scores, nrmse_scores
        )

        write_results_to_csv(
            filename=results_file,
            dataset_name=loader.__name__,
            task_type=task_type,
            metric_name=metric_name,
            metric_mean=final_mean,
            metric_std=final_std,
            secondary_metric_name=secondary_metric_name,
            secondary_metric_mean=secondary_mean,
            secondary_metric_std=secondary_std,
            n_params=n_params,
            best_config=best_config,
            mono_metrics=None
        )


        print(f"{loader.__name__} | {metric_name}: {final_mean:.4f} ± {final_std:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(resume=args.resume)
