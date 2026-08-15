# exps_UniformPWL.py

import ast
import csv
import copy
from typing import Callable, Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import optuna
from schedulefree import AdamWScheduleFree

from src.MLP import StandardMLP
from src.CertifiedMonotonicNetworks import uniformPWL_mono_reg
from src.MLP import StandardMLP

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
    eval_regression_nrmse,
    eval_classification_error_rate,
    summarize_predictive_metric,
)
from src.result_paths import experiment_result_file

GLOBAL_SEED = 42
SEARCH_EPOCHS = 20
FINAL_EPOCHS = 100
N_SPLITS = 5
N_TRIALS = 20
PATIENCE = 10
MAX_MONO_POINTS = 1000



# Task Type
def get_task_type(data_loader: Callable) -> str:
    return get_dataset_task_type(data_loader)



# Loader output unification
def load_full_data(loader: Callable) -> Tuple[np.ndarray, np.ndarray]:
    out = loader()
    if isinstance(out, tuple) and len(out) == 2:
        X, y = out
        return X, np.asarray(y)
    if isinstance(out, tuple) and len(out) == 4:
        X, y, X_test, y_test = out
        X_full = np.vstack((np.asarray(X), np.asarray(X_test)))
        y_full = np.concatenate((np.asarray(y), np.asarray(y_test)))
        return X_full, y_full
    raise ValueError(f"Unexpected loader output format from {loader.__name__}: len={len(out) if isinstance(out, tuple) else 'N/A'}")



# Dataset builder
def make_tensor_dataset(X: np.ndarray, y: np.ndarray, task_type: str) -> TensorDataset:
    if task_type == "classification":
        y = ensure_binary_labels(y)

    X_t = torch.FloatTensor(X)
    y_t = torch.FloatTensor(y).reshape(-1, 1)
    return TensorDataset(X_t, y_t)


# Model
def create_model(config: Dict, input_size: int, monotonic_indices: List[int], seed: int) -> nn.Module:
    torch.manual_seed(seed)

    hidden_sizes = config["hidden_sizes"]
    if isinstance(hidden_sizes, str):
        hidden_sizes = ast.literal_eval(hidden_sizes)


    model = StandardMLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=1,
        activation=nn.ReLU(),
        output_activation=nn.Identity(),
    )
    return model


# Monotonicity
def sample_random_in_domain(X_ref: np.ndarray, n_points: int, seed: int, device: torch.device) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    x_min = np.nanmin(X_ref, axis=0)
    x_max = np.nanmax(X_ref, axis=0)
    span = x_max - x_min
    span[span == 0] = 1.0
    X_rand = x_min + rng.rand(n_points, X_ref.shape[1]) * span
    return torch.FloatTensor(X_rand).to(device)


def safe_monotonicity_check(model: nn.Module,
                            optimizer: torch.optim.Optimizer,
                            data_tensor: torch.Tensor,
                            monotonic_indices: List[int],
                            device: torch.device) -> float:
    model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    opt_state = copy.deepcopy(optimizer.state_dict())

    try:
        score = monotonicity_check(model, optimizer, data_tensor, monotonic_indices, device)
    finally:
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(opt_state)

        # keep optimizer state tensors on device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    return float(score)


# Training
def get_criterion(task_type: str):
    return nn.MSELoss() if task_type == "regression" else nn.BCEWithLogitsLoss()


def train_model(model: nn.Module,
                optimizer: torch.optim.Optimizer,
                train_loader: DataLoader,
                val_loader: DataLoader,
                config: Dict,
                task_type: str,
                device: torch.device,
                monotonic_indices: List[int]) -> float:
    criterion = get_criterion(task_type)

    best_val = float("inf")
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
                empirical = criterion(out, y_batch)

                if len(monotonic_indices) == 0:
                    mono_loss = torch.zeros((), device=device)
                else:
                    mono_loss = uniformPWL_mono_reg(
                        model,
                        X_batch,
                        monotonic_indices,
                        b=float(config["offset"]),
                        n_points=1024,
                    )

                loss = empirical + float(config["monotonicity_weight"]) * mono_loss
                loss.backward()
                return loss

            optimizer.step(closure)

        # early stop metric
        if hasattr(optimizer, "eval"):
            optimizer.eval()
        val_metric = eval_for_early_stop(model, val_loader, task_type, device)

        if val_metric < best_val:
            best_val = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1

        if counter >= PATIENCE:
            break

    model.load_state_dict(best_state)
    return float(best_val)



# Hyperparameter Optimization
def objective(trial, X_full: np.ndarray, y_full: np.ndarray, task_type: str, monotonic_indices: List[int]) -> float:
    hidden_sizes_options = generate_layer_combinations(
        min_layers=2,
        max_layers=2,
        units=[8, 16, 32, 64]
    )

    config = {
        "lr": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
        "hidden_sizes": trial.suggest_categorical("hidden_sizes", hidden_sizes_options),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "monotonicity_weight": trial.suggest_categorical("monotonicity_weight", [1.0, 10.0, 100.0, 1000.0]),
        "offset": 0.2,
        "epochs": SEARCH_EPOCHS,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(GLOBAL_SEED)

    tr_idx, va_idx = early_stopping_split_indices(
        y_full, task_type, GLOBAL_SEED
    )

    X_tr, X_va = X_full[tr_idx], X_full[va_idx]
    y_tr, y_va = y_full[tr_idx], y_full[va_idx]

    X_tr, X_va = fold_minmax_scale_X(X_tr, X_va)
    y_tr, y_va, _, _ = fold_standardize_y(y_tr, y_va, task_type)

    train_dataset = make_tensor_dataset(X_tr, y_tr, task_type)
    val_dataset = make_tensor_dataset(X_va, y_va, task_type)

    g = torch.Generator().manual_seed(GLOBAL_SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False
    )

    model = create_model(
        config, X_full.shape[1], monotonic_indices, GLOBAL_SEED
    ).to(device)

    optimizer = AdamWScheduleFree(
        model.parameters(),
        lr=float(config["lr"]),
        warmup_steps=5
    )

    return train_model(model, optimizer, train_loader, val_loader, config, task_type, device, monotonic_indices)


def optimize_hyperparameters(X: np.ndarray, y: np.ndarray, task_type: str, monotonic_indices: List[int]) -> Dict:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED)
    )

    study.optimize(
        lambda trial: objective(trial, X, y, task_type, monotonic_indices),
        n_trials=N_TRIALS,
        n_jobs=1
    )

    best = study.best_params
    best["offset"] = 0.2
    best["epochs"] = FINAL_EPOCHS
    return best



# Cross Validation
def cross_validate(X: np.ndarray,
                   y: np.ndarray,
                   best_config: Dict,
                   task_type: str,
                   monotonic_indices: List[int],
                   n_splits: int = N_SPLITS):

    if isinstance(best_config["hidden_sizes"], str):
        best_config["hidden_sizes"] = ast.literal_eval(best_config["hidden_sizes"])

    kf = make_cv_splitter(task_type, n_splits, GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric_scores = []

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
            best_config, X.shape[1], monotonic_indices, GLOBAL_SEED + fold
        ).to(device)

        if n_params is None:
            n_params = int(count_parameters(model))

        optimizer = AdamWScheduleFree(
            model.parameters(),
            lr=float(best_config["lr"]),
            warmup_steps=5
        )

        train_model(model, optimizer, train_loader, stop_loader, best_config, task_type, device, monotonic_indices)

        if task_type == "regression":
            metric = eval_regression_nrmse(
                model, val_loader, device, y_mean, y_std, y_range
            )
        else:
            metric = eval_classification_error_rate(model, val_loader, device)
        metric_scores.append(float(metric))

        # monotonicity evaluation
        if len(monotonic_indices) == 0:
            mono_collect["random"].append(0.0)
            mono_collect["train"].append(0.0)
            mono_collect["test"].append(0.0)
        else:
            n_train_points = min(MAX_MONO_POINTS, len(X_train))
            n_val_points = min(MAX_MONO_POINTS, len(X_val))
            n_random_points = MAX_MONO_POINTS
            n_points = min(n_train_points, n_val_points)
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

    avg_mono = {k: (float(np.mean(v)), float(np.std(v))) for k, v in mono_collect.items()}

    return metric_scores, avg_mono, int(n_params)


def process_dataset(data_loader: Callable):
    X, y = load_full_data(data_loader)
    task_type = get_task_type(data_loader)
    monotonic_indices = get_reordered_monotonic_indices(data_loader.__name__)
    X_dev, y_dev, X_eval, y_eval = split_development_evaluation_data(
        X, y, task_type, GLOBAL_SEED
    )

    best_config = optimize_hyperparameters(
        X_dev, y_dev, task_type, monotonic_indices
    )

    metric_scores, mono_metrics, n_params = cross_validate(
        X_eval, y_eval, best_config, task_type, monotonic_indices
    )

    return metric_scores, mono_metrics, best_config, n_params, task_type


def main():
    set_global_seed(GLOBAL_SEED)

    dataset_loaders = get_benchmark_loaders()


    results_file = experiment_result_file(__file__, "main", "exps_UniformPWL.csv")


    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Task Type", "Metric Name",
            "Metric Mean", "Metric Std",
            "NumOfParameters", "Best Configuration",
            "Mono Random Mean", "Mono Random Std",
            "Mono Train Mean", "Mono Train Std",
            "Mono Test Mean", "Mono Test Std"
        ])

    for loader in dataset_loaders:
        print(f"\nProcessing dataset: {loader.__name__} with UniformPWL Regularization...")


        metric_scores, mono_metrics, best_config, n_params, task_type = process_dataset(loader)

        metric_name, final_mean, final_std = summarize_predictive_metric(
            task_type, metric_scores
        )

        write_results_to_csv(
            filename=results_file,
            dataset_name=loader.__name__,
            task_type=task_type,
            metric_name=metric_name,
            metric_mean=final_mean,
            metric_std=final_std,
            n_params=n_params,
            best_config=best_config,
            mono_metrics=mono_metrics
        )

        print(f"{loader.__name__} | {metric_name}: {final_mean:.4f} ± {final_std:.4f}")


if __name__ == "__main__":
    main()
