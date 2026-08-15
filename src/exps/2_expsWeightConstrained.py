# exps_WeightsConstrained.py

import ast
import csv
import copy
from typing import Callable, Tuple, List, Dict, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import optuna
from schedulefree import AdamWScheduleFree

from src.WeightsConstrainedMLP import WeightsConstrainedMLP

from dataPreprocessing.loaders import (
    get_benchmark_loaders,
    get_dataset_task_type,
)

from src.utils import (
    monotonicity_check,
    get_reordered_monotonic_indices,
    write_results_to_csv,
    count_parameters,
    generate_layer_combinations
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
MAX_MONO_POINTS = 1000
REPORT_MONOTONICITY = False



# Task Type
def get_task_type(loader: Callable) -> str:
    return get_dataset_task_type(loader)



# Model
def create_model(config, input_size, monotonic_indices, seed):
    torch.manual_seed(seed)

    hidden_sizes = config["hidden_sizes"]
    if isinstance(hidden_sizes, str):
        hidden_sizes = ast.literal_eval(hidden_sizes)

    return WeightsConstrainedMLP(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=1,
        monotonic_indices=monotonic_indices,
    )



# Safe Monotonicity Check
def sample_random_in_domain(X_ref, n_points, seed, device):
    rng = np.random.RandomState(seed)
    x_min = np.nanmin(X_ref, axis=0)
    x_max = np.nanmax(X_ref, axis=0)
    span = x_max - x_min
    span[span == 0] = 1.0
    X_rand = x_min + rng.rand(n_points, X_ref.shape[1]) * span
    return torch.FloatTensor(X_rand).to(device)


def safe_monotonicity_check(model, optimizer, data_tensor, monotonic_indices, device):
    model_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    opt_state = copy.deepcopy(optimizer.state_dict())

    try:
        score = monotonicity_check(model, optimizer, data_tensor, monotonic_indices, device)
    finally:
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(opt_state)

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

    return float(score)



# Training
def train_model(model, optimizer, train_loader, val_loader,
                config, task_type, device):

    criterion = nn.MSELoss() if task_type == "regression" else nn.BCEWithLogitsLoss()

    best_val = float("inf")
    patience = 10
    counter = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for _ in range(config["epochs"]):

        model.train()
        optimizer.train()

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            def closure():
                optimizer.zero_grad()
                out = model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
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
    return best_val



# Optuna
def objective(trial, X, y, task_type, monotonic_indices):

    hidden_options = generate_layer_combinations(2, 2, [8, 16, 32, 64])

    config = {
        "lr": trial.suggest_float("lr", 1e-3, 1e-1, log=True),
        "hidden_sizes": trial.suggest_categorical("hidden_sizes", hidden_options),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128]),
        "epochs": SEARCH_EPOCHS,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_global_seed(GLOBAL_SEED)

    if task_type == "classification":
        y = ensure_binary_labels(y)

    tr_idx, va_idx = early_stopping_split_indices(y, task_type, GLOBAL_SEED)

    X_tr, X_va = X[tr_idx], X[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    X_tr, X_va = fold_minmax_scale_X(X_tr, X_va)
    y_tr, y_va, _, _ = fold_standardize_y(y_tr, y_va, task_type)

    train_ds = TensorDataset(torch.FloatTensor(X_tr),
                             torch.FloatTensor(y_tr).reshape(-1, 1))
    val_ds = TensorDataset(torch.FloatTensor(X_va),
                           torch.FloatTensor(y_va).reshape(-1, 1))

    g = torch.Generator().manual_seed(GLOBAL_SEED)

    train_loader = DataLoader(train_ds,
                              batch_size=config["batch_size"],
                              shuffle=True,
                              generator=g)

    val_loader = DataLoader(val_ds,
                            batch_size=config["batch_size"],
                            shuffle=False)

    model = create_model(config, X.shape[1], monotonic_indices, GLOBAL_SEED).to(device)
    optimizer = AdamWScheduleFree(model.parameters(),
                                  lr=config["lr"],
                                  warmup_steps=5)

    return train_model(model, optimizer, train_loader, val_loader,
                       config, task_type, device)


def optimize(X, y, task_type, monotonic_indices):

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=GLOBAL_SEED)
    )

    study.optimize(
        lambda trial: objective(trial, X, y, task_type, monotonic_indices),
        n_trials=20,
        n_jobs=1
    )

    best = study.best_params
    best["epochs"] = FINAL_EPOCHS
    return best



# Cross Validation (return avg_mono)
def cross_validate(X, y, config, task_type, monotonic_indices, n_splits=N_SPLITS):

    if task_type == "classification":
        y = ensure_binary_labels(y)

    kf = make_cv_splitter(task_type, n_splits, GLOBAL_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metric_scores = []

    mono_collect = {"random": [], "train": [], "test": []}
    n_params = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):

        set_global_seed(GLOBAL_SEED + fold)

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train, X_val = fold_minmax_scale_X(X_train, X_val)
        y_range = training_target_range(y_train) if task_type == "regression" else None
        y_train, y_val, y_mean, y_std = fold_standardize_y(y_train, y_val, task_type)
        fit_idx, stop_idx = early_stopping_split_indices(
            y_train, task_type, GLOBAL_SEED + fold
        )

        train_ds = TensorDataset(torch.FloatTensor(X_train[fit_idx]),
                                 torch.FloatTensor(y_train[fit_idx]).reshape(-1, 1))
        stop_ds = TensorDataset(torch.FloatTensor(X_train[stop_idx]),
                                torch.FloatTensor(y_train[stop_idx]).reshape(-1, 1))
        val_ds = TensorDataset(torch.FloatTensor(X_val),
                               torch.FloatTensor(y_val).reshape(-1, 1))

        g = torch.Generator().manual_seed(GLOBAL_SEED + fold)

        train_loader = DataLoader(train_ds,
                                  batch_size=config["batch_size"],
                                  shuffle=True,
                                  generator=g)

        stop_loader = DataLoader(stop_ds,
                                 batch_size=config["batch_size"],
                                 shuffle=False)

        val_loader = DataLoader(val_ds,
                                batch_size=config["batch_size"],
                                shuffle=False)

        model = create_model(
            config, X.shape[1], monotonic_indices, GLOBAL_SEED + fold
        ).to(device)

        if n_params is None:
            n_params = count_parameters(model)

        optimizer = AdamWScheduleFree(model.parameters(),
                                      lr=config["lr"],
                                      warmup_steps=5)

        train_model(model, optimizer, train_loader, stop_loader,
                    config, task_type, device)

        # performance
        if task_type == "regression":
            metric_score = eval_regression_nrmse(
                model, val_loader, device, y_mean, y_std, y_range
            )
        else:
            metric_score = eval_classification_error_rate(
                model, val_loader, device
            )
        metric_scores.append(float(metric_score))

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

    return metric_scores, avg_mono, n_params


def process_dataset(loader):

    X, y = loader()
    task_type = get_task_type(loader)

    monotonic_indices = get_reordered_monotonic_indices(loader.__name__)
    X_dev, y_dev, X_eval, y_eval = split_development_evaluation_data(
        X, y, task_type, GLOBAL_SEED
    )

    best_config = optimize(X_dev, y_dev, task_type, monotonic_indices)

    metric_scores, mono_metrics, n_params = cross_validate(
        X_eval, y_eval, best_config, task_type, monotonic_indices
    )

    return metric_scores, best_config, mono_metrics, n_params, task_type


def main():
    set_global_seed(GLOBAL_SEED)

    results_file = experiment_result_file(
        __file__, "main", "exps_WeightsConstrained.csv"
    )

    dataset_loaders = get_benchmark_loaders()

    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Task Type", "Metric Name",
            "Metric Mean", "Metric Std",
            "NumOfParameters", "Best Configuration"
        ])

    for loader in dataset_loaders:

        metric_scores, best_config, mono_metrics, n_params, task_type = process_dataset(loader)

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
            mono_metrics=None
        )

        print(f"{loader.__name__} | {metric_name}: {final_mean:.4f} ± {final_std:.4f}")


if __name__ == "__main__":
    main()
