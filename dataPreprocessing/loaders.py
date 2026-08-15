import os
from glob import glob
import numpy as np
import pandas as pd
from typing import List, Optional, Callable
from sklearn.model_selection import train_test_split



# Path settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "datasets")

# Monotonicity metadata is defined once here and shared by both the loaders and
# the experiment utilities. The feature sets encode benchmark-level domain
# assumptions rather than guarantees supplied by the dataset providers. Indices
# refer to columns after dataset-specific preprocessing but before the
# monotonic-first reordering.
MONOTONIC_FEATURES = {
    "abalone": {"increasing": [4, 5, 6, 7], "decreasing": []},
    # Displacement, horsepower, and weight decrease fuel economy.
    "auto_mpg": {"increasing": [], "decreasing": [1, 2, 3]},
    # The paper constrains only average room count (RM).
    "boston_housing": {"increasing": [5], "decreasing": []},
    "compas": {"increasing": [0, 1, 2, 3], "decreasing": []},
    "era": {"increasing": [0, 1, 2, 3], "decreasing": []},
    "esl": {"increasing": [0, 1, 2, 3], "decreasing": []},
    "heart": {"increasing": [3, 4], "decreasing": []},
    "lev": {"increasing": [0, 1, 2, 3], "decreasing": []},
    "swd": {"increasing": [0, 1, 2, 4, 6, 8, 9], "decreasing": []},
    "adult": {"increasing": [0, 1], "decreasing": []},
    "default_credit": {"increasing": [0, 1, 2, 3, 4, 5], "decreasing": []},
    "blogfeedback": {"increasing": [0, 1, 2, 3], "decreasing": []},
}

DATASET_TASK_TYPES = {
    "abalone": "regression",
    "auto_mpg": "regression",
    "boston_housing": "regression",
    "compas": "classification",
    "era": "regression",
    "esl": "regression",
    "heart": "classification",
    "lev": "regression",
    "swd": "regression",
    "adult": "classification",
    "default_credit": "classification",
    "blogfeedback": "regression",
}


class PredefinedSplitArray(np.ndarray):
    """Feature array carrying an immutable train/test protocol.

    The row order is training rows followed by test rows. Experiment utilities
    inspect ``predefined_split`` before converting the object to a base ndarray.
    """

    def __new__(cls, values, train_size: int, split_name: str):
        obj = np.asarray(values, dtype=np.float32).view(cls)
        obj.predefined_split = (
            np.arange(train_size, dtype=np.int64),
            np.arange(train_size, len(obj), dtype=np.int64),
        )
        obj.split_name = split_name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.predefined_split = getattr(obj, "predefined_split", None)
        self.split_name = getattr(obj, "split_name", None)


def get_monotonic_feature_count(dataset_name: str) -> int:
    """Return the number of constrained inputs after monotonic-first ordering."""
    dataset_name = dataset_name.replace("load_", "")
    try:
        spec = MONOTONIC_FEATURES[dataset_name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {dataset_name}") from exc
    return len(spec["increasing"]) + len(spec["decreasing"])


def get_dataset_task_type(loader_or_name) -> str:
    """Return the benchmark task type for a loader function or dataset name."""
    name = loader_or_name.__name__ if callable(loader_or_name) else str(loader_or_name)
    name = name.replace("load_", "")
    try:
        return DATASET_TASK_TYPES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset: {name}") from exc


def _combine_predefined_split(X_train, y_train, X_test, y_test, split_name):
    X_train = np.asarray(X_train, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_train = np.asarray(y_train).reshape(-1)
    y_test = np.asarray(y_test).reshape(-1)
    if X_train.ndim != 2 or X_test.ndim != 2:
        raise ValueError(f"{split_name}: feature matrices must be two-dimensional")
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(f"{split_name}: train/test feature dimensions differ")
    if len(X_train) != len(y_train) or len(X_test) != len(y_test):
        raise ValueError(f"{split_name}: feature and target row counts differ")
    X = PredefinedSplitArray(
        np.concatenate([X_train, X_test], axis=0),
        train_size=len(X_train),
        split_name=split_name,
    )
    return X, np.concatenate([y_train, y_test], axis=0)


def _make_fixed_holdout_split(X, y, dataset_name, test_size=0.2, seed=42):
    """Create a deterministic outer split for data without a supplied split."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).reshape(-1)
    task_type = get_dataset_task_type(dataset_name)
    indices = np.arange(len(y))
    stratify = y if task_type == "classification" else None
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    split_kind = "stratified" if stratify is not None else "random"
    return _combine_predefined_split(
        X[train_indices], y[train_indices],
        X[test_indices], y[test_indices],
        f"{dataset_name}_{split_kind}_80_20_seed{seed}",
    )

def get_data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)

# Core loader
def load_data(
    path: str,
    mono_inc_list: List[int],
    mono_dec_list: List[int],
    target_column: str,
    preprocess_func: Optional[Callable] = None,
    strict_mono_check: bool = True,
    dataset_name: Optional[str] = None,
):

    """
    Generic data loading function with strict monotonic feature handling.
    """

    # Load
    data = pd.read_csv(get_data_path(path))

    if preprocess_func is not None:
        data = preprocess_func(data)

    data = data.dropna()

    # Split features and target
    X = data.drop(columns=[target_column]).values.astype(np.float32)
    y = data[target_column].values

    d = X.shape[1]

    # 1) Monotonic index validity check

    bad_inc = [i for i in mono_inc_list if i < 0 or i >= d]
    bad_dec = [i for i in mono_dec_list if i < 0 or i >= d]

    if strict_mono_check and (bad_inc or bad_dec):
        raise ValueError(
            f"[load_data] Monotonic index out of range for {path}: "
            f"d={d}, bad_inc={bad_inc}, bad_dec={bad_dec}. "
            f"Check preprocessing and monotonic index definitions."
        )

    # 2) Transform decreasing to increasing

    for col in mono_dec_list:
        if 0 <= col < d:
            X[:, col] = -X[:, col]

    # 3) Reorder features: monotonic first

    mono_list = list(mono_inc_list) + list(mono_dec_list)
    mono_list = [i for i in mono_list if 0 <= i < d]

    if strict_mono_check and len(mono_list) != len(set(mono_list)):
        raise ValueError(
            f"[load_data] Duplicated monotonic indices for {path}: {mono_list}"
        )

    non_mono_list = [i for i in range(d) if i not in set(mono_list)]

    new_order = mono_list + non_mono_list
    X = X[:, new_order]

    # 4) Structural consistency check

    if strict_mono_check:
        k = len(mono_list)
        assert X.shape[1] == d
        assert k <= d

    if dataset_name is not None:
        return _make_fixed_holdout_split(X, y, dataset_name)
    return X, y


# Preprocessing functions

def preprocess_abalone(data):
    data = data.copy()
    data["Sex"] = pd.Categorical(data["Sex"]).codes
    return data


def preprocess_auto_mpg(data):
    data = data.copy()
    if "car name" in data.columns:
        data = data.drop("car name", axis=1)
    return data


def preprocess_compas(data):
    data = data.copy()

    data = data[
        (data["days_b_screening_arrest"] <= 30)
        & (data["days_b_screening_arrest"] >= -30)
    ]
    data = data[data["is_recid"] != -1]
    data = data[data["c_charge_degree"] <= "O"]
    data = data[data["score_text"] != "N/A"]

    data["race"] = pd.Categorical(data["race"]).codes
    data["sex"] = pd.Categorical(data["sex"]).codes

    data = data[
        [
            "priors_count",
            "juv_fel_count",
            "juv_misd_count",
            "juv_other_count",
            "age",
            "race",
            "sex",
            "two_year_recid",
        ]
    ]

    return data


# Dataset-specific loaders

def load_abalone():
    spec = MONOTONIC_FEATURES["abalone"]
    return load_data(
        os.path.join("abalone", "abalone.csv"),
        mono_inc_list=spec["increasing"],
        mono_dec_list=spec["decreasing"],
        target_column="Rings",
        dataset_name="abalone",
        preprocess_func=preprocess_abalone,
    )


def load_auto_mpg():
    spec = MONOTONIC_FEATURES["auto_mpg"]
    return load_data(
        os.path.join("auto_mpg", "auto-mpg.csv"),
        mono_inc_list=spec["increasing"],
        mono_dec_list=spec["decreasing"],
        target_column="mpg",
        dataset_name="auto_mpg",
        preprocess_func=preprocess_auto_mpg,
    )


def load_boston_housing():
    spec = MONOTONIC_FEATURES["boston_housing"]
    return load_data(
        os.path.join("boston_housing", "BostonHousing.csv"),
        mono_inc_list=spec["increasing"],
        mono_dec_list=spec["decreasing"],
        target_column="MEDV",
        dataset_name="boston_housing",
    )


def load_compas():
    spec = MONOTONIC_FEATURES["compas"]
    return load_data(
        os.path.join("compas", "compas_scores_two_years.csv"),
        mono_inc_list=spec["increasing"],
        mono_dec_list=spec["decreasing"],
        target_column="two_year_recid",
        dataset_name="compas",
        preprocess_func=preprocess_compas,
    )


def load_era():
    spec = MONOTONIC_FEATURES["era"]
    return load_data(
        os.path.join("era", "era.csv"),
        mono_inc_list=spec["increasing"],
        mono_dec_list=spec["decreasing"],
        target_column="out1",
        dataset_name="era",
    )


def load_esl():
    spec = MONOTONIC_FEATURES["esl"]
    return load_data(
        os.path.join("esl", "esl.csv"),
        mono_inc_list=spec["increasing"],
        mono_dec_list=spec["decreasing"],
        target_column="out1",
        dataset_name="esl",
    )


def load_heart():
    spec = MONOTONIC_FEATURES["heart"]
    return load_data(
        os.path.join("heart", "heart.csv"),
        mono_inc_list=spec["increasing"],
        mono_dec_list=spec["decreasing"],
        target_column="target",
        dataset_name="heart",
    )


def load_lev():
    spec = MONOTONIC_FEATURES["lev"]
    return load_data(
        os.path.join("lev", "lev.csv"),
        mono_inc_list=spec["increasing"],
        mono_dec_list=spec["decreasing"],
        target_column="Out1",
        dataset_name="lev",
    )



def load_swd():
    spec = MONOTONIC_FEATURES["swd"]
    return load_data(
        os.path.join("swd", "swd.csv"),
        mono_inc_list=spec["increasing"],
        mono_dec_list=spec["decreasing"],
        target_column="Out1",
        dataset_name="swd",
    )


def _encode_train_test_categories(train, test, categorical_columns):
    """One-hot encode categories using training levels only."""
    train_values = train[categorical_columns].astype("string").fillna("Unknown")
    test_values = test[categorical_columns].astype("string").fillna("Unknown")
    train_cat = pd.get_dummies(
        train_values, prefix=categorical_columns, dtype=np.float32
    )
    test_cat = pd.get_dummies(
        test_values, prefix=categorical_columns, dtype=np.float32
    ).reindex(columns=train_cat.columns, fill_value=0.0)
    return train_cat, test_cat


def load_adult():
    """Load Adult with UCI's official train/test split.

    Monotonic assumptions: income likelihood is non-decreasing in education-num
    and capital-gain, holding every other encoded input fixed.
    """
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num",
        "marital-status", "occupation", "relationship", "race", "sex",
        "capital-gain", "capital-loss", "hours-per-week", "native-country",
        "income",
    ]
    read_kwargs = dict(
        header=None, names=columns, skipinitialspace=True, comment="|",
        na_values="?",
    )
    train = pd.read_csv(get_data_path(os.path.join("adult", "adult.data")), **read_kwargs)
    test = pd.read_csv(get_data_path(os.path.join("adult", "adult.test")), **read_kwargs)

    categorical = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country",
    ]
    train[categorical] = train[categorical].fillna("Unknown")
    test[categorical] = test[categorical].fillna("Unknown")
    train_cat, test_cat = _encode_train_test_categories(train, test, categorical)
    monotonic = ["education-num", "capital-gain"]
    other_numeric = ["age", "fnlwgt", "capital-loss", "hours-per-week"]
    X_train = pd.concat(
        [train[monotonic + other_numeric].reset_index(drop=True), train_cat.reset_index(drop=True)],
        axis=1,
    ).to_numpy(dtype=np.float32)
    X_test = pd.concat(
        [test[monotonic + other_numeric].reset_index(drop=True), test_cat.reset_index(drop=True)],
        axis=1,
    ).to_numpy(dtype=np.float32)
    y_train = train["income"].astype(str).str.rstrip(".").eq(">50K").to_numpy(np.float32)
    y_test = test["income"].astype(str).str.rstrip(".").eq(">50K").to_numpy(np.float32)
    return _combine_predefined_split(X_train, y_train, X_test, y_test, "adult_official")


def load_default_credit():
    """Load Default Credit with a fixed stratified 80/20 split.

    Monotonic assumptions use repayment status X6--X11: larger values indicate
    longer payment delays and therefore non-decreasing default risk.
    """
    data = pd.read_csv(get_data_path(os.path.join("default_credit", "default_credit.csv")))
    required = {"ID", "Y", *(f"X{i}" for i in range(1, 24))}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"default_credit.csv is missing columns: {sorted(missing)}")
    train, test = train_test_split(
        data, test_size=0.2, random_state=42, shuffle=True, stratify=data["Y"]
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    categorical = ["X2", "X3", "X4"]
    train_cat, test_cat = _encode_train_test_categories(train, test, categorical)
    monotonic = [f"X{i}" for i in range(6, 12)]
    other_numeric = ["X1", "X5", *(f"X{i}" for i in range(12, 24))]
    X_train = pd.concat(
        [train[monotonic + other_numeric], train_cat], axis=1
    ).to_numpy(dtype=np.float32)
    X_test = pd.concat(
        [test[monotonic + other_numeric], test_cat], axis=1
    ).to_numpy(dtype=np.float32)
    return _combine_predefined_split(
        X_train, train["Y"].to_numpy(np.float32),
        X_test, test["Y"].to_numpy(np.float32),
        "default_credit_stratified_80_20_seed42",
    )


def load_blogfeedback():
    """Load BlogFeedback with UCI's temporally disjoint train/test files.

    Monotonic assumptions use UCI features 51--54 (zero-based 50--53): total
    comments before base time and three recent-comment count summaries.
    """
    folder = get_data_path("blogfeedback")
    train = pd.read_csv(os.path.join(folder, "blogData_train.csv"), header=None)
    test_paths = sorted(glob(os.path.join(folder, "blogData_test-*.csv")))
    if len(test_paths) != 60:
        raise ValueError(f"Expected 60 BlogFeedback test files, found {len(test_paths)}")
    test = pd.concat(
        [pd.read_csv(path, header=None) for path in test_paths], ignore_index=True
    )
    if train.shape[1] != 281 or test.shape[1] != 281:
        raise ValueError(
            f"BlogFeedback must contain 280 features plus target; got "
            f"train={train.shape[1]}, test={test.shape[1]} columns"
        )
    monotonic = [50, 51, 52, 53]
    other = [index for index in range(280) if index not in monotonic]
    order = monotonic + other
    X_train = train.iloc[:, order].to_numpy(dtype=np.float32)
    X_test = test.iloc[:, order].to_numpy(dtype=np.float32)
    return _combine_predefined_split(
        X_train, train.iloc[:, 280].to_numpy(np.float32),
        X_test, test.iloc[:, 280].to_numpy(np.float32),
        "blogfeedback_official_temporal",
    )


def get_benchmark_loaders():
    """Return the 12 benchmark loaders in the paper's display order."""
    return [
        load_abalone,
        load_auto_mpg,
        load_boston_housing,
        load_compas,
        load_era,
        load_esl,
        load_heart,
        load_lev,
        load_swd,
        load_adult,
        load_default_credit,
        load_blogfeedback,
    ]
