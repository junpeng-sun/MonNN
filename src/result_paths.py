"""Centralized filesystem locations for generated experiment artifacts.

Experiment code lives under ``src/exps`` while all generated CSV files and
figures live under the repository-level ``results`` directory. Tests and
cluster jobs may override the root with ``MONNN_RESULTS_ROOT``.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT_ENV = "MONNN_RESULTS_ROOT"


def results_root() -> Path:
    """Return the absolute root for generated results without using the CWD."""
    override = os.environ.get(RESULTS_ROOT_ENV)
    root = Path(override).expanduser() if override else REPO_ROOT / "results"
    return root.resolve()


def _safe_filename(filename: str) -> str:
    if not filename or Path(filename).name != filename:
        raise ValueError("filename must be a plain file name without directories")
    return filename


def experiment_results_dir(script_file: str, category: str) -> Path:
    """Create and return ``results/<category>/<script stem>``."""
    if category not in {"main", "lambda"}:
        raise ValueError("category must be 'main' or 'lambda'")
    directory = results_root() / category / Path(script_file).stem
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def experiment_result_file(script_file: str, category: str, filename: str) -> Path:
    """Return a result file owned by one main or sensitivity script."""
    return experiment_results_dir(script_file, category) / _safe_filename(filename)
