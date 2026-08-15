# Monotonicity constraints in neural networks

This repository contains the implementations and benchmark scripts used by the
survey's empirical comparison.

## Environment

The code requires Python 3.10 or newer and the following main packages:

- PyTorch
- NumPy, pandas, and scikit-learn
- Optuna
- `schedulefree`
- `pmlayer` for HLL
- Gurobi and `gurobipy` for the optional exact certification routines


## Datasets

Each dataset is stored in its own directory under `datasets/`. The benchmark
comprises 12 real-world tabular datasets: `abalone`, `auto_mpg`,
`boston_housing`, `compas`, `era`, `esl`, `heart`, `lev`, `swd`, `adult`,
`default_credit`, and `blogfeedback`. Eight are regression tasks and four are
binary-classification tasks. Paths, task types, monotonic feature definitions,
and the ordered benchmark loader registry are centralized in
`dataPreprocessing/loaders.py`.

Adult retains the official UCI training and test files, and BlogFeedback
retains its official temporally disjoint training set and all 60 test files.
The remaining ten datasets use deterministic 80/20 outer train/test splits
with seed 42; binary-classification splits are stratified. All specified
monotonic relationships are benchmark-level domain assumptions rather than
guarantees supplied by the dataset providers. For example, Adult constrains
`education-num` and `capital-gain`; Default Credit constrains repayment-delay
status `X6`--`X11`; and BlogFeedback constrains UCI features 51--54 (the four
corresponding count summaries).

Official sources: [Adult](https://archive.ics.uci.edu/dataset/2/adult),
[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients),
and [BlogFeedback](https://archive.ics.uci.edu/dataset/304/blogfeedback).

## Reproducibility notes

- Dataset-specific monotonic feature definitions are centralized in
  `dataPreprocessing/loaders.py`.
- Decreasing constrained inputs are sign-reversed and all constrained inputs
  are reordered to the beginning of the feature matrix.
- Feature scaling and regression-target standardization are fitted from the
  outer training partition in every seeded run.
- NRMSE uses the raw target range of the corresponding training partition.
- All real datasets retain the same fixed outer test partition across five
  seeded final fits.
- PWL uses the original empirical negative-divergence penalty; MixupPWL uses
  training/random mixtures; UniformPWL draws 1,024 fresh uniform points at
  every optimization step with derivative margin 0.2.
- UMNN retains the upstream generalized model's exponential positive
  parameterization. Its learning rate is selected from `1e-4`, `5e-4`, and
  `1e-3`; Schedule-Free AdamW uses weight decay `1e-2`, and gradients are
  value-clipped to `[-1, 1]` after backpropagation for numerical stability.
- All numerical tables must be regenerated after changing any dataset metadata,
  model guarantee, metric, or split implementation.
