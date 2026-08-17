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
