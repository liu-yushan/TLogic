# TLogic: Temporal Logical Rules for Explainable Link Forecasting on Temporal Knowledge Graphs

This repository contains the code for the paper [TLogic: Temporal Logical Rules for Explainable Link Forecasting on Temporal Knowledge Graphs](https://arxiv.org/abs/2112.08025).


<h3> How to run </h3>

The dependencies required to run the code are specified in [`pyproject.toml`](https://github.com/liu-yushan/TLogic/blob/main/pyproject.toml). Run `poetry install` to install the dependencies from [`poetry.lock`](https://github.com/liu-yushan/TLogic/blob/main/poetry.lock). For more information about Poetry, a tool for dependency management and packaging in Python, see https://python-poetry.org/docs/.

The commands for running TLogic and recreating the results from the paper can be found in [`run.txt`](https://github.com/liu-yushan/TLogic/blob/main/mycode/run.txt).

[`demo.ipynb`](https://github.com/liu-yushan/TLogic/blob/main/mycode/demo.ipynb) contains a demonstration of the components rule learning and rule application.


<h3> Datasets </h3>

Each event in the temporal knowledge graph is written in the format `subject predicate object timestamp`, with tabs as separators.
The dataset is split into `train.txt`, `valid.txt`, and `test.txt`, where we use the same split as provided by [Han et al.](https://github.com/TemporalKGTeam/xERTE)
The files `entity2id.json`, `relation2id.json`, `ts2id.json` define the mapping of entities, relations, and timestamps to their corresponding IDs, respectively.
The file `statistics.yaml` summarizes the statistics of the dataset and is not needed for running the code.


<h3> Parameters </h3>

In `learn.py`:

`--dataset`, `-d`: str. Dataset name.

`--rule_lengths`, `-l`: int. Length(s) of rules that will be learned, e.g., `2`, `1 2 3`.

`--num_walks`, `-n`: int. Number of walks that will be extracted during rule learning.

`--transition_distr`: str. Transition distribution; either `unif` for uniform distribution or `exp` for exponentially weighted distribution.

`--num_processes`, `-p`: int. Number of processes to be run in parallel.

`--seed`, `-s`: int. Random seed for reproducibility.


In `apply.py`:

`--dataset`, `-d`: str. Dataset name.

`--test_data`: str. Data for rule application; either `test` for test set or any other string for validation set.

`--rules`, `-r`: str. Name of the rules file.

`--rule_lengths`, `-l`: int. Length(s) of rules that will be applied, e.g., `2`, `1 2 3`.

`--window`, `-w`: int. Size of the time window before the query timestamp for rule application.

`--top_k`: int. Minimum number of candidates. The rule application stops for a query if this number is reached.

`--num_processes`, `-p`: int. Number of processes to be run in parallel.


In `evaluate.py`:

`--dataset`, `-d`: str. Dataset name.

`--test_data`: str. Data for rule application; either `test` for test set or any other string for validation set.

`--candidates`, `-c`: str. Name of the candidates file.
