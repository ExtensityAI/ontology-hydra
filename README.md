# research-ontology

## Setup

To set up the environment, install the Python package manager [uv](https://github.com/astral-sh/uv).

Then, create a virtual environment and install the dependencies by running:

```bash
$ uv sync
```

## Evaluation

First, download the SQuAD v2.0 dataset from [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/) and save it as `./eval/train-v2.0.json`.

Then, run the evaluation script:

```bash
$ uv run eval --scenarios sample.json
```

This will create a new evaluation run under `./eval/runs/<run-id>`, where logs, checkpoints and results will be saved. The `--scenarios` parameter denotes the path to a JSON file that contains the evaluation scenarios.

If you want to rerun the evaluation (in case you cancelled it while running), you can provide the ``--run-id` parameter to the script. This will allow you to rerun the evaluation from the last checkpoint.

```bash
$ uv run eval --run-id <run-id>
```
