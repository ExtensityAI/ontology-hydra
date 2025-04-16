# research-ontology

## Setup

To set up the environment, install the Python package manager [uv](https://github.com/astral-sh/uv).

Then, create a virtual environment and install the dependencies by running:

```bash
$ uv sync
```

## Evaluation

First, download the SQuAD v2.0 dataset from [https://rajpurkar.github.io/SQuAD-explorer/](https://rajpurkar.github.io/SQuAD-explorer/) and save it as `./eval/train-v2.0.json`.

Then, use the `new` subcommand to create a new evaluation run:

```bash
$ uv run eval new [--config <file.json>]
```

This will create a new evaluation run under `./eval/runs/<run-id>`, where logs, checkpoints and results will be saved. The `--config` parameter is optional and defines the configuration file used for this evaluation run. The configuration file defines scenarios and parameters for the evaluation. If not provided, the default configuration file `./eval/config.json` will be used.

If you want to resume an evaluation run, you can use the `resume` command:

```bash
$ uv run eval resume --path <path>
```

This will resume the evaluation from the last checkpoint. The path should be the path to the evaluation run folder, e.g. `./eval/runs/<run-id>`.
