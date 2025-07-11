# Research Ontology

> An advanced tool for generating, visualizing, and evaluating domain-specific knowledge graphs and ontologies.

<div align="center">
  <img src=".assets/kg-zoom.png" alt="Knowledge Graph Visualization" width="800"/>
</div>

## About

ontopipe is a powerful framework for generating domain-specific ontologies and knowledge graphs. It uses a committee-based approach with AI personas to comprehensively scope domains, generate ontological structures, and create knowledge graphs that can be used for various applications including question answering and domain exploration.

It combines symbolic AI techniques (using the [symbolicai](https://github.com/ExtensityAI/symbolicai) framework) with large language models to create structured representations of domain knowledge that are both human-readable and machine-actionable.

## Setup

```bash
$ git clone <repository-url>
cd <repository-name>
```

To set up the environment, install the Python package manager [uv](https://github.com/astral-sh/uv).

Then, create a virtual environment and install the dependencies by running:

```bash
$ uv sync --all-packages
```

_Note: It is important to use the `--all-packages` parameter as the evaluation framework is located in a separate package under `./packages/eval`._

Now, you need to configure your `symbolicai` config. First, run:
```bash
$ uv run symconfig
```

Upon running this command for the first time, it will start the initial packages caching and initializing the `symbolicai` configuration files in the `uv`'s `.venv` directory, ultimately displaying the following warning:
```text
UserWarning: No configuration file found for the environment. A new configuration file has been created at <full-path>/<repository-name>/.venv/.symai/symai.config.json. Please configure your environment.
```

You then must edit the `symai.config.json` file. A neurosymbolic engine is **required** for the `symbolicai` framework to be used. More about configuration management [here](https://extensityai.gitbook.io/symbolicai/installation#configuration-file).

Once you've set up the `symbolicai` config, you can must also installed an additional plugin for the `ontopipe` package:
```bash
$ uv run sympkg i ExtensityAI/chonkie-symai
```

Now, you are set up.

## Usage

### Generating Ontologies and Knowledge Graphs

You can use the ontopipe API to generate ontologies and knowledge graphs for a specific domain:

```python
from ontopipe import ontopipe, generate_kg
from pathlib import Path

# Define the domain and output directory
domain = "biology"
artifacts_dir = Path("./artifacts/biology")

# Generate ontology
ontology = ontopipe(domain, cache_dir=artifacts_dir)

# Generate knowledge graph
kg = generate_kg(ontology, artifacts_dir / "data.txt")

# Visualize the ontology and knowledge graph
from ontopipe.vis import visualize_ontology, visualize_kg

visualize_ontology(ontology, artifacts_dir / "ontology_viz.html")
visualize_kg(kg, artifacts_dir / "kg_viz.html")
```

## Evaluation

The evaluation framework allows you to assess the performance of ontology and knowledge graph generation on question-answering tasks using the SQuAD v2.0 dataset format.

### Prerequisites

  **Neo4j Database**: For Neo4j-based evaluation, you need a running Neo4j instance:
   - Install Neo4j Desktop or Neo4j Community Edition
   - Start a Neo4j database on `bolt://localhost:7687`
   - Set the default password to `ontology` (or update the config file)
   - Ensure the database has admin privileges for creating run-specific databases

### Configuration

The evaluation framework uses JSON configuration files to define evaluation scenarios. Several pre-configured configs are available in the `./eval/` directory:

- `config_gpt-4o.json` - Full evaluation with GPT-4o model
- `config_gpt-4.1.json` - Full evaluation with GPT-4.1 model
- `config_gemini-2.5-pro.json` - Full evaluation with Gemini 2.5 Pro
- `config_gemini-2.5-flash.json` - Full evaluation with Gemini 2.5 Flash
- `test_config.json` - Minimal test configuration
...

#### Configuration Format

```json
{
  "model": {
    "engine": "gpt-4o",
    "api_key": "your-api-key-here"
  },
  "neo4j": {
    "enabled": true,
    "uri": "bolt://localhost:7687",
    "user": "neo4j",
    "password": "ontology",
    "batch_size": 5,
    "num_iterations": 1,
    "use_run_specific_databases": true,
    "default_database": "neo4j",
    "auto_cleanup": false
  },
  "scenarios": [
    {
      "id": "scenario_name",
      "domain": "Domain description for ontology generation (optional)",
      "squad_titles": ["Topic Title"],
      "dataset_mode": "test",
      "neo4j": {
        "enabled": true
      }
    }
  ]
}
```

#### Key Configuration Options

- **`model`**: LLM configuration (engine and API key)
- **`neo4j`**: Global Neo4j settings (can be overridden per scenario)
- **`scenarios`**: Array of evaluation scenarios
  - `id`: Unique identifier for the scenario
  - `domain`: Domain description for ontology generation (set to `null` to skip ontology generation)
  - `squad_titles`: List of SQuAD dataset topic titles to evaluate
  - `dataset_mode`: `"dev"` (smaller, faster) or `"test"` (full dataset)
  - `neo4j`: Scenario-specific Neo4j settings (optional)

### Running Evaluations

#### Start a New Evaluation

```bash
$ uv run eval new [--config <file.json>]
```

This creates a new evaluation run under `./eval/runs/<run-id>` with a unique timestamp-based ID. The `--config` parameter is optional - if not provided, the default configuration file `./eval/config.json` will be used.

**Parallel Safety**: The evaluation framework is designed to be parallel-safe. Multiple evaluation runs can execute concurrently without interference. Each run:
- Creates a unique run directory with timestamp-based ID
- Uses run-specific Neo4j databases (when enabled)
- Maintains separate logging and output files
- Can be resumed independently

#### Resume an Evaluation

```bash
$ uv run eval resume --path <path>
```

Resume an interrupted evaluation from the last checkpoint. The path should be the evaluation run folder, e.g. `./eval/runs/<run-id>`.

### Evaluation Types

#### Full Evaluation (with Ontology)

Run a complete evaluation including ontology generation:

```bash
$ uv run eval new --config eval/config_gpt-4o.json
```

#### Knowledge Graph Only Evaluation

To run evaluation without ontology generation, set `domain: null` in your configuration:

```json
{
  "scenarios": [{
    "id": "biomed_kg_only",
    "domain": null,
    "squad_titles": ["Biomedical Engineer, Clinical Psychologist"],
    "dataset_mode": "dev"
  }]
}
```

#### Neo4j-Free Evaluation

To skip Neo4j evaluation entirely, set `neo4j.enabled: false` in your configuration or omit the neo4j section.

### Output Structure

Each evaluation run creates a directory with the following structure:

```
eval/runs/YYYYMMDD_XXXXX/
├── config.json              # Configuration used for this run
├── logs/                    # Log files
├── run_metrics.json         # Overall evaluation metrics
├── run_runtime_stats.csv    # Runtime statistics
└── topics/
    └── Topic Title/
        ├── kg.html         # Knowledge graph visualization
        ├── kg.json         # Knowledge graph data
        ├── metrics.json    # Evaluation metrics
        └── qas.json        # Question-answer pairs
```

### Evaluation Metrics

The framework computes comprehensive metrics including:
- **SQuAD v2 Metrics**: Exact Match (EM), F1 Score, No-Answer Probability
- **Neo4j Metrics**: Query success rate, results rate, accuracy
- **Runtime Statistics**: Processing times, token usage, cost estimates
