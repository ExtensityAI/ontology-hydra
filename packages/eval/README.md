# Ontopipe Evaluation Framework

This package provides an evaluation framework for the ontopipe system, allowing you to assess the performance of ontology and knowledge graph generation on question-answering tasks.

## Features

- **Full Evaluation**: Generate ontologies and knowledge graphs, then evaluate QA performance
- **Knowledge Graph Only**: Generate knowledge graphs without ontologies for comparison (set `domain: null` in config)
- **SQuAD v2 Integration**: Uses the SQuAD v2 dataset format for evaluation
- **Caching**: Intermediate results are cached to avoid recomputation
- **Visualization**: Automatic generation of HTML visualizations for ontologies and knowledge graphs

## Installation

```bash
uv sync
```

## Usage

### Full Evaluation (with Ontology)

Run a complete evaluation including ontology generation:

```bash
uv run eval new --config eval/config.json
```

### Knowledge Graph Only Evaluation (without Ontology)

To run evaluation without ontology generation, simply set `domain: null` in your configuration:

```json
{
  "scenarios": [{
    "id": "biomed_kg_only",
    "domain": null,
    "squad_titles": ["Biomedical Engineer"],
    "dataset_mode": "dev"
  }]
}
```

Then run the normal evaluation command:

```bash
uv run eval new --config eval/config_kg_only.json
```

### Resume Evaluation

Resume an interrupted evaluation run:

```bash
uv run eval resume --path eval/runs/20250101_ABC12
```

## Configuration

The evaluation framework uses JSON configuration files to define evaluation scenarios.

### Configuration Format

```json
{
  "scenarios": [
    {
      "id": "scenario_name",
      "domain": "Domain description for ontology generation (optional)",
      "squad_titles": ["Topic Title 1", "Topic Title 2"],
      "dataset_mode": "test",
      "neo4j": {
        "enabled": true,
        "uri": "bolt://localhost:7687",
        "user": "neo4j",
        "password": "ontology",
        "batch_size": 5,
        "num_iterations": 1
      }
    }
  ]
}
```

### Parameters

- `id`: Unique identifier for the scenario
- `domain`: Domain description used for ontology generation. Set to `null` to skip ontology generation
- `squad_titles`: List of SQuAD dataset topic titles to evaluate
- `dataset_mode`: Dataset to use for evaluation - `"dev"` (smaller, faster) or `"test"` (full dataset, default)
- `neo4j`: (optional) Neo4j evaluation configuration. If omitted or `enabled: false`, Neo4j evaluation is skipped.

#### Dataset Modes
- `"dev"`: Uses the development dataset (smaller, faster for testing)
- `"test"`: Uses the full test dataset (default, comprehensive evaluation)

#### Neo4j Config Options
- `enabled`: Set to `true` to enable Neo4j evaluation for this scenario
- `uri`: Neo4j connection URI (default: `bolt://localhost:7687`)
- `user`: Neo4j username (default: `neo4j`)
- `password`: Neo4j password (default: `ontology`)
- `batch_size`: Number of questions per batch (default: 5)
- `num_iterations`: Number of evaluation iterations (default: 1)

### Example Configurations

**Development mode (fast testing):**
```json
{
  "scenarios": [{
    "id": "biomed_dev",
    "domain": "Biomedical engineering...",
    "squad_titles": ["Biomedical Engineer"],
    "dataset_mode": "dev",
    "neo4j": {
      "enabled": true
    }
  }]
}
```

**Full evaluation with test dataset:**
```json
{
  "scenarios": [{
    "id": "biomed_test",
    "domain": "Biomedical engineering...",
    "squad_titles": ["Biomedical Engineer"],
    "dataset_mode": "test",
    "neo4j": {
      "enabled": true
    }
  }]
}
```

**Knowledge graph only evaluation (no ontology, no Neo4j):**
```json
{
  "scenarios": [{
    "id": "biomed_kg_only",
    "domain": null,
    "squad_titles": ["Biomedical Engineer"],
    "dataset_mode": "dev"
  }]
}
```

## Output Structure

Each evaluation run creates a directory with the following structure:

```
eval/runs/YYYYMMDD_XXXXX/
├── config.json              # Configuration used for this run
├── logs/                    # Log files
├── ontology.html           # Ontology visualization (if domain provided)
└── topics/
    └── Topic Title/
        ├── kg.html         # Knowledge graph visualization
        ├── kg.json         # Knowledge graph data
        ├── metrics.json    # Evaluation metrics
        └── qas.json        # Question-answer pairs
```

## Evaluation Metrics

The framework computes SQuAD v2 metrics including:
- Exact Match (EM)
- F1 Score
- No-Answer Probability

## Comparison Studies

To compare ontology-guided vs. ontology-free knowledge graph generation:

1. Run full evaluation: `uv run eval new --config eval/config.json`
2. Run KG-only evaluation by setting `domain: null` in your config and running: `uv run eval new --config eval/config_kg_only.json`
3. Compare metrics in the respective `metrics.json` files

## Customization

### Batch Sizes

Modify `KG_BATCH_SIZE` and `QA_BATCH_SIZE` in `src/eval/eval.py` to adjust processing batch sizes.

### Model Configuration

Update the model used for question answering in the `_answer_questions` function.

### Visualization Options

Customize visualization parameters in `src/ontopipe/vis.py`.
