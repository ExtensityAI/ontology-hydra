# Ontopipe Evaluation Framework

This package provides an evaluation framework for the ontopipe system, allowing you to assess the performance of ontology and knowledge graph generation on question-answering tasks.

## Features

- **Full Evaluation**: Generate ontologies and knowledge graphs, then evaluate QA performance
- **Knowledge Graph Only**: Generate knowledge graphs without ontologies for comparison
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

Run evaluation without ontology generation to compare performance:

```bash
uv run eval kg-only --config eval/config.json
```

Or use a dedicated config file:

```bash
uv run eval kg-only --config eval/config_kg_only.json
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
      "squad_titles": ["Topic Title 1", "Topic Title 2"]
    }
  ]
}
```

### Parameters

- `id`: Unique identifier for the scenario
- `domain`: Domain description used for ontology generation. Set to `null` to skip ontology generation
- `squad_titles`: List of SQuAD dataset topic titles to evaluate

### Example Configurations

**Full evaluation with ontology:**
```json
{
  "scenarios": [{
    "id": "biomed",
    "domain": "Biomedical engineering is a multidisciplinary field...",
    "squad_titles": ["Biomedical Engineering"]
  }]
}
```

**Knowledge graph only evaluation:**
```json
{
  "scenarios": [{
    "id": "biomed_kg_only",
    "domain": null,
    "squad_titles": ["Biomedical Engineering"]
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
2. Run KG-only evaluation: `uv run eval kg-only --config eval/config.json`
3. Compare metrics in the respective `metrics.json` files

## Customization

### Batch Sizes

Modify `KG_BATCH_SIZE` and `QA_BATCH_SIZE` in `src/eval/eval.py` to adjust processing batch sizes.

### Model Configuration

Update the model used for question answering in the `_answer_questions` function.

### Visualization Options

Customize visualization parameters in `src/ontopipe/vis.py`.
