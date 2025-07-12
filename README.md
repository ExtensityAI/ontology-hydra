# **HyDRA: A Hybrid-Driven Reasoning Architecture for Verifiable Knowledge Graphs**
<img src="https://raw.githubusercontent.com/ExtensityAI/symbolicai/refs/heads/main/assets/images/banner.png">

<div align="center">

[![SymbolicAI](https://img.shields.io/badge/SymbolicAI-blue?style=for-the-badge)](https://github.com/ExtensityAI/symbolicai)
[![Paper](https://img.shields.io/badge/Paper-32758e?style=for-the-badge)](?)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-yellow?style=for-the-badge)](?)

[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/symbolicapi.svg?style=social&label=@ExtensityAI)](https://twitter.com/ExtensityAI)

</div>

---

<div align="center">
  <img src=".assets/ontology.gif" alt="Knowledge Graph Visualization" width="800"/>
</div>

## About

`HyDRA` is a framework for generating **domain-specific ontologies** and **knowledge graphs**. It employs an AI persona committee-based approach to comprehensively scope domains, generate ontological structures, and create knowledge graphs for various applications including question answering and domain exploration. `HyDRA` is built entirely on the [symbolicai](https://github.com/ExtensityAI/symbolicAI) framework. Please support the project by starring the repository.

## Setup

```bash
$ git clone <repository-url>
cd <repository-name>
```

To set up the environment, install the Python package manager [uv](https://github.com/astral-sh/uv).

Then, create a virtual environment and install the dependencies by running:

```bash
$ uv sync
```

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
