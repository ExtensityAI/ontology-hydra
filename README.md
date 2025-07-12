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
git clone git@github.com:ExtensityAI/ontology-hydra.git
cd ontology-hydra
```

To set up the environment, install the Python package manager [uv](https://github.com/astral-sh/uv).

Then, create a virtual environment and install the dependencies by running:

```bash
uv sync
```

Now, you need to configure your `symbolicai` config. First, run:
```bash
uv run symconfig
```

Upon running this command for the first time, it will start the initial packages caching and initializing the `symbolicai` configuration files in the `uv`'s `.venv` directory, ultimately displaying the following warning:
```text
UserWarning: No configuration file found for the environment. A new configuration file has been created at <full-path>/ontology-hydra/.venv/.symai/symai.config.json. Please configure your environment.
```

You then must edit the `symai.config.json` file. A neurosymbolic engine is **required** for the `symbolicai` framework to be used. More about configuration management [here](https://extensityai.gitbook.io/symbolicai/installation#configuration-file).

Once you've set up the `symbolicai` config, you can must also installed an additional plugin for the `ontopipe` package:
```bash
uv run sympkg i ExtensityAI/chonkie-symai
```

Now, you are set up.

## Usage

### Generating Ontologies and Knowledge Graphs

You can use the ontopipe API to generate ontologies and knowledge graphs for a specific domain:

```python
from pathlib import Path

from symai import Import, Symbol

from ontopipe import generate_kg, ontopipe
from ontopipe.models import KG, Ontology

# Define the domain and output directory
domain = "fiction"
cache_path = Path("cache")

# Generate ontology
ontology = ontopipe(domain, cache_path=cache_path) # saves to cache_path / 'ontology.json'
# or load from cache
# ontology = Ontology.from_json_file(cache_path / 'ontology.json')

texts = ['...'] # provide your list of texts chunks here
                # the chunk length has an impact on the quality of the generated KG
                # shorter chunks, denser KG
                # longer chunks, sparser KG

# We also provide functionality to easily chunk text appropriately to your needs.
# We built on top of the chonkie library.
# E.g.:
# ex_str = Symbol('this is a test string to generate a knowledge graph')
# ChonkieChunker = Import.load_expression(
#     'ExtensityAI/chonkie-symai',
#     'ChonkieChunker'
# )
# chonkie = ChonkieChunker(tokenizer_name='gpt2')
# texts = chonkie(ex_str, chunk_size=...)

kg = generate_kg(
    texts=texts,
    ontology=ontology,
    cache_path=cache_path,
    kg_name='test_kg',
    epochs=1 # iterates multiple times over the texts to improve the KG
)
# or load from cache
# kg = KG.from_json_file(cache_path / 'kg.json')

from ontopipe.vis import visualize_kg, visualize_ontology

visualize_ontology(ontology, output_html_path=cache_path / 'ontology_vis.html')
visualize_kg(kg, output_html_path=cache_path / 'kg_vis.html')
```
