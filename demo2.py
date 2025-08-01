from argparse import ArgumentParser
from pathlib import Path

import tiktoken
from chonkie import RecursiveChunker, TokenChunker
from tokenizers import Tokenizer

from ontopipe import generate_kg, ontopipe

parser = ArgumentParser(description="Run the ontopipe pipeline to generate an ontology and knowledge graph.")
parser.add_argument(
    "--domain",
    type=str,
    required=True,
    help="The domain for which to generate the ontology.",
)
parser.add_argument(
    "-o", "--output", type=Path, required=True, help="Output directory for the generated ontology and knowledge graph."
)
parser.add_argument(
    "-i",
    "--input",
    type=Path,
    nargs="+",
    required=True,
    help="Paths to text files to process for knowledge graph generation.",
)

args = parser.parse_args()

domain = args.domain
output_path = args.output
text_paths = args.input


if not output_path.exists():
    raise ValueError(f"Output path '{output_path}' does not exist or is not a directory.")

cache_path = output_path / "cache"

ontology = ontopipe(domain, cache_path=cache_path)  # saves to cache_path / 'ontology.json'

texts = [
    tp.read_text(encoding="utf-8", errors="ignore") for tp in text_paths
]  # TODO add support for other text formats

tokenizer = tiktoken.get_encoding("o200k_base")
chunker = TokenChunker(chunk_size=2048, chunk_overlap=256, tokenizer=tokenizer)

chunks = chunker(texts)

texts = [c.text for cg in chunks for c in cg]


print(f"Generated {len(texts)} text chunks for knowledge graph generation.")

kg = generate_kg(
    texts=texts,
    ontology=ontology,
    cache_path=cache_path,
    epochs=1,  # iterates multiple times over the texts to improve the KG
)
