import argparse
import json
import re
import shutil
from pathlib import Path
from typing import List, Optional, Union

import networkx as nx
from symai import Import, Symbol
from symai.components import FileReader

from ontopipe.kg import generate_kg
from ontopipe.models import Ontology
from ontopipe.pipe import ontopipe
from ontopipe.vis import visualize_kg, visualize_ontology


def is_supported_file(file_path: Path) -> bool:
    """
    Check if a file is of a supported type for text extraction.

    Args:
        file_path: Path to the file

    Returns:
        True if the file is supported, False otherwise
    """
    # List of supported file extensions
    supported_extensions = {
        # Text formats
        ".txt",
        ".md",
        ".rst",
        # Document formats
        ".pdf",
        ".docx",
        ".doc",
        ".rtf",
        ".odt",
        # Code formats
        ".py",
        ".java",
        ".js",
        ".ts",
        ".c",
        ".cpp",
        ".h",
        ".cs",
        ".go",
        ".rb",
        ".php",
        ".html",
        ".css",
        ".json",
        ".xml",
        ".yml",
        ".yaml",
        ".toml",
        # Other common text formats
        ".csv",
        ".tsv",
    }

    return file_path.suffix.lower() in supported_extensions


def get_all_supported_files(dir_path: Path) -> List[Path]:
    """
    Recursively get all supported files from a directory.

    Args:
        dir_path: Path to the directory

    Returns:
        List of paths to supported files
    """
    supported_files = []

    for item in dir_path.rglob("*"):
        if item.is_file() and is_supported_file(item):
            supported_files.append(item)

    return supported_files


def extract_text_from_file(file_path: Union[str, Path]) -> str:
    """
    Extracts text from a file using symai's FileReader.
    """
    reader = FileReader()
    try:
        # Ensure file_path is a string
        if isinstance(file_path, Path):
            file_path = str(file_path)

        # FileReader returns a Symbol, so we need to get its value
        return reader(file_path).value[0]
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def extract_texts_from_folder(folder_path: Union[str, Path]) -> List[str]:
    """
    Extracts text from all supported files in a folder recursively.

    Args:
        folder_path: Path to the folder

    Returns:
        List of extracted text content from each supported file
    """
    folder_path = Path(folder_path) if isinstance(folder_path, str) else folder_path
    supported_files = get_all_supported_files(folder_path)

    print(f"Found {len(supported_files)} supported files in {folder_path}")

    texts = []
    for file_path in supported_files:
        text = extract_text_from_file(file_path)
        if text:  # Only add non-empty texts
            texts.append(text)
            print(f"Extracted {len(text)} characters from {file_path}")

    return texts


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string to be used as a filename by removing/replacing invalid characters.

    Args:
        name: The string to sanitize

    Returns:
        A sanitized string that can be safely used as part of a filename
    """
    # Replace spaces and invalid characters with underscore
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", name)
    sanitized = re.sub(r"\s+", "_", sanitized)
    # Remove leading/trailing underscores and ensure it's not too long
    sanitized = sanitized.strip("_")[:100]
    return sanitized.lower()


def dump_ontology(ontology: Ontology, folder: Path, fname: str = "ontology.json"):
    if not folder.exists():
        folder.mkdir(parents=True)
    with open(folder / fname, "w") as f:
        json.dump(ontology.model_dump(), f, indent=4)
    return folder / fname


def chunk_text(text: str, chunk_size: int = 512) -> List[str]:
    """
    Chunks a large text into smaller pieces of specified size.

    Args:
        text: The text to chunk
        chunk_size: Maximum size of each chunk in tokens

    Returns:
        List of text chunks
    """
    chunker = Import.load_expression("ExtensityAI/chonkie-symai", "ChonkieChunker")(tokenizer_name="Xenova/gpt-4o")
    sym = Symbol(text)
    chunks = chunker(sym, chunk_size=chunk_size)

    # Debug the chunking result
    chunks = chunks.value
    if isinstance(chunks, str):
        # If it returns a single string instead of chunks, wrap it in a list
        return [chunks]

    # Filter out empty chunks
    chunks = [chunk for chunk in chunks if chunk.strip()]

    print(f"Created {len(chunks)} chunks from text of length {len(text)}")
    return chunks


def create_default_ontology(domain: str, folder: Path) -> Path:
    """
    Creates a default ontology for a domain and saves it to the specified file.

    Args:
        domain: Domain to create ontology for
        folder: Folder to save the ontology

    Returns:
        Path to the saved ontology file
    """
    print(f"Creating default ontology for domain: {domain}")
    cache_path = folder / "cache"
    cache_path.mkdir(parents=True, exist_ok=True)
    ontology = ontopipe(domain=domain, cache_path=cache_path)

    # Create safe filename from domain
    fname = "final_ontology.json"

    # Ensure output folder exists
    folder.mkdir(parents=True, exist_ok=True)

    # Path to the final ontology file
    target_ontology_path = folder / fname
    ontology_file_found = False

    # Copy files from cache_path to the specified folder
    for item in cache_path.iterdir():
        if item.is_file():
            if item.name == "ontology_fixed.json":
                # This is the main ontology file that needs to be renamed
                print(f"Found main ontology file: {item.name}")
                shutil.copy2(item, target_ontology_path)
                ontology_file_found = True
                print(f"Copied ontology file to {target_ontology_path}")
            elif ".json.html" in item.name:
                # Copy HTML visualization
                html_target = folder / "final_ontology.html"
                shutil.copy2(item, html_target)
                print(f"Copied HTML visualization to {html_target}")
            elif "_transformation_history.json" in item.name:
                # Copy transformation history
                history_target = folder / "final_ontology_transformation_history.json"
                shutil.copy2(item, history_target)
                print(f"Copied transformation history to {history_target}")

    if not ontology_file_found:
        # If no ontology file was found, save the ontology object directly
        print("No ontology file found in cache, saving ontology object directly")
        dump_ontology(ontology, folder=folder, fname=fname)

    print(f"Ontology saved to {target_ontology_path}")
    return target_ontology_path


def compute_ontology_and_kg(
    input_path: Union[str, Path],
    ontology_file: Optional[Path] = None,
    domain: Optional[str] = None,
    kg_name: str = "DefaultKG",
    output_path: Union[str, Path] = "output",
    threshold: float = 0.7,
    batch_size: int = 1,
    chunk_size: int = 512,
) -> nx.DiGraph:
    """
    Computes the ontology and knowledge graph from input files and returns a NetworkX DiGraph.

    Args:
        input_path: Path to file or directory to process
        ontology_file: Path to ontology JSON file (optional)
        domain: Domain to create ontology for if ontology_file not provided (optional)
        kg_name: Name for the generated knowledge graph
        output_path: Directory to save output files
        threshold: Threshold value for knowledge graph generation
        batch_size: Batch size for processing texts
        chunk_size: Maximum size of each text chunk in tokens

    Returns:
        NetworkX DiGraph representing the knowledge graph
    """
    # Convert string paths to Path objects
    input_path = Path(input_path)
    output_path = Path(output_path)
    if ontology_file and not isinstance(ontology_file, Path):
        ontology_file = Path(ontology_file)

    # Validate arguments
    if not ontology_file and not domain:
        raise ValueError("Either an ontology file or a domain name must be provided.")
    elif ontology_file and domain:
        print(f"Both ontology file and domain provided. Using ontology file: {ontology_file}")

    print(f"Processing input: {input_path}")

    # Determine if input is a file or directory and extract texts accordingly
    texts: List[str] = []
    if input_path.is_file():
        print(f"Reading single file: {input_path}")
        text = extract_text_from_file(input_path)
        if text:
            texts.append(text)
            print(f"Extracted {len(text)} characters from file")
    elif input_path.is_dir():
        print(f"Reading directory: {input_path}")
        texts = extract_texts_from_folder(input_path)
        print(f"Extracted text from {len(texts)} files")
    else:
        raise ValueError(f"Invalid input path: {input_path}. Must be a file or directory.")

    if not texts:
        raise ValueError("No valid text content was extracted from the input.")

    print(f"Total number of text documents: {len(texts)}")

    # Preprocess texts by chunking them into smaller parts
    print(f"Preprocessing texts by chunking into smaller segments (chunk size: {chunk_size} tokens)...")
    chunked_texts = []

    # Use a threshold for chunking based on approximate character count
    # Typically about 4-5 characters per token, so multiply token size by 4
    char_threshold = chunk_size * 4

    for i, text in enumerate(texts):
        print(f"Processing document {i + 1}/{len(texts)}")
        if len(text) > char_threshold:  # Only chunk texts that are large enough to need it
            chunks = chunk_text(text, chunk_size=chunk_size)
            chunked_texts.extend(chunks)
            print(f"Chunked text of length {len(text)} into {len(chunks)} parts")
        else:
            chunked_texts.append(text)

    print(f"After chunking: {len(chunked_texts)} text segments")

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Handle ontology - either load from file or create from domain
    if ontology_file and ontology_file.exists():
        try:
            with open(ontology_file, "r", encoding="utf-8") as f:
                ontology_data = json.load(f)
            Ontology.model_validate(ontology_data)
            print(f"Loaded ontology from {ontology_file}")
        except Exception as e:
            raise ValueError(f"Error loading ontology file: {e}")
    elif domain:
        # Create ontology from domain and save directly to output folder
        ontology_file = create_default_ontology(domain=domain, folder=output_path)

        Ontology.from_json_file(ontology_file)
        print(f"Created and loaded ontology for domain '{domain}' from {ontology_file}")
    else:
        # This shouldn't happen due to validation above
        raise ValueError("Either ontology_file or domain must be provided")

    try:
        print(f"Generating knowledge graph with {len(chunked_texts)} text segments...")
        print("Ontology file:", ontology_file)
        ontology = Ontology.from_json_file(ontology_file)
        visualize_ontology(ontology, output_path / "ontology_kg.html")
        kg = generate_kg(
            kg_path=output_path / "kg.json",
            texts=chunked_texts,
            kg_name=kg_name,
            ontology=ontology,
            threshold=threshold,
            batch_size=batch_size,
        )
        visualize_kg(kg, output_path / "kg.html", ontology)
    except Exception as e:
        print(f"Error generating knowledge graph: {e}")
        # print stack trace for debugging
        import traceback

        traceback.print_exc()
        raise

    # Convert KG to NetworkX DiGraph
    G = nx.DiGraph()
    if kg.triplets:
        for triplet in kg.triplets:
            G.add_edge(triplet.subject, triplet.object, label=triplet.predicate)
        print(f"Created graph with {len(G.nodes())} nodes and {len(G.edges())} edges")
    else:
        print("Warning: No triplets were generated.")

    return G


def visualize_from_files(
    kg_json_file: Optional[Path] = None,
    ontology_json_file: Optional[Path] = None,
    output_path: Union[str, Path] = "output",
) -> Optional[nx.DiGraph]:
    """
    Visualizes ontology and/or knowledge graph from existing JSON files.

    Args:
        kg_json_file: Path to knowledge graph JSON file
        ontology_json_file: Path to ontology JSON file
        output_path: Directory to save visualization files

    Returns:
        NetworkX DiGraph if kg_json_file is provided, otherwise None
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    graph = None

    # Visualize ontology if provided
    if ontology_json_file and ontology_json_file.exists():
        try:
            print(f"Loading ontology from {ontology_json_file}")
            ontology = Ontology.from_json_file(ontology_json_file)
            ontology_html = output_path / "ontology_visualization.html"
            visualize_ontology(ontology, ontology_html)
            print(f"Ontology visualization saved to {ontology_html}")
        except Exception as e:
            print(f"Error visualizing ontology: {e}")

    # Visualize knowledge graph if provided
    if kg_json_file and kg_json_file.exists():
        try:
            print(f"Loading knowledge graph from {kg_json_file}")
            with open(kg_json_file, "r") as f:
                kg_data = json.load(f)

            kg = Ontology.model_validate(kg_data)

            kg_html = output_path / "kg_visualization.html"
            visualize_kg(kg, kg_html)
            print(f"Knowledge graph visualization saved to {kg_html}")

            # Convert KG to NetworkX DiGraph for return
            graph = nx.DiGraph()
            if kg.triplets:
                for triplet in kg.triplets:
                    graph.add_edge(
                        triplet.subject,
                        triplet.object,
                        label=triplet.predicate,
                    )
                print(f"Created graph with {len(graph.nodes())} nodes and {len(graph.edges())} edges")

                # Save graph statistics as JSON
                stats = {
                    "nodes_count": len(graph.nodes()),
                    "edges_count": len(graph.edges()),
                    "nodes": list(graph.nodes()),
                    "edges": [
                        {"source": u, "target": v, "label": d.get("label", "")} for u, v, d in graph.edges(data=True)
                    ],
                }

                stats_file = output_path / "graph_statistics.json"
                with open(stats_file, "w", encoding="utf-8") as f:
                    json.dump(stats, f, indent=2)
                print(f"Graph statistics saved to {stats_file}")

                # Save the full NetworkX DiGraph as JSON
                graph_data = nx.node_link_data(graph)
                graph_file = output_path / "graph.json"
                with open(graph_file, "w", encoding="utf-8") as f:
                    json.dump(graph_data, f, indent=2)
                print(f"Full graph saved to {graph_file}")
            else:
                print("Warning: No triplets were found in the knowledge graph.")
        except Exception as e:
            print(f"Error visualizing knowledge graph: {e}")
            import traceback

            traceback.print_exc()

    return graph


def main():
    """Parse arguments and run the knowledge graph generation"""
    parser = argparse.ArgumentParser(description="Generate knowledge graph from text documents")
    parser.add_argument("--input", "-i", help="Path to input file or directory")
    parser.add_argument("--ontology", "-o", help="Path to ontology JSON file (optional)")
    parser.add_argument(
        "--domain",
        "-d",
        help="Domain to create ontology for if --ontology not provided",
    )
    parser.add_argument("--name", "-n", default="EnhancedKG", help="Name for the knowledge graph")
    parser.add_argument("--output", default="output", help="Output directory for the knowledge graph")
    parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.7,
        help="Threshold for knowledge graph generation (default: 0.7)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=1,
        help="Batch size for knowledge graph generation (default: 1)",
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=512,
        help="Maximum size of each text chunk in tokens (default: 512)",
    )

    # Add visualization-only mode arguments
    parser.add_argument(
        "--visualize-only",
        action="store_true",
        help="Only visualize existing files without regenerating",
    )
    parser.add_argument("--kg-json", help="Path to existing knowledge graph JSON file for visualization")
    parser.add_argument("--ontology-json", help="Path to existing ontology JSON file for visualization")

    args = parser.parse_args()

    # Process visualization-only mode
    if args.visualize_only:
        if not args.kg_json and not args.ontology_json:
            print("Error: In visualization-only mode, you must specify at least one of --kg-json or --ontology-json")
            return None

        kg_json_file = Path(args.kg_json) if args.kg_json else None
        ontology_json_file = Path(args.ontology_json) if args.ontology_json else None
        output_path = Path(args.output)

        return visualize_from_files(kg_json_file, ontology_json_file, output_path)

    # For regular mode, input is required
    if not args.input:
        print("Error: --input is required unless using --visualize-only mode")
        return None

    ontology_file = Path(args.ontology) if args.ontology else None

    try:
        graph = compute_ontology_and_kg(
            args.input,
            ontology_file=ontology_file,
            domain=args.domain,
            kg_name=args.name,
            output_path=args.output,
            threshold=args.threshold,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )

        # Output basic statistics
        print("\nKnowledge Graph Statistics:")
        print(f"Nodes: {len(graph.nodes())}")
        print(f"Edges: {len(graph.edges())}")
        print(f"Graph Nodes: {list(graph.nodes())}")
        print(f"Graph Edges: {list(graph.edges(data=True))}")

        # Save graph statistics as JSON
        output_path = Path(args.output)
        stats = {
            "nodes_count": len(graph.nodes()),
            "edges_count": len(graph.edges()),
            "nodes": list(graph.nodes()),
            "edges": [{"source": u, "target": v, "label": d.get("label", "")} for u, v, d in graph.edges(data=True)],
        }

        stats_file = output_path / "graph_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        print(f"Graph statistics saved to {stats_file}")

        # Save the full NetworkX DiGraph as JSON
        graph_data = nx.node_link_data(graph)
        graph_file = output_path / "graph.json"
        with open(graph_file, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2)
        print(f"Full graph saved to {graph_file}")

        return graph
    except Exception as e:
        print(f"Error generating knowledge graph: {e}")

        import traceback

        traceback.print_exc()
        raise


# Example usage
if __name__ == "__main__":
    main()
