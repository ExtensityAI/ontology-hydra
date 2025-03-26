from pathlib import Path
from pydantic import Field
from typing import List
from symai.models import LLMDataModel
from tqdm import tqdm
from loguru import logger
from symai import Expression, Import, Symbol
from symai.components import MetadataTracker, FileReader
from symai.strategy import contract
import copy
from chonkie import (RecursiveChunker, SDPMChunker, SemanticChunker,
                     SentenceChunker, TokenChunker)
from chonkie.embeddings.base import BaseEmbeddings
from tokenizers import Tokenizer
import json

#=========================================#
#----Chunker------------------------------#
#=========================================#
CHUNKER_MAPPING = {
    "TokenChunker": TokenChunker,
    "SentenceChunker": SentenceChunker,
    "RecursiveChunker": RecursiveChunker,
    "SemanticChunker": SemanticChunker,
    "SDPMChunker": SDPMChunker,
}

class ChonkieChunker(Expression):
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        embedding_model_name: str | BaseEmbeddings = "minishlab/potion-base-8M",
        **symai_kwargs,
    ):
        super().__init__(**symai_kwargs)
        self.tokenizer_name = tokenizer_name
        self.embedding_model_name = embedding_model_name

    def forward(self, data: Symbol[str | list[str]], chunker_name: str = "RecursiveChunker", **chunker_kwargs) -> Symbol[list[str]]:
        chunker = self._resolve_chunker(chunker_name, **chunker_kwargs)
        chunks = [self._clean_text(chunk.text) for chunk in chunker(data.value)]
        return self._to_symbol(chunks)

    def _resolve_chunker(self, chunker_name: str, **chunker_kwargs) -> TokenChunker | SentenceChunker | RecursiveChunker | SemanticChunker | SDPMChunker:
        if chunker_name in ["TokenChunker", "SentenceChunker", "RecursiveChunker"]:
            tokenizer = Tokenizer.from_pretrained(self.tokenizer_name)
            return CHUNKER_MAPPING[chunker_name](tokenizer, **chunker_kwargs)
        elif chunker_name in ["SemanticChunker", "SDPMChunker"]:
            return CHUNKER_MAPPING[chunker_name](embedding_model=self.embedding_model_name, **chunker_kwargs)
        else:
            raise ValueError(f"Chunker {chunker_name} not found. Available chunkers: {CHUNKER_MAPPING.keys()}. See docs (https://docs.chonkie.ai/getting-started/introduction) for more info.")

    def _clean_text(self, text: str) -> str:
        """Cleans text by removing problematic characters."""
        text = text.replace('\x00', '')                              # Remove null bytes (\x00)
        text = text.encode('utf-8', errors='ignore').decode('utf-8') # Replace invalid UTF-8 sequences
        return text

#=========================================#
#----Data Models--------------------------#
#=========================================#
class Entity(LLMDataModel):
    name: str = Field(description="Abstract and general type of an entity")

class Relationship(LLMDataModel):
    name: str = Field(description="Name of the relationship used to connect abstract entities")

class DynamicOntology(LLMDataModel):
    """Represents the evolving ontology as extracted incrementally."""
    competency_questions: List[str] = Field(default_factory=list, description="List of competency questions discovered so far")
    entities: List[Entity] = Field(default_factory=list, description="Unique entities identified")
    relationships: List[Relationship] = Field(default_factory=list, description="Unique relationship types identified")

class DynamicOntologyInput(LLMDataModel):
    """Input for dynamic ontology extraction over a chunk of domain text."""
    text: str = Field(..., description="A chunk of text from the domain")
    current_ontology: DynamicOntology = Field(
        default_factory=DynamicOntology,
        description="The current state of the ontology (growing over chunks)"
    )

class DynamicOntologyOutput(LLMDataModel):
    """Output containing updated ontology information."""
    new_competency_questions: list[str] | None = Field(default=None, description="Competency questions extracted from the text chunk")
    new_entities: list[Entity] | None = Field(default=None, description="New entities discovered in the text chunk")
    new_relationships: list[Relationship] | None = Field(default=None, description="New relationship types discovered in the text chunk")
    updated_ontology: DynamicOntology = Field(..., description="The updated ontology after merging new discoveries")

#=========================================#
#----Contract-----------------------------#
#=========================================#
@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=True,
    remedy_retry_params=dict(
        tries=10,
        delay=0.5,
        max_delay=15,
        jitter=0.1,
        backoff=2,
        graceful=False
    )
)
class DynamicOntologyExtractor(Expression):
    """
    Dynamically updates the evolving ontology for a domain.

    For the given text chunk and existing ontology, this contract:
      1. Extracts new competency questions (if any)
      2. Identifies any new entities and relationship types
      3. Merges them with the current ontology, avoiding duplicates.

    The approach follows the best practices of modular extraction.
    """

    def forward(self, input: DynamicOntologyInput, **kwargs) -> DynamicOntologyOutput:
        if self.contract_result is None:
            raise ValueError("Contract failed!")
        return self.contract_result

    def pre(self, input: DynamicOntologyInput) -> bool:
        return True

    def post(self, output: DynamicOntologyOutput) -> bool:
        return True

    @property
    def prompt(self) -> str:
        return (
            "You are an ontology engineer. Given a chunk of domain text and a current ontology of competency questions, "
            "entity types, and relationship types, extract any new competency questions or novel entities/relationships "
            "that should be added. Ensure that you do not include duplicates and that the new elements are clearly delineated. "
            "IMPORTANT: Only extract abstract, general entity types (e.g., 'Algorithm', 'Person', 'Organization') and avoid concrete instances (e.g., 'bubble sort', 'John Doe', 'Google')."
        )


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    ARTIFACTS = ROOT / "artifacts"
    DOMAIN = ROOT / "artifacts/domain"
    current_ontology = DynamicOntology()
    ontology_extractor = DynamicOntologyExtractor()
    reader = FileReader()
    chunker = ChonkieChunker()
    files = [file for file in DOMAIN.iterdir() if file.is_file() and file.suffix.lower() in [".pdf", ".md", ".txt"]]
    chunks = []
    for file in tqdm(files):
        sample_text = reader(str(file))
        file_chunks = chunker(data=Symbol(sample_text[0]), chunk_size=2048).value
        chunks.extend(file_chunks)
    usage = None

    with MetadataTracker() as tracker:  # For gpt-* models
        for chunk in tqdm(chunks):
            input_data = DynamicOntologyInput(text=chunk, current_ontology=current_ontology)
            try:
                result = ontology_extractor(input=input_data, temperature=0.)
                current_ontology = result.updated_ontology
                logger.info(f"Updated ontology: {current_ontology.model_dump_json(indent=2)}")
            except Exception as e:
                logger.error(f"Error processing ontology update from chunk: {chunk}")
                logger.error(f"Error message: {str(e)}")
        ontology_extractor.contract_perf_stats()
        usage = tracker.usage

    with open(ARTIFACTS / "ontology.json", "w") as f:
        json.dump(current_ontology.model_dump(), f, indent=2)

    logger.info(f"\nAPI Usage:\n{usage}")
    logger.info("Dynamic ontology extraction completed!")
