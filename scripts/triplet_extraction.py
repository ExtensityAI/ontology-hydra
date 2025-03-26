import json
from pathlib import Path
from tqdm import tqdm
from loguru import logger
from pydantic import Field
from symai import Expression, Import, Symbol
from symai.components import FileReader, MetadataTracker
from symai.models import LLMDataModel
from symai.strategy import contract

from chonkie import (RecursiveChunker, SDPMChunker, SemanticChunker,
                     SentenceChunker, TokenChunker)
from chonkie.embeddings.base import BaseEmbeddings
from tokenizers import Tokenizer

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

# Data Models
class Entity(LLMDataModel):
    """Represents an entity in the ontology"""
    name: str = Field(description="Name of the entity")
    # type: str = Field(description="Type/category of the entity")

class Relationship(LLMDataModel):
    """Represents a relationship type in the ontology"""
    name: str = Field(description="Name of the relationship")

class OntologySchema(LLMDataModel):
    """Defines the ontology schema with allowed entities and relationships"""
    entities: list[Entity] = Field(description="List of valid entity types")
    relationships: list[Relationship] = Field(description="List of valid relationship types")

class TripletInput(LLMDataModel):
    """Input for triplet extraction"""
    text: str = Field(description="Text to extract triplets from")
    ontology: OntologySchema = Field(description="Ontology schema to use for extraction")

class Triplet(LLMDataModel):
    """A semantic triplet with typed entities and relationship"""
    subject: Entity
    predicate: Relationship
    object: Entity
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score for the extracted triplet [0, 1]")

class TripletOutput(LLMDataModel):
    """Collection of extracted triplets forming a knowledge graph"""
    triplets: list[Triplet] | None = Field(default=None, description="List of extracted triplets")

@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=True,
    remedy_retry_params=dict(
        tries=5,
        delay=0.5,
        max_delay=15,
        jitter=0.1,
        backoff=2,
        graceful=False
    )
)
class OntologyTripletExtractor(Expression):
    """Extracts typed triplets according to an ontology schema"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = kwargs.get('threshold', 0.7)

    def forward(self, input: TripletInput, **kwargs) -> TripletOutput:
        if self.contract_result is None:
            return TripletOutput(triplets=None)
        return self.contract_result

    def pre(self, input: TripletInput) -> bool:
        # No semantic validation for now
        return True

    def post(self, output: TripletOutput) -> bool:
        # No semantic validation for now
        if output.triplets is None:
            return True
        for triplet in output.triplets:
            if triplet.confidence < self.threshold:
                raise ValueError(f"Confidence score {triplet.confidence} has to be above threshold {self.threshold}! Extract relationships between entities that are meaningful and relevant.")
        return True

    @property
    def prompt(self) -> str:
        return (
            "You are an expert at extracting semantic relationships from text according to ontology schemas. "
            "For the given text and ontology:\n"
            "1. Identify entities matching the allowed entity types\n"
            "2. Extract relationships between entities matching the defined relationship types\n"
            "3. Assign confidence scores based on certainty of extraction\n"
            "4. Ensure all entity and relationship types conform to the ontology\n"
            "5. Do not duplicate triplets\n"
            "6. If triplets can't be found, default to None"
        )

SAMPLE_ONTOLOGY = OntologySchema(
    entities=[
        # People and Organizations
        Entity(name="Person", type="PERSON"),
        Entity(name="Organization", type="ORG"),
        Entity(name="Location", type="LOC"),

        # Legal Entities
        Entity(name="Agreement", type="AGREEMENT"),
        Entity(name="Policy", type="POLICY"),
        Entity(name="Service", type="SERVICE"),
        Entity(name="Feature", type="FEATURE"),
        Entity(name="Right", type="RIGHT"),
        Entity(name="Obligation", type="OBLIGATION"),

        # Data Related
        Entity(name="PersonalData", type="PERSONAL_DATA"),
        Entity(name="DataCategory", type="DATA_CATEGORY"),
        Entity(name="DataProcessor", type="DATA_PROCESSOR"),

        # Time and Events
        Entity(name="Date", type="DATE"),
        Entity(name="Event", type="EVENT"),

        # Financial
        Entity(name="Payment", type="PAYMENT"),
        Entity(name="Currency", type="CURRENCY")
    ],
    relationships=[
        # Organizational Relations
        Relationship(name="works_for"),
        Relationship(name="located_in"),
        Relationship(name="owns"),
        Relationship(name="operates"),

        # Legal Relations
        Relationship(name="governs"),
        Relationship(name="requires"),
        Relationship(name="prohibits"),
        Relationship(name="permits"),
        Relationship(name="provides"),

        # Data Relations
        Relationship(name="processes"),
        Relationship(name="collects"),
        Relationship(name="stores"),
        Relationship(name="shares"),
        Relationship(name="transfers"),

        # Temporal Relations
        Relationship(name="starts_on"),
        Relationship(name="ends_on"),
        Relationship(name="modified_on"),

        # Financial Relations
        Relationship(name="charges"),
        Relationship(name="pays"),
        Relationship(name="costs")
    ]
)


if __name__ == "__main__":
    ROOT = Path(__file__).parent.parent
    ARTIFACTS = ROOT / "artifacts"
    DOMAIN = ROOT / "artifacts/domain"
    if not (ARTIFACTS / "ontology.json").exists():
        logger.info("No ontology file found, using sample ontology")
        sample_ontology = SAMPLE_ONTOLOGY
    else:
        logger.info("Loading ontology from file")
        ontology = json.load(open(ARTIFACTS / "ontology.json"))
        sample_ontology = OntologySchema(entities=ontology["entities"], relationships=ontology["relationships"])

    reader = FileReader()
    chunker = ChonkieChunker()
    extractor = OntologyTripletExtractor(tokenizer_name="Xenova/gpt-4o")
    files = [file for file in DOMAIN.iterdir() if file.is_file() and file.suffix.lower() in [".pdf", ".md", ".txt"]]
    chunks = []
    for file in tqdm(files):
        sample_text = reader(str(file))
        file_chunks = chunker(data=Symbol(sample_text[0]), chunk_size=1048).value
        chunks.extend(file_chunks)
    triplets = []
    usage = None

    with MetadataTracker() as tracker: # For gpt-* models
        for chunk in tqdm(chunks):
            input_data = TripletInput( text=chunk, ontology=sample_ontology)
            try:
                result = extractor(input=input_data)
                if result.triplets is None:
                    continue
                triplets.extend(result.triplets)
            except Exception as e:
                logger.error(f"Error extracting triplets from chunk: {chunk}")
                logger.error(f"Error message: {str(e)}")
        usage = tracker.usage
        extractor.contract_perf_stats()

    for triplet in triplets:
        if triplet is None:
            continue
        logger.info(f"\n-------\n{triplet}\n-------\n")

    with open(ARTIFACTS / "kg.json", "w") as f:
        json.dump({i: triplet.model_dump() for i, triplet in enumerate(triplets)}, f, indent=2)

    logger.info(f"\nAPI Usage:\n{usage}")
    logger.info("\nExtraction Completed!\n")
