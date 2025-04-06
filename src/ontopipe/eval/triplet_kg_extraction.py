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
    type: str = Field(description="Type/category of the entity")

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

if __name__ == "__main__":
    sample_ontology = OntologySchema(
        entities=[
            # People and Organizations
            Entity(name="Person", type="PERSON"),
            Entity(name="Organization", type="ORG"),
            Entity(name="Location", type="LOC"),

            # Historical Entities
            Entity(name="Kingdom", type="KINGDOM"),
            Entity(name="Duchy", type="DUCHY"),
            Entity(name="Principality", type="PRINCIPALITY"),
            Entity(name="County", type="COUNTY"),
            Entity(name="Fief", type="FIEF"),
            Entity(name="Ruler", type="RULER"),
            Entity(name="Noble", type="NOBLE"),
            Entity(name="Dynasty", type="DYNASTY"),
            Entity(name="Battle", type="BATTLE"),
            Entity(name="Conquest", type="CONQUEST"),
            Entity(name="Crusade", type="CRUSADE"),
            Entity(name="Expedition", type="EXPEDITION"),
            Entity(name="Settlement", type="SETTLEMENT"),

            # Cultural Entities
            Entity(name="Language", type="LANGUAGE"),
            Entity(name="Dialect", type="DIALECT"),
            Entity(name="Religion", type="RELIGION"),
            Entity(name="Church", type="CHURCH"),
            Entity(name="Monastery", type="MONASTERY"),
            Entity(name="Architecture", type="ARCHITECTURE"),
            Entity(name="Tradition", type="TRADITION"),
            Entity(name="Manuscript", type="MANUSCRIPT"),
            Entity(name="Artwork", type="ARTWORK"),
            Entity(name="Tapestry", type="TAPESTRY"),
            Entity(name="Music", type="MUSIC"),

            # Ethnic Groups
            Entity(name="EthnicGroup", type="ETHNIC_GROUP"),
            Entity(name="Tribe", type="TRIBE"),

            # Legal Entities
            Entity(name="Agreement", type="AGREEMENT"),
            Entity(name="Treaty", type="TREATY"),
            Entity(name="Law", type="LAW"),
            Entity(name="CustomaryLaw", type="CUSTOMARY_LAW"),
            Entity(name="Right", type="RIGHT"),
            Entity(name="Obligation", type="OBLIGATION"),
            Entity(name="Fealty", type="FEALTY"),

            # Military Entities
            Entity(name="Army", type="ARMY"),
            Entity(name="Navy", type="NAVY"),
            Entity(name="Castle", type="CASTLE"),
            Entity(name="Fortification", type="FORTIFICATION"),
            Entity(name="MilitaryOrder", type="MILITARY_ORDER"),
            Entity(name="Mercenary", type="MERCENARY"),
            Entity(name="Knight", type="KNIGHT"),

            # Political Concepts
            Entity(name="Feudalism", type="FEUDALISM"),
            Entity(name="Vassalage", type="VASSALAGE"),

            # Time and Events
            Entity(name="Date", type="DATE"),
            Entity(name="Period", type="PERIOD"),
            Entity(name="Century", type="CENTURY"),
            Entity(name="Event", type="EVENT"),

            # Financial
            Entity(name="Payment", type="PAYMENT"),
            Entity(name="Currency", type="CURRENCY"),
            Entity(name="Tribute", type="TRIBUTE"),
            Entity(name="Booty", type="BOOTY")
        ],
        relationships=[
            # Hierarchical Relations
            Relationship(name="rules_over"),
            Relationship(name="vassal_of"),
            Relationship(name="succeeds"),
            Relationship(name="descends_from"),
            Relationship(name="belongs_to"),
            Relationship(name="swears_fealty_to"),
            Relationship(name="pays_homage_to"),

            # Organizational Relations
            Relationship(name="works_for"),
            Relationship(name="located_in"),
            Relationship(name="founded"),
            Relationship(name="owns"),
            Relationship(name="operates"),
            Relationship(name="allied_with"),
            Relationship(name="marries"),
            Relationship(name="related_to"),

            # Military Relations
            Relationship(name="conquers"),
            Relationship(name="defeats"),
            Relationship(name="defends"),
            Relationship(name="invades"),
            Relationship(name="besieges"),
            Relationship(name="pillages"),
            Relationship(name="raids"),
            Relationship(name="commands"),
            Relationship(name="serves_under"),

            # Cultural Relations
            Relationship(name="speaks"),
            Relationship(name="practices"),
            Relationship(name="builds"),
            Relationship(name="influences"),
            Relationship(name="adopts"),
            Relationship(name="creates"),
            Relationship(name="patronizes"),
            Relationship(name="commissions"),
            Relationship(name="develops"),
            Relationship(name="assimilates"),
            Relationship(name="merges_with"),

            # Migration Relations
            Relationship(name="settles_in"),
            Relationship(name="migrates_to"),
            Relationship(name="originates_from"),

            # Legal Relations
            Relationship(name="governs"),
            Relationship(name="requires"),
            Relationship(name="prohibits"),
            Relationship(name="permits"),
            Relationship(name="signs"),
            Relationship(name="establishes"),
            Relationship(name="grants"),
            Relationship(name="receives"),

            # Temporal Relations
            Relationship(name="starts_on"),
            Relationship(name="ends_on"),
            Relationship(name="occurs_during"),
            Relationship(name="precedes"),
            Relationship(name="follows"),
            Relationship(name="exists_between"),

            # Financial Relations
            Relationship(name="pays"),
            Relationship(name="collects"),
            Relationship(name="costs"),
            Relationship(name="plunders"),
            Relationship(name="taxes")
        ]
    )
    data_path = "data/squad_2.0_Normans"
    reader = FileReader()
    chunker = ChonkieChunker()
    extractor = OntologyTripletExtractor(tokenizer_name="Xenova/gpt-4o")
    sample_text = reader(data_path + "/paragraphs.txt")
    chunks = chunker(data=Symbol(sample_text[0]), chunk_size=512).value
    triplets = []
    usage = None

    with MetadataTracker() as tracker: # For gpt-* models
        for chunk in tqdm(chunks):
            input_data = TripletInput(text=chunk, ontology=sample_ontology)
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
    logger.info(f"\nAPI Usage:\n{usage}")
    logger.info("\nExtraction Completed!\n")

    # Save triplets to a text file
    output_file = data_path + "/extracted_kg.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for triplet in triplets:
            if triplet is not None:
                f.write(f"{str(triplet)}\n")
    logger.info(f"\nTriplets saved to {output_file}")
