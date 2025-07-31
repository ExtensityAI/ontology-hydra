from pathlib import Path

from pydantic import Field
from symai.models import LLMDataModel

from ontopipe.ontology.models import Ontology

# ==================================================#
# ----Triplet Extraction Data Models----------------#
# ==================================================#


class Triplet(LLMDataModel):
    subject: str = Field(description="Subject entity name")
    predicate: str = Field(description="Name of the relationship")
    object: str = Field(description="Object entity name")

    def __eq__(self, other):
        if not isinstance(other, Triplet):
            return False
        return self.subject == other.subject and self.predicate == other.predicate and self.object == other.object

    def __hash__(self):
        return hash((hash(self.subject), hash(self.predicate), hash(self.object)))

    def __str__(self, indent: int = 0) -> str:
        return f"({self.subject}, {self.predicate}, {self.object})"


class KGState(LLMDataModel):
    triplets: list[Triplet] | None = Field(description="List of triplets.")


class KG(LLMDataModel):
    name: str = Field(description="The name of the KG domain.")
    triplets: list[Triplet] | None = Field(description="List of triplets.")

    @classmethod
    def from_json_file(cls, path: Path | str):
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8", errors="ignore"))


class TripletExtractorInput(LLMDataModel):
    text: str = Field(description="Text to extract triplets from.")
    ontology: Ontology | None = Field(
        default=None,
        description="Ontology schema to use for discovery. If None, no ontology constraints will be applied.",
    )
    state: KGState | None = Field(description="Existing knowledge graph state (triplets), if any.")
