from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

from pydantic import Field, field_validator
from symai.models import LLMDataModel


# ==================================================#
# ----Ontology Generation Data Models---------------#
# ==================================================#
class Characteristic(LLMDataModel):
    value: str = Field(description="Property characteristic value.")

    @field_validator("value")
    @classmethod
    def validate_characteristic(cls, v):
        valid_characteristics = {
            "functional",
            "inverseFunctional",
            "transitive",
            "symmetric",
            "asymmetric",
            "reflexive",
            "irreflexive",
        }
        if v not in valid_characteristics:
            raise ValueError(
                f"Invalid characteristic: {v}. Must be one of {valid_characteristics}"
            )
        return v

    def __hash__(self):
        return hash(self.value)


class Datatype(LLMDataModel):
    value: str = Field(description="Datatype value (e.g., xsd:string).")

    @field_validator("value")
    @classmethod
    def validate_datatype(cls, v):
        valid_datatypes = {
            "xsd:string",
            "xsd:integer",
            "xsd:float",
            "xsd:boolean",
            "xsd:dateTime",
            "xsd:date",
            "xsd:time",
            "xsd:anyURI",
            "xsd:language",
            "xsd:decimal",
        }
        if v not in valid_datatypes:
            raise ValueError(f"Invalid datatype: {v}. Must be one of {valid_datatypes}")
        return v

    def __hash__(self):
        return hash(self.value)


class SubClassRelation(LLMDataModel):
    subclass: str = Field(description="The subclass (without namespace).")
    superclass: str = Field(description="The superclass (without namespace).")

    def __eq__(self, other):
        if not isinstance(other, SubClassRelation):
            return False
        return (self.subclass, self.superclass) == (other.subclass, other.superclass)

    def __hash__(self):
        return hash((self.subclass, self.superclass))


class ObjectProperty(LLMDataModel):
    name: str = Field(description="Name of the object property (without namespace).")
    domain: list[str] = Field(description="Domain classes.")
    range: list[str] = Field(description="Range classes.")
    characteristics: list[Characteristic] = Field(
        description="Property characteristics."
    )

    def is_valid_for(
        self, subject_types: Iterable[str], object_types: Iterable[str]
    ) -> bool:
        # pass in lists for both as we have subclass relations
        return any(st in self.domain for st in subject_types) and any(
            ot in self.range for ot in object_types
        )

    def __eq__(self, other):
        if not isinstance(other, ObjectProperty):
            return False
        return (
            self.name,
            tuple(self.domain),
            tuple(self.range),
            tuple(self.characteristics),
        ) == (
            other.name,
            tuple(other.domain),
            tuple(other.range),
            tuple(other.characteristics),
        )

    def __hash__(self):
        return hash(
            (
                self.name,
                tuple(self.domain),
                tuple(self.range),
                tuple(self.characteristics),
            )
        )


class DataProperty(LLMDataModel):
    name: str = Field(description="Name of the data property (without namespace).")
    domain: list[str] = Field(description="Names of domain classes.")
    range: Datatype = Field(description="Datatype (e.g., xsd:string).")
    characteristics: list[Characteristic] = Field(
        description="Property characteristics."
    )

    def __eq__(self, other):
        if not isinstance(other, DataProperty):
            return False
        return (
            self.name,
            tuple(self.domain),
            self.range,
            tuple(self.characteristics),
        ) == (
            other.name,
            tuple(other.domain),
            other.range,
            tuple(other.characteristics),
        )

    def __hash__(self):
        return hash(
            (self.name, tuple(self.domain), self.range, tuple(self.characteristics))
        )


class OntologyState(LLMDataModel):
    concepts: list[SubClassRelation | ObjectProperty | DataProperty] | None = Field(
        description="List of the newly extracted concepts in the ontology. Only return new and unique concepts."
    )


class OWLBuilderInput(LLMDataModel):
    competency_question: list[str] = Field(
        description="A list of competency questions discovered during an interview process by the ontology engineer. Extract a list of relevant concepts."
    )
    ontology_state: OntologyState = Field(
        description="A dynamic state of the ontology that evolves with each iteration. Use this state to expand the ontology with new concepts."
    )


class Ontology(LLMDataModel):
    name: str = Field(description="Name of the ontology (without namespace).")
    subclass_relations: list[SubClassRelation] = Field(
        description="List of subclass relationships."
    )
    object_properties: list[ObjectProperty] = Field(
        description="List of object properties."
    )
    data_properties: list[DataProperty] = Field(description="List of data properties.")

    @classmethod
    def from_json_file(cls, path: Path | str):
        return cls.model_validate_json(Path(path).read_text())

    def has_class(self, class_name: str):
        """Check if the ontology contains a class with the given name."""
        return (
            any(
                class_name == rel.superclass or class_name == rel.subclass
                for rel in self.subclass_relations
            )
            or any(class_name == prop.name for prop in self.object_properties)
            or any(class_name == prop.name for prop in self.data_properties)
        )

    def has_property(self, property_name: str):
        """Check if the ontology contains a property with the given name."""
        return any(
            property_name == prop.name for prop in self.object_properties
        ) or any(property_name == prop.name for prop in self.data_properties)

    def get_property(self, property_name: str):
        """Get a property by name."""
        for prop in self.object_properties:
            if prop.name == property_name:
                return prop

        for prop in self.data_properties:
            if prop.name == property_name:
                return prop
        return None

    @property
    def superclasses(self):
        """Returns a dict of class: superclasses/class itself for each class in the ontology"""

        d = defaultdict(lambda: set())

        for relation in self.subclass_relations:
            d[relation.subclass].add(relation.superclass)
            d[relation.subclass].add(relation.subclass)
        return d


# ==================================================#
# ----Ontology Fixing Data Models0000---------------#
# ==================================================#
class Cluster(LLMDataModel):
    index: int = Field(description="The cluster's index.")
    relations: list[SubClassRelation] = Field(
        description="A list of discovered superclass-subclass relations that form a cluster."
    )


class Merge(LLMDataModel):
    indexes: list[int] = Field(
        description="The indices of the clusters that are being merged."
    )
    relations: list[SubClassRelation] = Field(
        description="A list of superclass-subclass relations chosen from the existing two clusters in such a way that they merge."
    )

    @field_validator("indexes")
    @classmethod
    def is_binary(cls, v):
        if len(v) != 2:
            raise ValueError(
                f"Binary op error: Invalid number of clusters: {len(v)}. The merge operation requires exactly two clusters."
            )
        return v


class Bridge(LLMDataModel):
    indexes: list[int] = Field(
        description="The indices of the clusters that are being bridged."
    )
    relations: list[SubClassRelation] = Field(
        description="A list of new superclass-subclass relations used to bridge the two clusters from the ontology."
    )

    @field_validator("indexes")
    @classmethod
    def is_binary(cls, v):
        if len(v) != 2:
            raise ValueError(
                f"Binary op error: Invalid number of clusters: {len(v)}. The merge operation requires exactly two clusters."
            )
        return v


class Prune(LLMDataModel):
    indexes: list[int] = Field(
        description="The indices of the clusters that are being pruned."
    )
    classes: list[str] = Field(
        description="A list of classes that are being pruned from the ontology."
    )

    @field_validator("indexes")
    @classmethod
    def is_unary(cls, v):
        if len(v) > 1:
            raise ValueError(
                f"Unary op error: Invalid number of clusters: {len(v)}. The prune operation requires exactly one cluster."
            )
        return v


class Operation(LLMDataModel):
    type: Merge | Bridge | Prune = Field(
        description="The type of operation to perform."
    )


class WeaverInput(LLMDataModel):
    ontology: Ontology = Field(
        description="The current state of the ontology that's the result of the performed chain of operations from history. At the beginning of the process, it is initialized with a predefined ontology and the history will be None."
    )
    clusters: list[Cluster] = Field(
        description="A list of clusters that are being managed by the weaver. The objective is to create only one cluster."
    )
    history: list[Operation] | None = Field(
        description="The history of performed operations."
    )


# ==================================================#
# ----Triplet Extraction Data Models----------------#
# ==================================================#


class Triplet(LLMDataModel):
    subject: str = Field(description="Subject entity name")
    predicate: str = Field(description="Name of the relationship")
    object: str = Field(description="Object entity name")

    """confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the extracted triplet [0, 1]",
    )"""

    def __eq__(self, other):
        if not isinstance(other, Triplet):
            return False
        return (
            self.subject == other.subject
            and self.predicate == other.predicate
            and self.object == other.object
        )

    def __hash__(self):
        return hash((hash(self.subject), hash(self.predicate), hash(self.object)))

    def __str__(self, indent: int = 0) -> str:
        return f"({self.subject}, {self.predicate}, {self.object})"


class KGState(LLMDataModel):
    triplets: list[Triplet] | None = Field(description="List of triplets.")


class KG(LLMDataModel):
    name: str = Field(description="The name of the KG domain.")
    triplets: list[Triplet] | None = Field(description="List of triplets.")


class TripletExtractorInput(LLMDataModel):
    text: str = Field(description="Text to extract triplets from.")
    ontology: Ontology = Field(description="Ontology schema to use for discovery.")
    state: KGState | None = Field(
        description="Existing knowledge graph state (triplets), if any."
    )
