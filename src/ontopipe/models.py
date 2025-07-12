from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from symai.models import LLMDataModel

# ==================================================#
# ----Ontology Generation Data Models---------------#
# ==================================================#

Characteristic = Literal[
    "functional",
    "inverseFunctional",
    "transitive",
    "symmetric",
    "asymmetric",
    "reflexive",
    "irreflexive",
]

Datatype = Literal[
    "xsd:string",
    "xsd:integer",
    "xsd:float",
    "xsd:boolean",
    "xsd:dateTime",
    "xsd:date",
    "xsd:time",
]


class UsageGuideline(LLMDataModel):
    """Represents usage guidelines for ontology elements."""

    description: str | None = Field(
        default=None,
        description="Textual description of how to use this element. Keep this concise and only provide relevant information!",
    )
    constraints: str | None = Field(
        default=None,
        description="Constraints or rules that must be followed when using this element.",
    )

    def __hash__(self):
        return hash((self.description, self.constraints))


class Class(LLMDataModel):
    """Represents an ontology class (concept/category).

    Classes define types of things in the domain, not relationships.
    Use PascalCase naming (e.g., ResearchPaper, Person).
    """

    name: str = Field(description="Name of the class (without namespace).")

    description: str = Field(description="Description of what this class represents.")
    usage_guideline: UsageGuideline | None = Field(default=None, description="Usage guidelines for this class.")


class SubClassRelation(LLMDataModel):
    """Represents a taxonomic (isA) relationship between two classes.

    Establishes class hierarchy where every instance of the subclass
    is also an instance of the superclass.
    """

    subclass: str = Field(description="The subclass (without namespace).")
    superclass: str = Field(description="The superclass (without namespace).")


class ObjectProperty(LLMDataModel):
    """Represents a relationship between instances of two classes.

    Use for modeling relationships between entities (NOT as classes).
    Always use camelCase verb phrases (e.g., hasAuthor, isPartOf).
    """

    name: str = Field(description="Name of the object property (without namespace).")
    description: str = Field(description="Description of what this object property represents.")
    domain: list[str] = Field(description="Domain classes.")
    range: list[str] = Field(description="Range classes.")
    characteristics: list[Characteristic] = Field(description="Property characteristics.")

    usage_guideline: UsageGuideline | None = Field(
        default=None, description="Usage guidelines for this object property."
    )

    def is_valid_for(self, subject_types: Iterable[str], object_types: Iterable[str]) -> bool:
        # pass in lists for both as we have subclass relations
        return any(st in self.domain for st in subject_types) and any(ot in self.range for ot in object_types)


class DataProperty(LLMDataModel):
    """Represents an attribute of class instances with literal values.

    Use for simple attributes that have literal values (strings, numbers, etc.).
    Do not use classes to represent attributes that should be data properties.
    """

    name: str = Field(description="Name of the data property (without namespace).")
    description: str = Field(description="Description of what this data property represents.")
    domain: list[str] = Field(description="Names of domain classes.")
    range: Datatype = Field(description="Datatype (e.g., xsd:string).")
    characteristics: list[Characteristic] = Field(description="Property characteristics.")

    usage_guideline: UsageGuideline | None = Field(default=None, description="Usage guidelines for this data property.")


class OntologyState(LLMDataModel):
    concepts: list[Class | SubClassRelation | ObjectProperty | DataProperty] = Field(
        description="List of the newly extracted concepts in the ontology. Only return new and unique concepts."
    )


class OWLBuilderInput(LLMDataModel):
    competency_question: list[str] = Field(
        description="A list of competency questions discovered during an interview process by the ontology engineer. Extract a list of relevant concepts."
    )
    ontology_state: OntologyState = Field(
        description="A dynamic state of the ontology that evolves with each iteration. Use this state to expand the ontology with new concepts."
    )


Concept = Class | SubClassRelation | ObjectProperty | DataProperty


class Ontology(LLMDataModel):
    name: str = Field(description="Name of the ontology (without namespace).")
    classes: list[Class] = Field(description="List of classes in the ontology.")
    subclass_relations: list[SubClassRelation] = Field(description="List of subclass relationships.")
    object_properties: list[ObjectProperty] = Field(description="List of object properties.")
    data_properties: list[DataProperty] = Field(description="List of data properties.")

    @classmethod
    def from_json_file(cls, path: Path | str):
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8", errors="ignore"))

    @property
    def root(self):
        # return the tl class that has no superclass relations
        return next(
            (
                cls
                for cls in self.classes
                if not any(relation.superclass == cls.name for relation in self.subclass_relations)
            ),
            None,
        )

    def extend(self, concepts: list[Concept]):
        for concept in concepts:
            if isinstance(concept, Class):
                self.classes.append(concept)
            elif isinstance(concept, SubClassRelation):
                self.subclass_relations.append(concept)
            elif isinstance(concept, ObjectProperty):
                self.object_properties.append(concept)
            elif isinstance(concept, DataProperty):
                self.data_properties.append(concept)

    def get_class(self, class_name: str):
        return next(
            (cls for cls in self.classes if class_name.lower() == cls.name.lower()),
            None,
        )

    def get_superclass(self, subclass_name: str):
        """Get the superclass of a given subclass name."""
        return next(
            (relation.superclass for relation in self.subclass_relations if relation.subclass == subclass_name),
            None,
        )

    def get_subclasses(self, superclass_name: str):
        """Get all subclasses of a given superclass name."""
        return [relation.subclass for relation in self.subclass_relations if relation.superclass == superclass_name]

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

        d = defaultdict(set)

        for cls in self.classes:
            d[cls.name] = {cls.name}

        for relation in self.subclass_relations:
            d[relation.subclass].add(relation.superclass)
            d[relation.subclass].add(relation.subclass)

        return d

    def clone(self):
        return self.model_copy(deep=True)


# ==================================================#
# ----Ontology Fixing Data Models0000---------------#
# ==================================================#
class Cluster(LLMDataModel):
    index: int = Field(description="The cluster's index.")
    relations: list[SubClassRelation] = Field(
        description="A list of discovered superclass-subclass relations that form a cluster."
    )


class Merge(LLMDataModel):
    indexes: list[int] = Field(description="The indices of the clusters that are being merged.")
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
    indexes: list[int] = Field(description="The indices of the clusters that are being bridged.")
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
    indexes: list[int] = Field(description="The indices of the clusters that are being pruned.")
    classes: list[str] = Field(description="A list of classes that are being pruned from the ontology.")

    @field_validator("indexes")
    @classmethod
    def is_unary(cls, v):
        if len(v) > 1:
            raise ValueError(
                f"Unary op error: Invalid number of clusters: {len(v)}. The prune operation requires exactly one cluster."
            )
        return v


class Operation(LLMDataModel):
    type: Merge | Bridge | Prune = Field(description="The type of operation to perform.")


class WeaverInput(LLMDataModel):
    ontology: Ontology = Field(
        description="The current state of the ontology that's the result of the performed chain of operations from history. At the beginning of the process, it is initialized with a predefined ontology and the history will be None."
    )
    clusters: list[Cluster] = Field(
        description="A list of clusters that are being managed by the weaver. The objective is to create only one cluster."
    )
    history: list[Operation] | None = Field(description="The history of performed operations.")


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
