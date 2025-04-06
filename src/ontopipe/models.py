from pydantic import Field, field_validator
from symai.models import LLMDataModel


#==================================================#
#----Ontology Generation Data Models---------------#
#==================================================#
class Characteristic(LLMDataModel):
    value: str = Field(description="Property characteristic value.")

    @field_validator('value')
    @classmethod
    def validate_characteristic(cls, v):
        valid_characteristics = {
            "functional", "inverseFunctional", "transitive",
            "symmetric", "asymmetric", "reflexive", "irreflexive"
        }
        if v not in valid_characteristics:
            raise ValueError(f"Invalid characteristic: {v}. Must be one of {valid_characteristics}")
        return v

    def __hash__(self):
        return hash(self.value)


class Datatype(LLMDataModel):
    value: str = Field(description="Datatype value (e.g., xsd:string).")

    @field_validator('value')
    @classmethod
    def validate_datatype(cls, v):
        valid_datatypes = {
            "xsd:string", "xsd:integer", "xsd:float", "xsd:boolean",
            "xsd:dateTime", "xsd:date", "xsd:time", "xsd:anyURI",
            "xsd:language", "xsd:decimal"
        }
        if v not in valid_datatypes:
            raise ValueError(f"Invalid datatype: {v}. Must be one of {valid_datatypes}")
        return v

    def __hash__(self):
        return hash(self.value)


class OwlClass(LLMDataModel):
    name: str = Field(description="Name of the class (without namespace).")

    def __eq__(self, other):
        if not isinstance(other, OwlClass):
            return False
        return (self.name,) == (other.name,)

    def __hash__(self):
        return hash((self.name,))


class SubClassRelation(LLMDataModel):
    subclass: OwlClass = Field(description="The subclass (without namespace).")
    superclass: OwlClass = Field(description="The superclass (without namespace).")

    def __eq__(self, other):
        if not isinstance(other, SubClassRelation):
            return False
        return (self.subclass, self.superclass) == (other.subclass, other.superclass)

    def __hash__(self):
        return hash((self.subclass, self.superclass))


class ObjectProperty(LLMDataModel):
    name: str = Field(description="Name of the object property (without namespace).")
    domain: list[OwlClass] = Field(description="Domain classes.")
    range: list[OwlClass] = Field(description="Range classes.")
    characteristics: list[Characteristic] = Field(description="Property characteristics.")

    def __eq__(self, other):
        if not isinstance(other, ObjectProperty):
            return False
        return (
            self.name,
            tuple(self.domain),
            tuple(self.range),
            tuple(self.characteristics)
        ) == (
            other.name,
            tuple(other.domain),
            tuple(other.range),
            tuple(other.characteristics)
        )

    def __hash__(self):
        return hash((
            self.name,
            tuple(self.domain),
            tuple(self.range),
            tuple(self.characteristics)
        ))


class DataProperty(LLMDataModel):
    name: str = Field(description="Name of the data property (without namespace).")
    domain: list[OwlClass] = Field(description="Names of domain classes.")
    range: Datatype = Field(description="Datatype (e.g., xsd:string).")
    characteristics: list[Characteristic] = Field(description="Property characteristics.")

    def __eq__(self, other):
        if not isinstance(other, DataProperty):
            return False
        return (
            self.name,
            tuple(self.domain),
            self.range,
            tuple(self.characteristics)
        ) == (
            other.name,
            tuple(other.domain),
            other.range,
            tuple(other.characteristics)
        )

    def __hash__(self):
        return hash((
            self.name,
            tuple(self.domain),
            self.range,
            tuple(self.characteristics)
        ))


class OntologyState(LLMDataModel):
    concepts: list[SubClassRelation | ObjectProperty | DataProperty] | None = Field(description="List of the newly extracted concepts in the ontology. Only return new and unique concepts.")


class OWLBuilderInput(LLMDataModel):
    competency_question: list[str] = Field(description="A list of competency questions discovered during an interview process by the ontology engineer. Extract a list of relevant concepts.")
    ontology_state: OntologyState = Field(description="A dynamic state of the ontology that evolves with each iteration. Use this state to expand the ontology with new concepts.")


class Ontology(LLMDataModel):
    name: str = Field(description="Name of the ontology (without namespace).")
    subclass_relations: list[SubClassRelation] = Field(description="List of subclass relationships.")
    object_properties: list[ObjectProperty] = Field(description="List of object properties.")
    data_properties: list[DataProperty] = Field(description="List of data properties.")

#==================================================#
#----Triplet Extraction Data Models----------------#
#==================================================#
class Entity(LLMDataModel):
    name: str = Field(description="Name of the entity.")

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Relationship(LLMDataModel):
    name: str = Field(description="Name of the relationship.")

    def __eq__(self, other):
        if not isinstance(other, Relationship):
            return False
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)


class Triplet(LLMDataModel):
    subject: Entity
    predicate: Relationship
    object: Entity
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the extracted triplet [0, 1]",
    )

    def __eq__(self, other):
        if not isinstance(other, Triplet):
            return False
        return (
            self.subject == other.subject and
            self.predicate == other.predicate and
            self.object == other.object
        )

    def __hash__(self):
        return hash((hash(self.subject), hash(self.predicate), hash(self.object)))


class KGState(LLMDataModel):
    triplets: list[Triplet] | None = Field(description="List of triplets.")


class KG(LLMDataModel):
    name: str = Field(description="The name of the KG domain.")
    triplets: list[Triplet] | None = Field(description="List of triplets.")


class TripletExtractorInput(LLMDataModel):
    text: str = Field(description="Text to extract triplets from.")
    ontology: Ontology = Field(description="Ontology schema to use for discovery.")
    state: KGState | None = Field(description="Existing knowledge graph state (triplets), if any.")
