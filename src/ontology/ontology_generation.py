import json
from enum import Enum
from pathlib import Path
from typing import List

from loguru import logger
from prompts import prompt_registry
from pydantic import Field, field_validator
from symai import Expression
from symai.components import MetadataTracker
from symai.models import LLMDataModel
from symai.strategy import contract
from tqdm import tqdm


#=========================================#
#----Data Models--------------------------#
#=========================================#
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

#=========================================#
#----Contract-----------------------------#
#=========================================#
@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=True,
    remedy_retry_params=dict(
        tries=25,
        delay=0.5,
        max_delay=15,
        jitter=0.1,
        backoff=2,
        graceful=False
    )
)
class OWLBuilder(Expression):
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self._classes = set()
        self._subclass_relations = set()
        self._object_properties = set()
        self._data_properties = set()

    @property
    def prompt(self) -> str:
        return prompt_registry.instruction("owl_builder")

    def forward(self, input: OWLBuilderInput, **kwargs) -> OntologyState:
        if self.contract_result is None:
            raise ValueError("Contract failed!")
        return self.contract_result

    def pre(self, input: OWLBuilderInput) -> bool:
        return True

    def post(self, output: OntologyState) -> bool:
        #@TODO: 3rd party validation of the ontology (something like OOPS!)
        for concept in output.concepts:
            if isinstance(concept, OwlClass):
                if concept in self._classes:
                    raise ValueError(f"You've generated a duplicate concept: {concept}. It is already defined. Please focus on new and unique concepts while taking the history into account.")
        return True

    def extend_concepts(self, concepts: list):
        for concept in concepts:
            if isinstance(concept, SubClassRelation):
                self._classes.add(concept.subclass)
                self._classes.add(concept.superclass)
                self._subclass_relations.add(concept)
            elif isinstance(concept, ObjectProperty):
                for domain in concept.domain:
                    self._classes.add(domain)
                for range in concept.range:
                    self._classes.add(range)
                self._object_properties.add(concept)
            elif isinstance(concept, DataProperty):
                for domain in concept.domain:
                    self._classes.add(domain)
                self._data_properties.add(concept)

    def get_ontology(self) -> Ontology:
        return Ontology(
            name=self.name,
            subclass_relations=self._subclass_relations,
            object_properties=self._object_properties,
            data_properties=self._data_properties
        )

    def dump_ontology(self, folder: Path, fname: str = "ontology.json"):
        ontology = self.get_ontology()
        if not folder.exists():
            folder.mkdir(parents=True)
        with open(folder / fname, "w") as f:
            json.dump(ontology.model_dump(), f, indent=4)

    def to_rdf(self, folder: Path, fname: str = "ontology.rdf"):
        raise NotImplementedError("to_rdf method not implemented")

def generate_ontology(
        cqs: list[str],
        ontology_name: str,
        folder: Path,
        fname: str = "ontology.json",
        batch_size: int = 1,
    ) -> Ontology:
    builder = OWLBuilder(name=ontology_name)

    usage = None
    state = OntologyState(concepts=None)
    concepts = []
    with MetadataTracker() as tracker:  # For gpt-* models
        for i in tqdm(range(0, len(cqs), batch_size)):
            batch_cqs = cqs[i:i+batch_size]
            input_data = OWLBuilderInput(competency_question=batch_cqs, ontology_state=state)
            try:
                new_state = builder(input=input_data)
            except Exception as e:
                logger.error(f"Error getting state update for batch: {e}")
                continue
            concepts.extend(new_state.concepts)
            builder.extend_concepts(concepts)
            state = OntologyState(concepts=concepts)
        builder.contract_perf_stats()
        usage = tracker.usage

    logger.info(f"\nAPI Usage:\n{usage}")
    builder.dump_ontology(folder, fname)
    logger.info("Ontology creation completed!")

    return builder.get_ontology()

if __name__ == "__main__":
    cqs = ["What is the capital of France?", "What is the population of New York City?"]
    ontology_name = "example_ontology"
    folder = Path("/tmp/output")
    generate_ontology(cqs, ontology_name, folder)
