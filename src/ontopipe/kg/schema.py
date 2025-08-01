from datetime import date, datetime, time
from typing import Annotated, Literal

from pydantic import Field, create_model
from symai.strategy import LLMDataModel

from ontopipe.ontology.models import (
    Class,
    DataProperty,
    DataType,
    Description,
    ObjectProperty,
    Ontology,
)


class DynamicPartialEntity(LLMDataModel):
    name: str = Field(..., description="Entity name.")


class DynamicPartialKnowledgeGraph(LLMDataModel):
    # this class is the base class for dynamic partial knowledge graphs generated using `generate_kg_schema`
    pass


_data_type_to_python: dict[DataType, type] = {
    "string": str,
    "int": int,
    "float": float,
    "boolean": bool,
    "datetime": datetime,
    "date": date,
    "time": time,
}


def _generate_description(description: Description | None):
    """Generates a description string for a class or property."""

    if description is None:
        return None

    return (
        f"{description.description or 'No description provided.'}\n"
        + f"(Constraints: {description.constraints or 'No constraints provided.'})\n"
    )


# TODO in prompt, mention not to generate information already present in the current kg!
# TODO think about a property "isMistake" s.t. the model can mark outputs as mistakes if the information is not correct in case it generated something wrong!!


def _generate_property_field(prop: DataProperty | ObjectProperty):
    """Generates a Pydantic field for a property in the ontology."""

    if isinstance(prop, DataProperty):
        return (
            list[_data_type_to_python[prop.range]]
            | None,  # data properties can be None (open-world assumption), also, if they are not functional, they can have multiple values
            # TODO respect functionality
            Field(None, description=_generate_description(prop.description)),
        )

    # TODO for date, time and datetime, add format information

    elif isinstance(prop, ObjectProperty):
        return (
            list[str]
            | None,  # Assuming object properties are represented as lists of strings (entity names) (can be None as well, open-world assumption)
            # TODO ADD DESCRIPTION AS TO WHAT ENTITIES CAN BE NAMED HERE!
            # TODO respect functionality
            Field(None, description=_generate_description(prop.description)),
        )


# TODO improve __doc__ and everything for the schemas, mention that the model can use it to extract partial data from the knowledge graph, i.e. just one field for a specific class

# TODO disallow "name" and "cls" properties in ontology generator - these are reserved by us!


def _generate_class_schema(ontology: Ontology, cls: Class):
    """Generates a Pydantic model schema for a class in the ontology."""

    fields: dict = {
        "cls": (Literal[cls.name], Field(..., description="Entity class.")),  # discriminator field
    }

    for prop in ontology.get_properties(cls).values():
        fields[prop.name] = _generate_property_field(prop)

    return create_model(
        f"Partial{cls.name}",
        __base__=DynamicPartialEntity,
        __doc__=_generate_description(cls.description),
        **fields,
    )


def generate_kg_schema(ontology: Ontology):
    """Generates a Pydantic model schema for a knowledge graph based on the provided ontology."""

    if not ontology.classes:
        raise ValueError("Ontology must contain at least one class.")

    # we do not allow the root class in the schema, as it should not be instantiated directly (TODO: check if this is good)
    classes = [_generate_class_schema(ontology, cls) for cls in ontology.classes.values() if cls.superclass is not None]

    # create a union type out of the classes
    any_class_type = classes[0]
    for cls in classes[1:]:
        any_class_type |= cls

    PartialKnowledgeGraph = create_model(
        "PartialKnowledgeGraph",
        __base__=DynamicPartialKnowledgeGraph,
        __doc__="A partial knowledge graph containing structured data.",
        data=(
            list[Annotated[any_class_type, Field(discriminator="cls")]],
            Field(default_factory=list),
        ),
    )

    # TODO update prompt! it currently asks for triplets

    return PartialKnowledgeGraph
