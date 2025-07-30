from datetime import date, datetime, time

from pydantic import BaseModel, Field, create_model

from ontopipe.ontology.models import (
    DataProperty,
    DataType,
    Description,
    ObjectProperty,
    Ontology,
)

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


def _generate_property_field(prop: DataProperty | ObjectProperty):
    """Generates a Pydantic field for a property in the ontology."""

    if isinstance(prop, DataProperty):
        return (
            _data_type_to_python[prop.type]
            | None,  # data properties can be None (open-world assumption)
            Field(None, description=_generate_description(prop.description)),
        )

    elif isinstance(prop, ObjectProperty):
        return (
            list[str]
            | None,  # Assuming object properties are represented as lists of strings (entity names) (can be None as well, open-world assumption)
            Field(None, description=_generate_description(prop.description)),
        )


def generate_kg_schema(ontology: Ontology):
    """Generates a Pydantic model schema for a knowledge graph based on the provided ontology."""

    # the schema is then used to extract structured data from the ontology.

    if not ontology.classes:
        raise ValueError("Ontology must contain at least one class.")

    classes = list[type[BaseModel]]()

    for name, cls in ontology.classes.items():
        fields: dict = {
            "name": (
                str,
                Field(..., description="Entity name."),
            ),  # TODO add proper description!
        }

        for name, prop in ontology.get_properties(cls).items():
            fields[name] = _generate_property_field(prop)

        classes.append(
            create_model(name, __doc__=_generate_description(cls.description), **fields)
        )

    # create a union type out of the classes
    any_class_type = classes[0]
    for cls in classes[1:]:
        any_class_type |= cls

    print(any_class_type)

    PartialKnowledgeGraph = create_model(
        "PartialKnowledgeGraph",
        __doc__="A partial knowledge graph containing entities from the ontology.",
        entities=(
            list[any_class_type] | None,
            Field(None, description="List of entities in the knowledge graph."),
        ),
    )

    return PartialKnowledgeGraph
