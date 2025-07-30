from typing import Literal

from pydantic import BaseModel, Field

Characteristic = Literal[
    "functional",
    "inverseFunctional",
    "transitive",
    "symmetric",
    "asymmetric",
    "reflexive",
    "irreflexive",
]

DataType = Literal[
    "string",
    "int",
    "float",
    "boolean",
    "datetime",
    "date",
    "time",
]


class Description(BaseModel):
    """Represents the description of an ontology element."""

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


class DataProperty(BaseModel):
    name: str
    description: Description | None = None

    type: DataType

    characteristics: list[Characteristic] = []

    domain: list[str] = []


class ObjectProperty(BaseModel):
    name: str
    description: Description | None = None

    characteristics: list[Characteristic] = []

    domain: list[str] = []
    range: list[str] = []


class Class(BaseModel):
    name: str
    description: Description | None = None
    own_properties: list[str]

    superclass: str | None


class Ontology(BaseModel):
    classes: dict[str, Class]
    object_properties: dict[str, ObjectProperty]
    data_properties: dict[str, DataProperty]

    @property
    def properties(self):
        """Returns a combined dictionary of all object and data properties."""
        return dict[str, DataProperty | ObjectProperty](
            **self.object_properties, **self.data_properties
        )

    def get_superclass(self, cls: Class):
        """Returns the super class of the given class, or None if it is the root."""
        return self.classes[cls.superclass] if cls.superclass is not None else None

    def get_class_hierarchy(self, cls: Class):
        """Returns the class hierarchy chain starting from the root down to the given class."""
        chain = []
        c = cls

        while c is not None:
            chain.append(c)
            c = self.get_superclass(c)

        return chain

    def get_properties(self, cls: Class, include_inherited: bool = True):
        """Returns all properties associated with a class, optionally including inherited properties."""

        all_props = self.properties
        chain = self.get_class_hierarchy(cls) if include_inherited else [cls]

        props = dict[str, DataProperty | ObjectProperty]()

        for c in chain:
            props.update({pn: all_props[pn] for pn in c.own_properties})

        return props

    def resolve_class_names(self, class_names: list[str]):
        """Resolves the given class names to class instances."""
        return [self.classes[cn] for cn in class_names]
