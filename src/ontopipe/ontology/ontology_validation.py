from dataclasses import dataclass

from ontopipe.models import (
    ClassModel,
    Concept,
    DataProperty,
    ObjectPropertyModel,
    OntologyModel,
    SubClassRelationModel,
)


@dataclass(frozen=True, slots=True)
class Issue:
    code: str
    path: str
    message: str
    context: str | None = None
    hint: str | None = None

    def __str__(self) -> str:
        return f"[{self.path}] {self.message}{f' (context: {self.context})' if self.context else ''}{f' (hint: {self.hint})' if self.hint else ''}"


def _try_add_classes(ontology: OntologyModel, classes: list[ClassModel]):
    for cls in classes:
        if existing_class := ontology.get_class(cls.name):
            # ensure class does not yet exist
            yield Issue(
                code="class_already_exists",
                path=f"class:{cls.name}",
                message=f"Class '{cls.name}' already exists in the ontology",
                context=f"Existing class is '{existing_class.name}'",
                hint="Choose a different name or if it is a this duplicate class definition, omit it.",
            )
            continue

        if cls.name[0].lower() == cls.name[0]:
            # ensure class name starts uppercase
            yield Issue(
                code="class_name_not_uppercase",
                path=cls.name,
                message=f"Class name '{cls.name}' must start with an uppercase letter.",
                hint="Rename the class to start with an uppercase letter.",
            )
            continue

        ontology.classes.append(cls)


def _try_add_subclass_relations(
    ontology: OntologyModel,
    classes: list[ClassModel],
    rels: list[SubClassRelationModel],
):
    all_superclasses = ontology.superclasses

    for rel in rels:
        path = f"relation:{rel.subclass}->{rel.superclass}"
        if not ontology.get_class(rel.subclass):
            # ensure subclass exists
            yield Issue(
                code="subclass_not_found",
                path=path,
                message=f"Subclass '{rel.subclass}' not found in ontology",
                context=None,
                hint="Define this class before creating the subclass relation",
            )
            continue

        if not ontology.get_class(rel.superclass):
            # ensure superclass exists
            yield Issue(
                code="superclass_not_found",
                path=path,
                message=f"Superclass '{rel.superclass}' not found in ontology",
                context=None,
                hint="Define this class before creating the subclass relation",
            )
            continue

        if sc := ontology.get_superclass(rel.subclass):
            # ensure subclass does not already have a superclass
            # TODO allow specification (i.e. a more specific superclass)
            yield Issue(
                code="subclass_already_has_superclass",
                path=path,
                message=f"'{rel.subclass}' already has superclass '{sc}'",
                context=None,
                hint="Remove this relation or replace the existing one if this is more specific",
            )

        if rel.superclass == rel.subclass:
            # ensure subclass is not the same as superclass
            yield Issue(
                code="subclass_equals_superclass",
                path=path,
                message=f"Class '{rel.subclass}' cannot be it's own superclass",
                context=None,
                hint="Remove this relation or correct the names",
            )
            continue

        # ensure no cycles
        su_superclasses = all_superclasses[rel.superclass]
        sc_superclasses = all_superclasses[rel.subclass]

        # TODO consider allowing redefinition of types? i.e. choosing a different parent?

        if any(sc in su_superclasses for sc in sc_superclasses):
            yield Issue(
                code="circular_subclass_relation",
                path=path,
                message="Circular hierarchy detected",
                context=f"'{rel.superclass}' is already a subclass of '{rel.subclass}' (directly or indirectly)",
                hint="Remove this relation or restructure the hierarchy",
            )
            continue

        ontology.subclass_relations.append(rel)

    # TODO we should omit classes from the bottom check if the subclass relation validation failed for them

    root = ontology.root
    has_root = root is not None

    if root and any(cls.name == root.name for cls in classes):
        # if the root class is one of the classes we have just defined, we do not assume that we have a root yet!
        root = None
        has_root = False

    classes_without_superclass = []

    # ensure all classes have subclass relations
    for cls in classes:
        if not ontology.get_superclass(cls.name):
            if has_root:
                # if the class has no superclass and there is a root, it should be a subclass of some class
                yield Issue(
                    code="superclass_not_found",
                    path=f"class:{cls.name}",
                    message=f"Class '{cls.name}' has no superclass",
                    hint=f"Add a subclass relation for '{cls.name}' to place it in the hierarchy",
                )
                continue

            # if we do not have a root yet, and there is just one class without a superclass, then that will be the root and this is not an issue. However, we need to collect them in case more than one have no root, which would be invalid again.
            classes_without_superclass.append(cls)

    if len(classes_without_superclass) > 1:
        yield Issue(
            code="multiple_classes_without_superclass",
            path="hierarchy",
            message="Multiple top-level classes detected",
            context=f"Classes without superclass are {', '.join(f"'{cls.name}'" for cls in classes_without_superclass)}",
            hint="It is recommended to create a new (general) top-level class that is general enough s.t. all classes can inherit from it. All classes but the top-level one must have a superclass, thus you need to define a subclass relation for each of them.",
        )


def _try_add_properties(
    ontology: OntologyModel, props: list[DataProperty | ObjectPropertyModel]
):
    for prop in props:
        path = f"property:{prop.name}"
        if existing_prop := ontology.get_property(prop.name):
            # ensure property does not yet exist
            yield Issue(
                code="property_already_exists",
                path=path,
                message=f"Property '{prop.name}' already exists",
                context=f"Existing property is '{existing_prop.name}'",
                hint="Choose a different name or remove this duplicate property",
            )
            continue

        # ensure all domain classes exist (applies to both data and object properties)
        invalid_domains = [
            domain for domain in prop.domain if not ontology.get_class(domain)
        ]
        if invalid_domains:
            yield Issue(
                code="domain_classes_not_found",
                path=path,
                message=f"Domain classes not found for property '{prop.name}'",
                context=f"Missing classes are {', '.join(f"'{domain}'" for domain in invalid_domains)}",
                hint="Define these classes first or remove them from the domain",
            )
            continue

        if isinstance(prop, ObjectPropertyModel):
            # ensure all range classes exist
            invalid_ranges = [
                range_ for range_ in prop.range if not ontology.get_class(range_)
            ]

            if invalid_ranges:
                # TODO check if xsd: is in range, then the model likely wanted a data property instead
                yield Issue(
                    code="range_classes_not_found",
                    path=path,
                    message=f"Range classes not found for property '{prop.name}'",
                    context=f"Missing classes are {', '.join(f"'{range_}'" for range_ in invalid_ranges)}",
                    hint="Define these classes first or remove them from the range.",
                )
                continue

            ontology.object_properties.append(prop)
        elif isinstance(prop, DataProperty):
            # TODO validate data type of range
            ontology.data_properties.append(prop)


def try_add_concepts(ontology: OntologyModel, concepts: list[Concept]):
    """Try to add concepts to the ontology, returning any issues found."""

    ontology = ontology.clone()

    issues = list[Issue]()

    classes = [c for c in concepts if isinstance(c, ClassModel)]
    subclass_rels = [c for c in concepts if isinstance(c, SubClassRelationModel)]
    props = [c for c in concepts if isinstance(c, (DataProperty, ObjectPropertyModel))]

    issues += _try_add_classes(ontology, classes)

    if issues:
        # TODO should we actually stop here if there was an issue with class defs?
        return False, issues, None

    issues += _try_add_subclass_relations(ontology, classes, subclass_rels)

    issues += _try_add_properties(ontology, props)

    is_valid = len(issues) == 0
    return is_valid, issues, ontology if is_valid else None
