from dataclasses import dataclass

from ontopipe.ontology.models import Class, ClassModel, Concept, DataProperty, ObjectProperty, Ontology


@dataclass(frozen=True, slots=True)
class Issue:
    code: str
    path: str
    message: str
    context: str | None = None
    hint: str | None = None

    def __str__(self) -> str:
        return f"[{self.path}] {self.message}{f' (context: {self.context})' if self.context else ''}{f' (hint: {self.hint})' if self.hint else ''}"


def _try_add_classes(ontology: Ontology, class_models: list[ClassModel]):
    for cls in class_models:
        if existing_class := ontology.classes.get(cls.name):
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

        ontology.classes[cls.name] = Class(
            name=cls.name,
            description=cls.description,
            superclass=cls.superclass,
            own_properties=[],  # this needs to be validated and filled after properties are done
        )


def _validate_class_hierarchy(
    ontology: Ontology,
    classes: list[Class],
):
    for cls in classes:
        path = f"class:{cls.name}"

        if cls.superclass is not None and not ontology.classes.get(cls.superclass):
            # ensure superclass exists (if class has one)
            yield Issue(
                code="superclass_not_found",
                path=path,
                message=f"Superclass '{cls.superclass}' of class '{cls.name}' not found in ontology",
                context=None,
                hint=f"Define the superclass '{cls.superclass}', choose a different one or remove the class.",
            )
            continue

        if cls.superclass == cls.name:
            # ensure subclass is not the same as superclass
            yield Issue(
                code="subclass_equals_superclass",
                path=path,
                message=f"Class '{cls.name}' cannot be its own superclass",
                context=None,
                hint="Choose a different superclass or remove the class.",
            )
            continue

        # ensure we have no cycles in class hierarchy
        for cls in classes:
            if (sup := ontology.get_superclass(cls)) is None:
                # if class has no superclass, it can't have a circular relation
                continue

            cls_descendants = [c.name for c in ontology.get_descendants(cls)]
            sup_ancestors = [c.name for c in ontology.get_ancestors(sup)]

            # TODO consider allowing redefinition of types? i.e. choosing a different parent?

            if any(sc in sup_ancestors for sc in cls_descendants) or cls.name in sup_ancestors:
                # if any descendant of the class or the class itself is an ancestor of the superclass, we have a circular relation
                yield Issue(
                    code="circular_class_hierarchy",
                    path=f"class:{cls.name}",
                    message="Circular hierarchy detected",
                    context=f"'{cls.superclass}' is already a subclass of '{cls.name}' (directly or indirectly)",
                    hint="Remove this relation or restructure the hierarchy",
                )
                continue

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
        if cls.superclass is None:
            if has_root:
                # if the class has no superclass and there is a root, it should be a subclass of some class
                yield Issue(
                    code="superclass_not_found",
                    path=f"class:{cls.name}",
                    message=f"Class '{cls.name}' has no superclass",
                    hint=f"Set the superclass of '{cls.name}' to place it in the hierarchy",
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


def _try_add_properties(ontology: Ontology, props: list[DataProperty | ObjectProperty]):
    for prop in props:
        path = f"property:{prop.name}"

        if existing_prop := ontology.properties.get(prop.name):
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
        invalid_domains = [domain for domain in prop.domain if not ontology.classes.get(domain)]

        if invalid_domains:
            yield Issue(
                code="domain_classes_not_found",
                path=path,
                message=f"Domain classes not found for property '{prop.name}'",
                context=f"Missing classes are {', '.join(f"'{domain}'" for domain in invalid_domains)}",
                hint="Define these classes first or remove them from the domain",
            )
            continue

        if isinstance(prop, ObjectProperty):
            # ensure all range classes exist
            invalid_ranges = [range_ for range_ in prop.range if not ontology.classes.get(range_)]

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

            ontology.object_properties[prop.name] = prop
        elif isinstance(prop, DataProperty):
            # no need to validate range as it is a fixed Literal type
            ontology.data_properties[prop.name] = prop


def _update_own_properties(ontology: Ontology):
    props = ontology.properties

    # add props to own_properties of classes where the class name is in the domain of a prop
    for prop in props.values():
        for class_name in prop.domain:
            c = ontology.classes[class_name]

            # add to own properties only if not already contained
            if prop.name not in c.own_properties:
                c.own_properties.append(prop.name)


def try_add_concepts(ontology: Ontology, concepts: list[Concept]):
    """Try to add concepts to the ontology, returning any issues found. If no issues are found, the ontology is updated with the new concepts."""

    original_ontology = ontology
    ontology = ontology.model_copy(deep=True)  # ensure we do not modify the original ontology while validating

    issues = list[Issue]()

    class_models = [c for c in concepts if isinstance(c, ClassModel)]
    props = [c for c in concepts if isinstance(c, (DataProperty | ObjectProperty))]

    issues += _try_add_classes(ontology, class_models)

    if issues:
        # TODO should we actually stop here if there was an issue with class defs?
        return False, issues

    classes = [ontology.classes[cm.name] for cm in class_models]

    issues += _validate_class_hierarchy(ontology, classes)

    issues += _try_add_properties(ontology, props)

    is_valid = len(issues) == 0

    try:
        if is_valid:
            # if no issues, we can add the concepts to the ontology
            for cls in classes:
                original_ontology.classes[cls.name] = cls

            for concept in concepts:
                if isinstance(concept, DataProperty):
                    original_ontology.data_properties[concept.name] = concept
                elif isinstance(concept, ObjectProperty):
                    original_ontology.object_properties[concept.name] = concept

            _update_own_properties(original_ontology)
    except Exception as e:
        print(e)
        print(e.__traceback__)
        return 1 / 0

    return is_valid, issues
