from dataclasses import dataclass

from ontopipe.models import Class, Concept, DataProperty, ObjectProperty, Ontology, SubClassRelation


@dataclass(frozen=True, slots=True)
class Issue:
    code: str
    concept: Concept | None
    message: str
    hint: str | None = None

    def _get_concept_name(self) -> str:
        if self.concept is None:
            return ""

        if not isinstance(self.concept, SubClassRelation):
            return f"{self.concept.__class__.__name__} '{self.concept.name}'"

        return f"SubClassRelation from '{self.concept.subclass}' to '{self.concept.superclass}'"

    def __str__(self) -> str:
        return f"{self._get_concept_name()}: {self.message}" + (f" (Hint: {self.hint})" if self.hint else "")


def _validate_class(ontology: Ontology, cls: Class, new_classes: set[str]):
    # check if class already exists
    if existing_class := ontology.get_class(cls.name):
        return Issue(
            code="class_already_exists",
            concept=cls,
            message=f"Class '{cls.name}' already exists in the ontology.",
            hint=f"Please ensure unique class names. Existing class: <value>{existing_class.model_dump_json()}</value>",
        )
    if cls.name[0] == cls.name[0].lower():
        return Issue(
            code="class_name_lowercase",
            concept=cls,
            message=f"Class name '{cls.name}' should start with an uppercase letter. Are you trying to define a class or a relationship?",
        )


def _validate_subclass_relation(ontology: Ontology, relation: SubClassRelation, new_classes: set[str]):
    # check if subclass and superclass exist
    if not ontology.get_class(relation.subclass) and relation.subclass not in new_classes:
        return Issue(
            code="subclass_not_found",
            concept=relation,
            message=f"Subclass {relation.subclass} does not exist in the ontology.",
        )

    if not ontology.get_class(relation.superclass) and relation.superclass not in new_classes:
        return Issue(
            code="superclass_not_found",
            concept=relation,
            message=f"Superclass {relation.superclass} does not exist in the ontology.",
        )

    # check if prospective subclass is already defined as a subclass
    if (superclass := ontology.get_superclass(relation.subclass)) is not None:
        return Issue(
            code="subclass_already_defined",
            concept=relation,
            message=f"Subclass '{relation.subclass}' is already defined as a subclass of '{superclass}'. Please ensure that each subclass has only one direct superclass.",
        )

    # TODO allow type narrowing for subclasses, i.e. a more specific superclass than currently assigned is allowed.


def _validate_object_property(ontology: Ontology, prop: ObjectProperty, new_classes: set[str]):
    # check if object property already exists
    if existing_prop := ontology.get_property(prop.name):
        return Issue(
            code="object_property_already_exists",
            concept=prop,
            message=f"Property '{prop.name}' already exists in the ontology.",
            hint=f"Please ensure unique object property names. Existing property: <value>{existing_prop.model_dump_json()}</value>",
        )

    # check if domains and ranges are valid classes
    for domain in prop.domain:
        if not ontology.get_class(domain) and domain not in new_classes:
            return Issue(
                code="domain_class_not_found",
                concept=prop,
                message=f"Domain class '{domain}' of object property '{prop.name}' does not exist.",
            )

    for range in prop.range:
        if not ontology.get_class(range) and range not in new_classes:
            return Issue(
                code="range_class_not_found",
                concept=prop,
                message=f"Range class '{range}' of object property '{prop.name}' does not exist.",
            )

    if len(prop.range) == 0:
        return Issue(
            code="object_property_range_not_defined",
            concept=prop,
            message=f"Object property '{prop.name}' must have at least one range class defined. Please add a range class or remove the property.",
        )

    if len(prop.domain) == 0:
        return Issue(
            code="object_property_domain_not_defined",
            concept=prop,
            message=f"Object property '{prop.name}' must have at least one domain class defined. Please add a domain class or remove the property.",
        )


def _validate_data_property(ontology: Ontology, prop: DataProperty, new_classes: set[str]):
    # check if data property already exists
    if existing_prop := ontology.get_property(prop.name):
        return Issue(
            code="data_property_already_exists",
            concept=prop,
            message=f"Property '{prop.name}' already exists in the ontology.",
            hint=f"Please ensure unique data property names. Existing property: <value>{existing_prop.model_dump_json()}</value>",
        )

    if len(prop.domain) == 0:
        return Issue(
            code="data_property_domain_not_defined",
            concept=prop,
            message=f"Data property '{prop.name}' must have at least one domain class defined. Please add a domain class or remove the property.",
        )

    # check if domains are valid classes
    for domain in prop.domain:
        if not ontology.get_class(domain) and domain not in new_classes:
            return Issue(
                code="domain_class_not_found",
                concept=prop,
                message=f"Domain class '{domain}' of data property '{prop.name}' does not exist. Either define the class or remove it from the domain.",
            )


_VALIDATORS = {
    Class: _validate_class,
    SubClassRelation: _validate_subclass_relation,
    ObjectProperty: _validate_object_property,
    DataProperty: _validate_data_property,
}


def _check_for_duplicates(concepts: list[Concept]):
    names = set()

    for concept in (concept for concept in concepts if not isinstance(concept, SubClassRelation)):
        # check if we have any duplicate names
        if concept.name in names:
            # TODO print both values?
            return Issue(
                code="duplicate_concept",
                concept=concept,
                message=f"Duplicate concept found: {concept.name}. Please ensure all concepts have unique names.",
            )
        names.add(concept.name)

    subclasses = set()
    for relation in (concept for concept in concepts if isinstance(concept, SubClassRelation)):
        # check if we have any duplicate subclass relations
        if relation.subclass in subclasses:
            return Issue(
                code="duplicate_subclass_relation",
                concept=relation,
                message=f"Duplicate subclass relation found: {relation.subclass} is already defined as a subclass. Please ensure each subclass has only one direct superclass.",
            )

        subclasses.add(relation.subclass)


def _build_graph(ontology: Ontology, additions: list[Concept]):
    """Return (adjacency, all_classes, new_classes)."""
    adjacency: dict[str, set[str]] = {c.name: set() for c in ontology.classes}
    all_classes: set[str] = set(adjacency)
    new_classes: set[str] = set()

    # merge in new classes
    for c in (x for x in additions if isinstance(x, Class)):
        all_classes.add(c.name)
        new_classes.add(c.name)
        adjacency.setdefault(c.name, set())

    # existing relations
    for r in ontology.subclass_relations:
        adjacency.setdefault(r.superclass, set()).add(r.subclass)

    # additions relations
    for r in (x for x in additions if isinstance(x, SubClassRelation)):
        adjacency.setdefault(r.superclass, set()).add(r.subclass)

    return adjacency, all_classes, new_classes


def _find_roots(adjacency: dict[str, set[str]]) -> set[str]:
    children = {child for kids in adjacency.values() for child in kids}
    return set(adjacency) - children


def validate_additions(ontology: Ontology, concepts: list[Concept]):
    """Validate that the concepts can be added to the ontology without conflicts.

    Args:
        ontology (Ontology): The ontology to validate against.
        concepts (list[Concept]): The concepts to validate.

    Returns:
        tuple[bool, list[str]]: A tuple containing a boolean indicating if the validation passed and a list of issues found."""

    issues = []

    if issue := _check_for_duplicates(concepts):
        issues.append(issue)

    # ensure that the ontology is well-formed (tree structure)
    adj, all_classes, new_classes = _build_graph(ontology, concepts)

    for concept in concepts:
        validator = _VALIDATORS.get(type(concept))

        if issue := validator(ontology, concept, new_classes):
            issues.append(issue)

    has_top_level_class = ontology.get_top_level_class() is not None

    # ensure all new classes have a subclass relation
    subclasses_in_rel = {r.subclass for r in concepts if isinstance(r, SubClassRelation)}
    for cls in new_classes:
        if cls not in subclasses_in_rel:
            if has_top_level_class:
                issues.append(
                    Issue(
                        code="missing_subclass_relation",
                        concept=None,
                        message=f"Class '{cls}' is defined but has no SubClassRelation.",
                        hint="Add exactly one SubClassRelation or delete the class.",
                    )
                )

            has_top_level_class = True

    roots = _find_roots(adj)
    if len(roots) != 1:
        # we have no root or multiple roots
        issues.append(
            Issue(
                code="multiple_roots" if roots else "no_root",
                concept=None,
                message=f"Expected exactly one root, found {len(roots)}: {', '.join(roots) or '∅'}.",
                hint="Ensure there is a single top-level class with no superclass.",
            )
        )
    else:
        # ensure we have no cycles
        root = next(iter(roots))
        visited, stack = set(), [root]
        parent = {}
        while stack:
            node = stack.pop()
            if node in visited:
                # back-edge ⇒ cycle
                loop = f"{node} → {parent[node]}"
                issues.append(
                    Issue(
                        code="cycle_detected",
                        concept=None,
                        message=f"Cycle detected starting at '{node}'.",
                        hint=f"Break the loop at '{loop}'.",
                    )
                )
                break
            visited.add(node)
            for child in adj.get(node, ()):
                parent[child] = node
                stack.append(child)

        # ensure we have no disconnected classes
        missing = all_classes - visited
        if missing:
            issues.append(
                Issue(
                    code="disconnected_classes",
                    concept=None,
                    message=f"{len(missing)} class(es) are not reachable from root '{root}'.",
                    hint=f"Link {' ,'.join(list(missing)[:3])}… to the tree.",
                )
            )

    return len(issues) == 0, issues
