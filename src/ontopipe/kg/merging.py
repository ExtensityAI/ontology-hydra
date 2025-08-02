from typing import cast

from ontopipe.kg.schema import DynamicPartialEntity, DynamicPartialKnowledgeGraph
from ontopipe.ontology.models import Ontology

_IGNORE_MERGE_KEYS = {"name", "cls"}


def _merge_values(a: list | None, b: list | None):
    if a is not None and b is not None:
        return list(set(a + b))  # deduplicate values
    elif a is not None:
        return list(a)
    else:
        return list(b)  # b is guaranteed to be not None here (given call args)


def try_merge(
    ontology: Ontology,
    kg_type: type[DynamicPartialKnowledgeGraph],
    a: DynamicPartialKnowledgeGraph,
    b: DynamicPartialKnowledgeGraph,
):
    issues = []

    a_dict = cast(
        dict[str, DynamicPartialEntity], {e.name: e for e in a.data}
    )  # both have data, we might be able to solve linter errors with Protocols
    b_dict = cast(dict[str, DynamicPartialEntity], {e.name: e for e in b.data})

    # TODO right now, with this, we essentially do not guarantee order of entities - maybe this is bad, because it might be the case that recently generated entities have more relevance for the following text chunks? Or maybe randomized order every time could increase robustness?

    output = list[DynamicPartialEntity]()

    # get names of all defined entities
    all_names = set(a_dict.keys()) | set(b_dict.keys())

    for name in all_names:
        a_entity = a_dict.get(name)
        b_entity = b_dict.get(name)

        # as same name <=> same entity, they need to have the same class
        if a_entity is not None and b_entity is not None and not a_entity.cls == b_entity.cls:
            # detected conflicting classes, this is an error
            # TODO yield issue
            raise NotImplementedError()

        entity_type = cast(
            type[DynamicPartialEntity], a_entity.__class__ if a_entity is not None else b_entity.__class__
        )  # get the actual Python type from the entities (at least one of them is not None)
        class_name = a_entity.cls if a_entity is not None else b_entity.cls

        a_json = a_entity.model_dump(exclude_none=True) if a_entity else {}
        b_json = b_entity.model_dump(exclude_none=True) if b_entity else {}

        # TODO if both JSONs are equal, just add one!

        # TODO when merging, prompt an LLM to perform merging if two (or some) values of the merged jsons are very similar (i.e. the model extracted the same information twice!)

        # TODO currently, we ignore functional properties, but should not do so later!

        all_keys = (set(a_json.keys()) | set(b_json.keys())) - _IGNORE_MERGE_KEYS

        d = {"name": name, "cls": class_name, **{k: _merge_values(a_json.get(k), b_json.get(k)) for k in all_keys}}
        output.append(entity_type.model_validate(d))

    success = len(issues) == 0

    return success, issues, kg_type(data=output) if success else None
