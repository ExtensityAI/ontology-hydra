from logging import getLogger
from pathlib import Path

from symai import Expression
from symai.components import MetadataTracker
from symai.strategy import LLMDataModel, contract
from tqdm import tqdm

from ontopipe.kg.merging import try_merge
from ontopipe.kg.schema import DynamicPartialKnowledgeGraph, generate_kg_schema
from ontopipe.ontology.models import Ontology
from ontopipe.prompts import prompt_registry

logger = getLogger("ontopipe.kg")


def is_snake_case(s):
    # Must not start or end with underscore
    if not s or s[0] == "_" or s[-1] == "_":
        return False
    # Must not have consecutive underscores
    if "__" in s:
        return False
    # Must only contain lowercase letters, numbers, or underscores
    return all(c.islower() or c.isdigit() or c == "_" for c in s)


"""
@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=True,
    remedy_retry_params=dict(tries=25, delay=0.5, max_delay=15, jitter=0.1, backoff=2, graceful=False),
    accumulate_errors=False,
)
class KnowledgeGraphGenerator(Expression):
    def __init__(self, name: str, ontology: Ontology | None = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.ontology = ontology
        self._triplets = set[Triplet]()

    def forward(self, input: TripletExtractorInput, **kwargs) -> KGState:
        if self.contract_result is None:
            raise ValueError("Contract failed!")
        return self.contract_result

    def pre(self, input: TripletExtractorInput) -> bool:
        return True

    def post(self, output: KGState) -> bool:
        if output.triplets is None:
            return True  # Nothing was extracted.

        # ignore triplets that are already in the KG
        triplets = [t for t in output.triplets if t not in self._triplets]

        # If no ontology is provided, skip validation and accept all triplets
        if self.ontology is None:
            return True

        errors = []

        new_type_defs = dict[str, str]()
        existing_type_defs = {x.subject: x.object for x in self._triplets if x.predicate == "isA"}

        # ?: add information on how to fix for each of the errors
        # ?: consider adding context information to errors (i.e. for what types a property is valid, etc.)

        # first iteration: check for new isA triplets, ensure that they are valid
        for triplet in (t for t in triplets if t.predicate == "isA"):
            if (subject_class := existing_type_defs.get(triplet.subject, None)) is not None or (
                subject_class := new_type_defs.get(triplet.subject, None)
            ) is not None:
                if subject_class in self.ontology.superclasses[triplet.object]:
                    # allow to redefine a type definition if the new one is a subclass of the current one
                    continue

                if not is_snake_case(triplet.subject):
                    # Ensure that the subject entity is in snake_case
                    errors.append(
                        f"{triplet}: Entity '{triplet.subject}' must be in snake_case format. Please rename it to follow the convention."
                    )
                    continue

                # Ensure that the subject entity does not have a type def already
                errors.append(
                    f"{triplet}: Entity '{triplet.subject}' already classified as '{subject_class}'. Cannot reclassify as '{triplet.object}' unless it's a subclass. Either remove this triplet or use a valid subclass."
                )
                continue

            if not self.ontology.get_class(triplet.object):
                # ensure ontology has class for object
                errors.append(
                    f"{triplet}: '{triplet.object}' is not a defined class in the ontology schema. For isA relations, the object must be a class defined in the ontology schema."
                )
                continue

            if self.ontology.get_class(triplet.subject):
                # Ensure that the subject is not an ontology class
                errors.append(
                    f"{triplet}: '{triplet.subject}' is a class, not an entity instance. In isA relations, the subject must be an entity instance, not a class."
                )
                continue

            new_type_defs[triplet.subject] = triplet.object

        all_type_defs = {**new_type_defs, **existing_type_defs}

        superclasses = self.ontology.superclasses

        # second iteration: make sure triplets have valid type definitions
        for triplet in (t for t in triplets if t.predicate != "isA"):
            # now, any triplet needs to be a property (either object or data)

            subject_class = all_type_defs.get(triplet.subject, None)
            object_class = all_type_defs.get(triplet.object, None)

            property = self.ontology.get_property(triplet.predicate)

            if not property:
                # Ensure that the predicate is a valid ontology property
                errors.append(
                    f"{triplet}: '{triplet.predicate}' is not a valid property in the ontology schema. Use only defined properties from the schema."
                )
                continue

            if not subject_class:
                # Ensure that the subject entity has a type definition
                errors.append(
                    f"{triplet}: Entity '{triplet.subject}' lacks class assignment. First add ({triplet.subject}, isA, <validClass>) before using this entity."
                )
                continue

            if self.ontology.get_class(triplet.subject):
                # Ensure that the subject entity is not an ontology class (this is only allowed for isA predicates!)
                errors.append(
                    f"{triplet}: '{triplet.subject}' is a class definition, not an entity instance. For non-isA relations, both subject and object must be entity instances."
                )
                continue

            if self.ontology.get_class(triplet.object):
                # Ensure that the object entity is not an ontology class (this is only allowed for isA predicates!)
                errors.append(
                    f"{triplet}: '{triplet.object}' is a class definition, not an entity instance. For non-isA relations, both subject and object must be entity instances."
                )
                continue

            if not is_snake_case(triplet.subject):
                # Ensure that the subject entity is in snake_case
                errors.append(
                    f"{triplet}: Entity '{triplet.subject}' must be in snake_case format. Please rename it to follow the convention."
                )
                continue

            if isinstance(property, ObjectPropertyModel):
                if not object_class:  # (properties do not need type definition for object)
                    # Object entity does not have a type definition yet
                    errors.append(
                        f"{triplet}: Entity '{triplet.object}' lacks class assignment. First add ({triplet.object}, isA, <validClass>) before using this entity."
                    )
                    continue

                if not property.is_valid_for(superclasses[subject_class], superclasses[object_class]):
                    # Ensure that the property is valid for the subject and object types
                    errors.append(
                        f"{triplet}: Property '{triplet.predicate}' cannot connect '{subject_class}' entities to '{object_class}' entities according to the ontology constraints."
                    )
                    continue

                if not is_snake_case(triplet.object):
                    # Ensure that the object entity is in snake_case
                    errors.append(
                        f"{triplet}: Entity '{triplet.object}' must be in snake_case format. Please rename it to follow the convention."
                    )
                    continue

            # ? consider adding validation for data properties

        if errors:
            raise ValueError(f"Triplet extraction failed with the following errors: \n- {'\n- '.join(errors)}")

        return True

    @property
    def prompt(self) -> str:
        if self.ontology is None:
            return prompt_registry.instruction("triplet_extraction_no_ontology")
        else:
            return prompt_registry.instruction("triplet_extraction")

    def extend_triplets(self, new_triplets: list[Triplet]):
        if new_triplets:
            self._triplets.update(new_triplets)

    def get_kg(self) -> KG:
        return KG(name=self.name, triplets=self._triplets)
"""


def _create_extractor(PartialKnowledgeGraphType: type[DynamicPartialKnowledgeGraph]):
    class Input(LLMDataModel):
        texts: list[str]
        kg: PartialKnowledgeGraphType  # pyright: ignore[reportInvalidTypeForm] we use this dynamically defined schema here

    @contract(
        pre_remedy=False,
        post_remedy=True,
        verbose=True,
        remedy_retry_params=dict(tries=25, delay=0.5, max_delay=15, jitter=0.1, backoff=2, graceful=False),
        accumulate_errors=False,
    )
    class Extractor(Expression):
        def __init__(self, ontology: Ontology, kg: PartialKnowledgeGraphType, *args, **kwargs):  # pyright: ignore[reportInvalidTypeForm] use dynamic type here
            super().__init__(*args, **kwargs)
            self._ontology = ontology
            self._kg = kg

        def forward(self, input: Input, **kwargs) -> PartialKnowledgeGraphType:  # pyright: ignore[reportInvalidTypeForm] we again use dynamically defined schema here
            if self.contract_result is None:
                raise ValueError("Contract failed!")
            return self.contract_result

        def pre(self, input: Input) -> bool:
            return True

        def post(self, output: PartialKnowledgeGraphType) -> bool:  # pyright: ignore[reportInvalidTypeForm] here too
            # TODO add combination step here (or maybe alternatively in forward? CHECK DOCS! Essentially, in the output, we want to combine all partial entities that have the same name as that simplifies merging)

            # combine output values

            # TODO validate that all object properties are related to the correct ontology classes, i.e. no leo hasParent car (if car is not a Person but a Car and leo is a Person)

            success, issues, merged = try_merge(self._ontology, PartialKnowledgeGraphType, self._kg, output)

            if not success or merged is None:
                raise ValueError("Some issues occured while merging:" + "\n".join(issues))

            self._kg = merged

            return True

        @property
        def prompt(self):
            return prompt_registry.instruction("triplet_extraction")

        @property
        def kg(self):
            return self._kg

    return Input, Extractor


def generate_kg(
    cache_path: Path,
    texts: list[str],
    ontology: Ontology | None = None,
    batch_size: int = 1,
    epochs: int = 3,
) -> None:
    cache_path = cache_path / "kg.json"

    partial_json_cache_path = cache_path.with_suffix(".partial.json")
    partial_html_cache_path = cache_path.with_suffix(".partial.html")

    if not ontology:
        raise NotImplementedError("For now, ontology must be provided to generate a knowledge graph due to update.")

    PartialKnowledgeGraph = generate_kg_schema(ontology)

    Input, Extractor = _create_extractor(PartialKnowledgeGraph)

    usage = None
    kg = PartialKnowledgeGraph(data=[])

    extractor = Extractor(ontology, kg)

    with MetadataTracker() as tracker:
        for i in range(epochs):
            for j in tqdm(range(0, len(texts), batch_size), desc=f"Epoch {i + 1}/{epochs}"):
                input_data = Input(
                    texts=texts[j : j + batch_size],
                    kg=kg,
                )

                # TODO annotate output with chunk information!

                _ = extractor(input=input_data)  # pyright: ignore[reportInvalidTypeForm] once again

                kg = extractor.kg

                partial_json_cache_path.write_text(kg.model_dump_json(indent=2, exclude_none=True))

        usage = tracker.usage
        extractor.contract_perf_stats()
        logger.debug("API Usage: %s", usage)

    kg = extractor.get_kg()
    cache_path.write_text(kg.model_dump_json(indent=2), encoding="utf-8")

    # remove partial cache files once again
    partial_json_cache_path.unlink(missing_ok=True)
    partial_html_cache_path.unlink(missing_ok=True)

    return kg
