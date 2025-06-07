from logging import getLogger

from symai import Expression
from symai.components import MetadataTracker
from symai.strategy import contract
from tqdm import tqdm

from ontopipe.models import (
    KG,
    KGState,
    ObjectProperty,
    Ontology,
    Triplet,
    TripletExtractorInput,
)
from ontopipe.prompts import prompt_registry

logger = getLogger("ontopipe.kg")


@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=True,
    remedy_retry_params=dict(tries=25, delay=0.5, max_delay=15, jitter=0.1, backoff=2, graceful=False),
    accumulate_errors=False,
)
class TripletExtractor(Expression):
    def __init__(self, name: str, ontology: Ontology, threshold: float = 0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.threshold = threshold
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

                # Ensure that the subject entity does not have a type def already
                errors.append(
                    f"{triplet}: Entity '{triplet.subject}' already classified as '{subject_class}'. Cannot reclassify as '{triplet.object}' unless it's a subclass. Either remove this triplet or use a valid subclass."
                )
                continue

            if not self.ontology.has_class(triplet.object):
                # ensure ontology has class for object
                errors.append(
                    f"{triplet}: '{triplet.object}' is not a defined class in the ontology schema. For isA relations, the object must be a class defined in the ontology schema."
                )
                continue

            if self.ontology.has_class(triplet.subject):
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

            if self.ontology.has_class(triplet.subject):
                # Ensure that the subject entity is not an ontology class (this is only allowed for isA predicates!)
                errors.append(
                    f"{triplet}: '{triplet.subject}' is a class definition, not an entity instance. For non-isA relations, both subject and object must be entity instances."
                )
                continue

            if self.ontology.has_class(triplet.object):
                # Ensure that the object entity is not an ontology class (this is only allowed for isA predicates!)
                errors.append(
                    f"{triplet}: '{triplet.object}' is a class definition, not an entity instance. For non-isA relations, both subject and object must be entity instances."
                )
                continue

            if isinstance(property, ObjectProperty):
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

            # ? consider adding validation for data properties

        if errors:
            raise ValueError(f"Triplet extraction failed with the following errors: \n- {'\n- '.join(errors)}")

        return True

    @property
    def prompt(self) -> str:
        return prompt_registry.instruction("triplet_extraction")

    def extend_triplets(self, new_triplets: list[Triplet]):
        if new_triplets:
            self._triplets.update(new_triplets)

    def get_kg(self) -> KG:
        return KG(name=self.name, triplets=self._triplets)


def generate_kg(
    texts: list[str],
    kg_name: str,
    ontology: Ontology,
    threshold: float = 0.7,
    batch_size: int = 1,
    epochs: int = 3,
) -> KG:
    extractor = TripletExtractor(name=kg_name, threshold=threshold, ontology=ontology)

    usage = None
    triplets = []
    for i in range(epochs):
        n_new_triplets_in_epoch = 0
        with MetadataTracker() as tracker:
            for j in tqdm(range(0, len(texts), batch_size), desc=f"Epoch {i + 1}/{epochs}"):
                text = "\n".join(texts[j : j + batch_size])

                input_data = TripletExtractorInput(
                    text=text,
                    ontology=ontology,
                    state=KGState(triplets=triplets) if triplets else None,
                )

                try:
                    # TODO we can drastically reduce input size by sending triplets in a form of (subject, predicate, object) instead of escaped JSON
                    result = extractor(input=input_data)

                    if result.triplets is not None:
                        n_triplets_before = len(triplets)
                        new_triplets = result.triplets
                        triplets.extend(new_triplets)
                        extractor.extend_triplets(new_triplets)
                        n_triplets = len(triplets)

                        n_new_triplets = n_triplets - n_triplets_before
                        n_new_triplets_in_epoch += n_new_triplets

                        logger.debug(
                            "Extracted %i new triplets from text chunk: %s",
                            n_new_triplets,
                            text[:50],
                        )

                except Exception as e:
                    logger.error("Error extracting triplets from text", exc_info=e)

            usage = tracker.usage
            extractor.contract_perf_stats()

        logger.info(
            "Epoch %i: Extracted %i new triplets, total %i triplets",
            i,
            n_new_triplets_in_epoch,
            len(triplets),
        )

        logger.debug("API Usage in Epoch %i: %s", i, usage)

        if n_new_triplets_in_epoch == 0:
            logger.info("No new triplets extracted in epoch %i, stopping early.", i)
            break

    return extractor.get_kg()
