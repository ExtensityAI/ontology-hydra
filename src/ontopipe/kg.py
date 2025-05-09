from logging import getLogger

from symai import Expression
from symai.components import MetadataTracker
from symai.strategy import contract
from tqdm import tqdm

from ontopipe.models import KG, KGState, Ontology, Triplet, TripletExtractorInput
from ontopipe.prompts import prompt_registry

logger = getLogger("ontopipe.kg")


@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=True,
    remedy_retry_params=dict(
        tries=25, delay=0.5, max_delay=15, jitter=0.1, backoff=2, graceful=False
    ),
    accumulate_errors=False,
)
class TripletExtractor(Expression):
    def __init__(
        self, name: str, ontology: Ontology, threshold: float = 0.7, *args, **kwargs
    ):
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
        existing_type_defs = {
            x.subject: x.object for x in self._triplets if x.predicate == "isA"
        }

        # TODO: add information on how to fix for each of the errors

        # first iteration: check for new isA triplets, ensure that they are valid
        for triplet in (t for t in triplets if t.predicate == "isA"):
            if (tds := existing_type_defs.get(triplet.subject, None)) is not None:
                # Ensure that the subject entity does not have a type def already
                errors.append(
                    f"{triplet.as_triplet_str()}: Subject '{triplet.subject}' is already defined as type '{tds}' and cannot be redefined. To fix, omit this triplet."
                )
                continue

            if not self.ontology.has_class(triplet.object):
                # ensure ontology has class for object
                errors.append(
                    f"{triplet.as_triplet_str()}: '{triplet.object}' is not a valid ontology class! To fix this, you need to add the class '{triplet.object}' to the ontology."
                )
                continue

            if self.ontology.has_class(triplet.subject):
                # Ensure that the subject is not an ontology class
                errors.append(
                    f"{triplet.as_triplet_str()}: Subject '{triplet.subject}' can not be an ontology class! To fix this, choose a name for the subject that is not an ontology class."
                )
                continue

            new_type_defs[triplet.subject] = triplet.object

        all_type_defs = {**new_type_defs, **existing_type_defs}

        # second iteration: make sure triplets have valid type definitions
        for triplet in (t for t in triplets if t.predicate != "isA"):
            tds = all_type_defs.get(triplet.subject, None)
            tdo = all_type_defs.get(triplet.object, None)

            # TODO validate properties as well!

            if not tds:
                # Ensure that the subject entity has a type definition
                errors.append(
                    f"{triplet.as_triplet_str()}: Subject '{triplet.subject}' does not have a type definition. To fix this, add a triplet ({triplet.subject}, isA, <class>) to define the ontology class of the subject."
                )
                continue

            if self.ontology.has_class(triplet.object):
                # Ensure that the object entity is not an ontology class (this is only allowed for isA predicates!)
                errors.append(
                    f"{triplet.as_triplet_str()}: Object '{triplet.object}' can not be an ontology class! To fix this, you need to choose an object that is an entity and not an ontology class."
                )
                continue

            if not tdo:
                # Object entity does not have a type definition yet
                errors.append(
                    f"{triplet.as_triplet_str()}: Object '{triplet.object}' does not belong to a class yet! To fix this, add a triplet ({triplet.object}, isA, <class>) to define the ontology class of the object."
                )
                continue

        if errors:
            raise ValueError(
                f"Triplet extraction failed with the following errors: \n- {'\n- '.join(errors)}"
            )

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
) -> KG:
    extractor = TripletExtractor(name=kg_name, threshold=threshold, ontology=ontology)

    usage = None
    triplets = []
    with MetadataTracker() as tracker:
        for i in tqdm(range(0, len(texts), batch_size)):
            text = "\n".join(texts[i : i + batch_size])

            input_data = TripletExtractorInput(
                text=text,
                ontology=ontology,
                state=KGState(triplets=triplets) if triplets else None,
            )

            try:
                result = extractor(input=input_data)
                if result.triplets is not None:
                    new_triplets = result.triplets
                    triplets.extend(new_triplets)
                    extractor.extend_triplets(new_triplets)
            except Exception as e:
                logger.error("Error extracting triplets from text", exc_info=e)

        usage = tracker.usage
        extractor.contract_perf_stats()

    logger.debug("API Usage: %s", usage)

    return extractor.get_kg()
