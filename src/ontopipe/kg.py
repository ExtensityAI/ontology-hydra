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
    remedy_retry_params=dict(tries=25, delay=0.5, max_delay=15, jitter=0.1, backoff=2, graceful=False),
)
class TripletExtractor(Expression):
    def __init__(self, name: str, threshold: float = 0.7, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self.threshold = threshold
        self._triplets = set()

    def forward(self, input: TripletExtractorInput, **kwargs) -> KGState:
        if self.contract_result is None:
            raise ValueError("Contract failed!")
        return self.contract_result

    def pre(self, input: TripletExtractorInput) -> bool:
        return True

    def post(self, output: KGState) -> bool:
        if output.triplets is None:
            return True  # Nothing was extracted.
        """for triplet in output.triplets:
            if triplet.confidence < self.threshold:
                raise ValueError(f"Confidence score {triplet.confidence} is below threshold {self.threshold}!")"""
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
    extractor = TripletExtractor(name=kg_name, threshold=threshold)

    usage = None
    triplets = []
    with MetadataTracker() as tracker:
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i : i + batch_size]
            for text in batch_texts:
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
