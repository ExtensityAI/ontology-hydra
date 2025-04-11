import json
from pathlib import Path

from loguru import logger
from symai import Expression
from symai.components import MetadataTracker
from symai.strategy import contract
from tqdm import tqdm

from ontopipe.models import KG, KGState, Ontology, Triplet, TripletExtractorInput
from ontopipe.prompts import prompt_registry


@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=True,
    remedy_retry_params=dict(
        tries=25, delay=0.5, max_delay=15, jitter=0.1, backoff=2, graceful=False
    ),
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
        for triplet in output.triplets:
            if triplet.confidence < self.threshold:
                raise ValueError(
                    f"Confidence score {triplet.confidence} is below threshold {self.threshold}!"
                )
        return True

    @property
    def prompt(self) -> str:
        return prompt_registry.instruction("triplet_extraction")

    @staticmethod
    def load_ontology(ontology_file: Path) -> Ontology:
        if not ontology_file.exists():
            raise FileNotFoundError(f"Ontology file {ontology_file} not found")
        ont_data = json.load(open(ontology_file))
        return Ontology.model_validate(ont_data)

    def extend_triplets(self, new_triplets: list[Triplet]):
        if new_triplets:
            self._triplets.update(new_triplets)

    def get_kg(self) -> KG:
        return KG(name=self.name, triplets=self._triplets)

    def dump_kg(self, folder: Path, fname: str = "kg.json"):
        kg = self.get_kg()
        if not folder.exists():
            folder.mkdir(parents=True)
        with open(folder / fname, "w") as f:
            json.dump(kg.model_dump(), f, indent=4)


def generate_kg(
    texts: list[str],
    kg_name: str,
    ontology_file: Path,
    output_folder: Path,
    output_filename: str = "kg.json",
    threshold: float = 0.7,
    batch_size: int = 1,
) -> KG:
    extractor = TripletExtractor(name=kg_name, threshold=threshold)

    try:
        ontology = TripletExtractor.load_ontology(ontology_file)
        logger.info("Ontology successfully loaded from file.")
    except Exception as e:
        logger.error(f"Error loading ontology: {str(e)}")
        raise

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
                    logger.error(f"Error extracting triplets from text: {str(e)}")

        usage = tracker.usage
        extractor.contract_perf_stats()

    logger.info(f"\nAPI Usage:\n{usage}")
    extractor.dump_kg(output_folder, output_filename)
    logger.info(f"Knowledge graph saved to {output_folder / output_filename}")

    return extractor.get_kg()


if __name__ == "__main__":
    sample_texts = [
        "Learning to imitate expert behavior from demonstrations can be challenging, "
        "especially in environments with high-dimensional, continuous observations and "
        "unknown dynamics. Supervised learning methods based on behavioral cloning (BC) "
        "suffer from distribution shift.",
        "Recent methods based on reinforcement learning (RL), "
        "such as inverse RL and generative adversarial imitation learning (GAIL), "
        "overcome this by training an RL agent to match the demonstrations over a long horizon.",
    ]

    ROOT = Path(__file__).parent.parent
    ARTIFACTS = ROOT / "artifacts"
    ontology_file = ARTIFACTS / "ontology.json"
    kg_name = "example_kg"
    fname = "example_kg.json"
    folder = Path("/tmp/output")

    # Generate the knowledge graph from the sample texts
    kg = generate_kg(
        texts=sample_texts,
        kg_name=kg_name,
        ontology_file=ontology_file,
        output_folder=folder,
        output_filename=fname,
        threshold=0.7,
        batch_size=1,
    )

    logger.info(f"Extracted {len(kg.triplets)} triplets")
    logger.info("Triplet Extraction Completed!")
