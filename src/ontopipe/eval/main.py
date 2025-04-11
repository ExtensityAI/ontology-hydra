import json
import logging
import secrets
import sys
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from ontopipe.eval.kg import generate_kg
from ontopipe.eval.utils import InterceptHandler
from ontopipe.models import KG
from ontopipe.pipe import ontopipe

SQUAD_TRAIN_DATASET_PATH = Path("eval/train-v2.0.json")

if not SQUAD_TRAIN_DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Could not find SQuAD dataset at {SQUAD_TRAIN_DATASET_PATH}. "
        "Please download the dataset from https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json "
        "and place it in the eval directory."
    )

SQUAD_TRAIN_DATASET = json.loads(SQUAD_TRAIN_DATASET_PATH.read_text(encoding="utf-8"))


class EvalScenario(BaseModel):
    id: str

    domain: str
    """The domain used for ontology creation"""

    squad_titles: list[str]
    """Titles of topics in the SQuAD dataset to use for evaluation (title field)"""
    # intuition: we create an ontology for the domain, create KGs for each SQuAD topic based on the associated texts and the ontology and then evaluate using SQuAD questions


SCENARIOS = [
    EvalScenario(id="fiction", domain="Fiction Books", squad_titles=["To_Kill_A_Mockingbird"]),
]


def _init_logging(eval_path: Path):
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.getLogger("openai._base_client").setLevel(logging.WARN)  # ignore unnecessary OpenAI logs

    def _only_ontopipe(record) -> bool:
        return "ontopipe" in record["name"]

    logger.remove()

    logger.add(sys.stderr, colorize=True, filter=_only_ontopipe)
    logger.add(eval_path / "eval_{time}.log", rotation="300 MB")


def _generate_run_id():
    """Generate a run ID based on the current date"""
    return date.today().strftime("%Y%m%d") + "_" + secrets.token_urlsafe(4)


def _parse_args():
    parser = ArgumentParser(description="Run evaluation")

    parser.add_argument(
        "--path",
        type=Path,
        help="Base path for evaluation runs",
        default=Path("eval/runs"),
    )

    parser.add_argument(
        "--run-id",
        type=str,
        help="Run ID for evaluation",
        default=None,
    )

    return parser.parse_args()


def _generate_kg(texts: list[str], domain: str, kg_path: Path, ontology_path: Path):
    """Generate a knowledge graph from a list of texts as well as the current ontology and domain"""

    if kg_path.exists():
        # load cached KG
        return KG.model_validate_json(kg_path.read_text(encoding="utf-8"))

    return generate_kg(
        texts,
        domain,
        ontology_file=ontology_path,
        output_folder=kg_path.parent,
        output_filename=kg_path.name,
    )


def _load_squad_contexts(title: str):
    topic = next((item for item in SQUAD_TRAIN_DATASET["data"] if item["title"].lower() == title.lower()), None)

    if topic is None:
        raise ValueError(f"Could not find topic with title '{title}' in SQuAD dataset")

    return [p["context"] for p in topic["paragraphs"] if "context" in p]


def _eval_squad_topic(title: str, scenario: EvalScenario, ontology_path: Path, path: Path):
    logger.info("Generating kg for '{}'", title)

    contexts = _load_squad_contexts(title)
    logger.debug(
        "Found {} contexts with {} words for topic '{}'", len(contexts), sum(len(c.split()) for c in contexts), title
    )

    kg = _generate_kg(
        contexts,  # TODO add texts here!
        title,
        path / "kg.json",
        ontology_path,
    )


def eval(scenario: EvalScenario, path: Path):
    """Run evaluation for a single scenario"""

    logger.info(
        "Evaluating ontopipe on scenario '{}'",
        scenario.id,
    )

    logger.info("Generating ontology...")
    ontology = ontopipe(scenario.domain, path)

    topics_path = path / "topics"
    topics_path.mkdir(exist_ok=True, parents=True)

    ontology_path = path / "ontology.json"

    for title in scenario.squad_titles:
        topic_path = topics_path / title
        topic_path.mkdir(exist_ok=True, parents=True)

        _eval_squad_topic(title, scenario, ontology_path, topic_path)


def _generate_unique_run_id(path: Path) -> str:
    """Generate a new run ID for which no directory exists in the given path"""
    run_id = None

    while run_id is None or (path / run_id).exists():
        run_id = _generate_run_id()

    return run_id


def main():
    args = _parse_args()

    run_id = args.run_id or _generate_unique_run_id(args.path)
    eval_path = args.path / run_id
    eval_path.mkdir(exist_ok=True, parents=True)

    _init_logging(eval_path)

    logger.info(
        "Starting evaluation '{}' under '{}'",
        run_id,
        eval_path,
    )

    # run eval for each domain
    for scenario in SCENARIOS:
        path = eval_path / scenario.id
        path.mkdir(exist_ok=True, parents=True)

        # store scenario in JSON file
        # TODO this is overwritten when rerunning the evaluation, should be fixed
        (path / "scenario.json").write_text(scenario.model_dump_json(indent=2))

        with logger.contextualize(domain_id=id):
            eval(scenario, path)
