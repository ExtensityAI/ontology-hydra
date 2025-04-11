import logging
import secrets
import sys
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict

from ontopipe.eval.kg import generate_kg
from ontopipe.eval.utils import InterceptHandler
from ontopipe.models import KG
from ontopipe.pipe import ontopipe

DOMAINS = [
    ("To_Kill_A_Mockingbird", "Fictional Works"),
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


class EvalParams(BaseModel):
    """Parameters for a single evaluation"""

    model_config = ConfigDict(frozen=True)

    id: str
    domain: str
    path: Path


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


def eval(params: EvalParams):
    """Run evaluation for a single domain"""

    logger.info(
        "Evaluating ontopipe on domain {} ({})",
        params.domain,
        params.id,
    )

    logger.info("Generating ontology...")
    ontology = ontopipe(params.domain, params.path)

    logger.info("Generating kg...")
    kg = _generate_kg(
        [],  # TODO add texts here!
        params.domain,
        params.path / "kg.json",
        params.path / "ontology.json",
    )

    # TODO now do evaluation with KG -> prompt model with question and kg and tell it to answer solely based on the KG (thus no answer if not in KG).


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
        "Starting evaluation {} under {}",
        run_id,
        eval_path,
    )

    # run eval for each domain
    for id, domain in DOMAINS:
        path = eval_path / id
        path.mkdir(exist_ok=True, parents=True)

        params = EvalParams(id=id, domain=domain, path=path)

        # store params in JSON file
        (path / "params.json").write_text(params.model_dump_json(indent=2))

        with logger.contextualize(domain_id=id):
            eval(params)
