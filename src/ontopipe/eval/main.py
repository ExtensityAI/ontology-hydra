import logging
import secrets
import sys
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

from loguru import logger

from ontopipe.eval.eval import EvalScenario, eval_scenario
from ontopipe.eval.utils import InterceptHandler

SCENARIOS = (EvalScenario(id="computer", domain="Computers", squad_titles=["Computer"]),)


def _init_logging(log_dir_path: Path):
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.getLogger("openai._base_client").setLevel(logging.WARN)  # ignore unnecessary OpenAI logs

    def _only_ontopipe(record) -> bool:
        return "ontopipe" in record["name"]

    logger.remove()

    logger.add(sys.stderr, colorize=True, filter=_only_ontopipe)
    logger.add(log_dir_path / "eval_{time}.log", rotation="300 MB")


def _generate_run_id():
    """Generate a run ID based on the current date"""
    return date.today().strftime("%Y%m%d") + "_" + secrets.token_urlsafe(4).replace("_", "-")


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

    log_dir_path = eval_path / "logs"
    log_dir_path.mkdir(exist_ok=True, parents=True)
    _init_logging(log_dir_path)

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
        # TODO this is overwritten when rerunning the evaluation, should this be changed?
        (path / "scenario.json").write_text(scenario.model_dump_json(indent=2))

        with logger.contextualize(domain_id=id):
            eval_scenario(scenario, path)
