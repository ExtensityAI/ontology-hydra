import secrets
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict

from eval.config import EvalScenario, EvalConfig
from eval.eval import eval_scenario, _generate_run_metrics_and_stats
from eval.logs import init_logging

from symai.backend.engines.neurosymbolic.engine_deepseekX_reasoning import \
    DeepSeekXReasoningEngine
from symai.backend.engines.neurosymbolic.engine_google_geminiX_reasoning import \
    GeminiXReasoningEngine
from symai.backend.engines.neurosymbolic.engine_openai_gptX_chat import \
    GPTXChatEngine
from symai.backend.engines.neurosymbolic.engine_openai_gptX_reasoning import \
    GPTXReasoningEngine
from symai.functional import EngineRepository


def _generate_run_id():
    """Generate a run ID based on the current date"""
    return date.today().strftime("%Y%m%d") + "_" + secrets.token_urlsafe(4).replace("_", "-")


def _generate_unique_run_path(path: Path):
    """Generate a new run path for which no directory exists in the given path"""
    run_id = None

    while run_id is None or (path / run_id).exists():
        run_id = _generate_run_id()

    return path / run_id


def _parse_args():
    parser = ArgumentParser(description="ontopipe evaluation script")
    subparsers = parser.add_subparsers(dest="command", required=True)

    new_cmd = subparsers.add_parser("new", help="Start a new evaluation run")
    new_cmd.add_argument("--config", type=Path, help="Path to the JSON config file", default=Path("eval/config.json"))
    new_cmd.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for the evaluation run",
        default=_generate_unique_run_path(Path("eval/runs")),
    )

    resume_cmd = subparsers.add_parser("resume", help="Resume an existing evaluation run")
    resume_cmd.add_argument("--path", type=Path, help="Path to the evaluation run", required=True)

    return parser.parse_args()


def _run(path: Path, config: EvalConfig):
    """Run the evaluation for the given config"""

    for scenario in config.scenarios:
        scenario_path = path / scenario.id
        scenario_path.mkdir(exist_ok=True, parents=True)

        with logger.contextualize(domain_id=scenario.id):
            eval_scenario(scenario, scenario_path, config.neo4j)

    _generate_run_metrics_and_stats(path, config)


def _start_new_evaluation(args):
    """Start a new evaluation run"""

    if args.output.exists():
        raise FileExistsError(
            f"Output path '{args.output}' already exists. Please choose a different path to start a new evaluation run."
        )

    config = EvalConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
    path = args.output

    if config.model.engine.startswith('gpt-4.1'):
        engine = GPTXChatEngine(
            api_key=config.model.api_key,
            model=config.model.engine
        )
        EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)
    elif config.model.engine == 'o4-mini' or config.model.engine.startswith('o3'):
        engine = GPTXReasoningEngine(
            api_key=config.model.api_key,
            model=config.model.engine
        )
        EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)
    elif config.model.engine.startswith('deepseek'):
        engine = DeepSeekXReasoningEngine(
            api_key=config.model.api_key,
            model=config.model.engine
        )
        EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)
    elif config.model.engine.startswith('gemini'):
        engine = GeminiXReasoningEngine(
            api_key=config.model.api_key,
            model=config.model.engine
        )
        EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)
    else:
        engine = GPTXChatEngine(
            api_key=config.model.api_key,
            model=config.model.engine
        )
        EngineRepository.register('neurosymbolic', engine, allow_engine_override=True)

    # prepare output path
    path.mkdir(exist_ok=True, parents=True)
    (path / "config.json").write_text(config.model_dump_json(indent=2), encoding="utf-8")

    log_dir_path = path / "logs"
    log_dir_path.mkdir(exist_ok=True, parents=True)
    init_logging(log_dir_path)

    logger.info(
        "Starting new evaluation under '{}'",
        path,
    )

    _run(path, config)


def _resume_evaluation(args):
    """Resume an existing evaluation run"""
    config_path = args.path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find config file under '{config_path}'. Are you sure this is a valid run path?"
        )

    config = EvalConfig.model_validate_json(config_path.read_text(encoding="utf-8"))

    init_logging(args.path / "logs")

    logger.info(
        "Resuming evaluation under '{}'",
        args.path,
    )

    _run(args.path, config)


def main():
    args = _parse_args()

    if args.command == "new":
        _start_new_evaluation(args)
    elif args.command == "resume":
        _resume_evaluation(args)
