import secrets
from argparse import ArgumentParser
from datetime import date
from pathlib import Path

from loguru import logger
from pydantic import BaseModel, ConfigDict

from eval.config import EvalScenario, EvalConfig
from eval.eval import eval_scenario, _generate_run_metrics_and_stats, _eval_squad_topic_existing_kg
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

    eval_existing_cmd = subparsers.add_parser("eval-existing", help="Run evaluation on existing knowledge graphs")
    eval_existing_cmd.add_argument("--config", type=Path, help="Path to the JSON config file", default=Path("eval/config.json"))
    eval_existing_cmd.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output path for the evaluation run",
        default=_generate_unique_run_path(Path("eval/runs")),
    )
    eval_existing_cmd.add_argument(
        "--kg-path",
        type=Path,
        help="Path to the existing knowledge graph JSON file",
        required=True
    )
    eval_existing_cmd.add_argument(
        "--topic",
        type=str,
        help="Topic name to evaluate (must match a topic in the config)",
        required=True
    )

    return parser.parse_args()


def _run(path: Path, config: EvalConfig):
    """Run the evaluation for the given config"""

    for scenario in config.scenarios:
        scenario_path = path / scenario.id
        scenario_path.mkdir(exist_ok=True, parents=True)

        with logger.contextualize(domain_id=scenario.id):
            eval_scenario(scenario, scenario_path, config.neo4j)

    _generate_run_metrics_and_stats(path, config)


def _run_eval_existing_kg(path: Path, config: EvalConfig, kg_path: Path, topic: str):
    """Run evaluation on an existing knowledge graph for a specific topic"""

    # Find the scenario that contains the specified topic
    target_scenario = None
    for scenario in config.scenarios:
        if topic in scenario.squad_titles:
            target_scenario = scenario
            break

    if target_scenario is None:
        raise ValueError(f"Topic '{topic}' not found in any scenario in the config")

    logger.info(f"Running evaluation on existing KG for topic '{topic}' in scenario '{target_scenario.id}'")

    # Create scenario directory
    scenario_path = path / target_scenario.id
    scenario_path.mkdir(exist_ok=True, parents=True)

    # Merge Neo4j configurations
    merged_neo4j_config = None
    if config.neo4j is not None:
        from eval.eval import _merge_neo4j_configs
        merged_neo4j_config = _merge_neo4j_configs(config.neo4j, target_scenario.neo4j)
    else:
        merged_neo4j_config = target_scenario.neo4j

    # Create topic directory
    topics_path = scenario_path / "topics"
    topics_path.mkdir(exist_ok=True, parents=True)
    topic_path = topics_path / topic
    topic_path.mkdir(exist_ok=True, parents=True)

    # Copy the existing KG to the topic path
    import shutil
    target_kg_path = topic_path / "kg.json"
    shutil.copy2(kg_path, target_kg_path)
    logger.info(f"Copied existing KG from {kg_path} to {target_kg_path}")

    # Run evaluation on the existing KG
    with logger.contextualize(domain_id=target_scenario.id):
        _eval_squad_topic_existing_kg(
            topic,
            None,  # No ontology for existing KG evaluation
            topic_path,
            merged_neo4j_config,
            target_scenario.dataset_mode,
            target_scenario.skip_qa
        )

    # Generate metrics for this single topic evaluation
    _generate_run_metrics_and_stats(path, config)


def _start_new_evaluation(args):
    """Start a new evaluation run"""

    if args.output.exists():
        raise FileExistsError(
            f"Output path '{args.output}' already exists. Please choose a different path to start a new evaluation run."
        )

    config = EvalConfig.model_validate_json(args.config.read_text(encoding='utf-8', errors='ignore'))
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


def _start_eval_existing_kg(args):
    """Start evaluation on existing knowledge graph"""

    if args.output.exists():
        raise FileExistsError(
            f"Output path '{args.output}' already exists. Please choose a different path to start a new evaluation run."
        )

    if not args.kg_path.exists():
        raise FileNotFoundError(f"Knowledge graph file '{args.kg_path}' does not exist")

    config = EvalConfig.model_validate_json(args.config.read_text(encoding='utf-8', errors='ignore'))
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
        "Starting evaluation on existing KG under '{}'",
        path,
    )

    _run_eval_existing_kg(path, config, args.kg_path, args.topic)


def _resume_evaluation(args):
    """Resume an existing evaluation run"""
    config_path = args.path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find config file under '{config_path}'. Are you sure this is a valid run path?"
        )

    config = EvalConfig.model_validate_json(config_path.read_text(encoding='utf-8', errors='ignore'))

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
    elif args.command == "eval-existing":
        _start_eval_existing_kg(args)
