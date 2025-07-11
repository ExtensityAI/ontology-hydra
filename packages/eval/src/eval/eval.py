import json
import time
from pathlib import Path

import openai
from loguru import logger
from ontopipe import ontopipe
from ontopipe.kg import generate_kg
from ontopipe.models import KG, Ontology
from ontopipe.vis import visualize_kg, visualize_ontology
from pydantic import BaseModel
from symai.components import MetadataTracker
from symai.utils import RuntimeInfo
from tqdm import tqdm

from eval.config import EvalConfig, EvalScenario
from eval.neo4j_eval import Neo4jConfig, _eval_neo4j_qa
from eval.squad_v2.data import SquadDataset, SquadQAPair
from eval.squad_v2.squad_v2 import SquadV2

KG_BATCH_SIZE = 4
QA_BATCH_SIZE = 4
NO_ANSWER_THRESHOLD = 0.5

squadv2 = SquadV2()


def get_dataset_path(mode: str, topic: str) -> Path:
    """Get the dataset path for a given mode and topic."""
    base_path = Path(__file__).parent.parent.parent.parent.parent / "MedExQA" / mode
    return base_path / f"{topic.lower().replace(' ', '_')}_{mode}.json"


def load_dataset(mode: str, topic: str) -> SquadDataset:
    """Load the SQuAD dataset for a given mode and topic."""
    dataset_path = get_dataset_path(mode, topic)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Could not find SQuAD {mode} dataset at {dataset_path}. "
            "Please follow the instructions in the README."
        )

    return SquadDataset.model_validate_json(
        dataset_path.read_text(encoding="utf-8", errors="ignore")
    )


def _generate_kg(
    texts: list[str],
    domain: str,
    kg_path: Path,
    ontology: Ontology | None = None,
    epochs: int = 3,
):
    """Generate a knowledge graph from a list of texts as well as the current ontology and domain"""

    if kg_path.exists():
        # load cached KG
        return KG.model_validate_json(
            kg_path.read_text(encoding="utf-8", errors="ignore")
        )

    # Track runtime for KG generation
    with MetadataTracker() as tracker:
        start_time = time.perf_counter()
        try:
            kg = generate_kg(
                kg_path,
                texts,
                domain,
                ontology=ontology,
                batch_size=KG_BATCH_SIZE,  # TODO hyperparam
                epochs=epochs,
            )
        finally:
            end_time = time.perf_counter()

    # Save runtime statistics for KG generation
    kg_runtime_path = kg_path.parent / "kg_runtime_stats.json"
    kg_runtime_stats = {
        "total_elapsed_time_seconds": end_time - start_time,
        "total_prompt_tokens": tracker.usage.get("prompt_tokens", 0)
        if hasattr(tracker, "usage")
        else 0,
        "total_completion_tokens": tracker.usage.get("completion_tokens", 0)
        if hasattr(tracker, "usage")
        else 0,
        "total_tokens": tracker.usage.get("total_tokens", 0)
        if hasattr(tracker, "usage")
        else 0,
        "total_calls": tracker.usage.get("calls", 0)
        if hasattr(tracker, "usage")
        else 0,
    }

    with open(kg_runtime_path, "w") as f:
        json.dump(kg_runtime_stats, f, indent=2)

    return kg


class ResponseDetail(BaseModel):
    qid: str
    """The ID of the question"""

    answer: str | None
    """The answer to the question, if there is one"""

    no_answer_prob: float
    """Probability that there is no answer (impossible question) (between 0 and 1)"""

    no_data_found: bool
    """True if the knowledge graph does not contain any data for this question"""


class ResponseDetails(BaseModel):
    responses: list[ResponseDetail]


def _answer_questions(kg: KG, qas: list[SquadQAPair]):
    """Prompt the model to answer questions based on the knowledge graph"""

    questions_str = "\n".join([f"{i}: " + qa.question for i, qa in enumerate(qas)])

    # Track runtime for QA
    with MetadataTracker() as tracker:
        start_time = time.perf_counter()
        try:
            response = openai.beta.chat.completions.parse(
                model="gpt-4.1-mini",  # TODO use contracts for this as well
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are answering questions based ONLY on the provided knowledge graph. "
                            "Give concise, clear answers. If a question is impossible to answer, indicate this with a high no_answer_prop. If the knowledge graph does not contain information to answer, indicate this with no_data_found. Never invent information.\n"
                            f"Knowledge graph: {kg.model_dump_json()}"
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Answer these questions precisely using only the knowledge graph data:\n\n{questions_str}",
                    },
                ],
                response_format=ResponseDetails,
            )
        finally:
            end_time = time.perf_counter()

    details = response.choices[0].message.parsed

    if details is None:
        raise ValueError("Failed to generate QA answers")

    assert (
        len(details.responses) == len(qas)
    ), f"Number of responses ({len(details.responses)}) does not match number of questions ({len(qas)})"

    for detail in details.responses:
        # reset relative question id
        detail.qid = qas[int(detail.qid)].id

    # Save runtime statistics for this QA batch
    qa_runtime_stats = {
        "total_elapsed_time_seconds": end_time - start_time,
        "total_prompt_tokens": tracker.usage.get("prompt_tokens", 0)
        if hasattr(tracker, "usage")
        else 0,
        "total_completion_tokens": tracker.usage.get("completion_tokens", 0)
        if hasattr(tracker, "usage")
        else 0,
        "total_tokens": tracker.usage.get("total_tokens", 0)
        if hasattr(tracker, "usage")
        else 0,
        "total_calls": tracker.usage.get("calls", 0)
        if hasattr(tracker, "usage")
        else 0,
        "questions_processed": len(qas),
    }

    return details.responses, qa_runtime_stats


def _answer_all_questions(kg: KG, qas: list[SquadQAPair], cache_path: Path):
    """Answer all questions in the SQuAD dataset based on the knowledge graph"""

    details = list[ResponseDetail]()
    total_qa_runtime_stats = {
        "total_elapsed_time_seconds": 0.0,
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_calls": 0,
        "total_questions_processed": 0,
    }

    if cache_path.exists():
        # load cached responses
        details = ResponseDetails.model_validate_json(
            cache_path.read_text(encoding="utf-8", errors="ignore")
        ).responses
        logger.debug("Loaded {} cached answers from {}", len(details), cache_path)

    logger.debug("Answering {} questions", len(qas))

    # start at the end of the cached responses (should be the same, TODO maybe check that!)
    for i in tqdm(
        range(len(details), len(qas), QA_BATCH_SIZE), desc="Answering questions"
    ):
        qas_batch = qas[i : i + QA_BATCH_SIZE]

        batch_details, batch_runtime_stats = _answer_questions(kg, qas_batch)
        details.extend(batch_details)

        # Aggregate runtime statistics
        total_qa_runtime_stats["total_elapsed_time_seconds"] += batch_runtime_stats[
            "total_elapsed_time_seconds"
        ]
        total_qa_runtime_stats["total_prompt_tokens"] += batch_runtime_stats[
            "total_prompt_tokens"
        ]
        total_qa_runtime_stats["total_completion_tokens"] += batch_runtime_stats[
            "total_completion_tokens"
        ]
        total_qa_runtime_stats["total_tokens"] += batch_runtime_stats["total_tokens"]
        total_qa_runtime_stats["total_calls"] += batch_runtime_stats["total_calls"]
        total_qa_runtime_stats["total_questions_processed"] += batch_runtime_stats[
            "questions_processed"
        ]

        # save intermediate results
        cache_path.write_text(
            ResponseDetails(responses=details).model_dump_json(indent=2),
            encoding="utf-8",
        )

    # Save aggregated QA runtime statistics
    qa_runtime_path = cache_path.parent / "qa_runtime_stats.json"
    with open(qa_runtime_path, "w") as f:
        json.dump(total_qa_runtime_stats, f, indent=2)

    return details


def _eval_squad_topic(
    title: str,
    ontology: Ontology | None,
    path: Path,
    neo4j_config: Neo4jConfig,
    dataset_mode: str,
    skip_qa: bool = True,
    epochs: int = 3,
):
    """Evaluate QA performance on a specific scenario consisting of multiple SQuAD topics using the generated ontology."""

    logger.info("Generating kg for '{}' using {} dataset", title, dataset_mode)

    # Load the appropriate dataset based on mode
    dataset = load_dataset(dataset_mode, title)
    topic = dataset.find_topic(title)

    assert (
        topic is not None
    ), f"Topic '{title}' not found in SQuAD {dataset_mode} dataset"

    contexts = topic.contexts

    logger.debug(
        "Found {} contexts with {} words for topic '{}'",
        len(contexts),
        sum(len(c.split()) for c in contexts),
        title,
    )

    kg = _generate_kg(contexts, title, path / "kg.json", ontology, epochs)

    visualize_kg(kg, path / "kg.html", ontology)

    logger.debug("KG generated with {} triplets", len(kg.triplets))

    qas = topic.qas

    logger.debug("Found {} questions for topic '{}'", len(qas), title)

    if skip_qa:
        logger.info("Skipping question answering evaluation")
        # Create empty metrics for consistency
        empty_metrics = {
            "exact_match": 0.0,
            "f1": 0.0,
            "no_answer_probability": 0.0,
            "skip_qa": True,
        }
        (path / "metrics.json").write_text(
            json.dumps(empty_metrics, indent=2), encoding="utf-8"
        )
        logger.debug("Saved empty evaluation results to '{}'", path / "metrics.json")
    else:
        qa_cache_path = path / "qas.json"

        details = _answer_all_questions(kg, qas, qa_cache_path)

        # TODO IMPORTANT how do we ensure that answers are sourced only from the KG?

        # TODO also save preds and refs to file?

        predictions, references = _format_predictions(details, qas)
        metrics = squadv2.compute(predictions=predictions, references=references)
        logger.info("Evaluation results for topic '{}':", title)
        logger.info("{}", metrics)

        # save results to file
        (path / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        logger.debug("Saved evaluation results to '{}'", path / "metrics.json")

    # --- Neo4j evaluation ---
    _eval_neo4j_qa(kg, qas, neo4j_config, path)


def _eval_squad_topic_existing_kg(
    title: str,
    ontology: Ontology | None,
    path: Path,
    neo4j_config: Neo4jConfig,
    dataset_mode: str,
    skip_qa: bool = True,
):
    """Evaluate QA performance on a specific topic using an existing knowledge graph."""

    logger.info("Loading existing kg for '{}' using {} dataset", title, dataset_mode)

    # Load the existing knowledge graph
    kg_path = path / "kg.json"
    if not kg_path.exists():
        raise FileNotFoundError(f"Knowledge graph file not found at {kg_path}")

    kg = KG.model_validate_json(kg_path.read_text(encoding="utf-8", errors="ignore"))
    logger.info("Loaded existing KG with {} triplets", len(kg.triplets))

    # Load the appropriate dataset based on mode
    dataset = load_dataset(dataset_mode, title)
    topic = dataset.find_topic(title)

    assert (
        topic is not None
    ), f"Topic '{title}' not found in SQuAD {dataset_mode} dataset"

    # Visualize the existing KG
    visualize_kg(kg, path / "kg.html", ontology)

    qas = topic.qas

    logger.debug("Found {} questions for topic '{}'", len(qas), title)

    if skip_qa:
        logger.info("Skipping question answering evaluation")
        # Create empty metrics for consistency
        empty_metrics = {
            "exact_match": 0.0,
            "f1": 0.0,
            "no_answer_probability": 0.0,
            "skip_qa": True,
        }
        (path / "metrics.json").write_text(
            json.dumps(empty_metrics, indent=2), encoding="utf-8"
        )
        logger.debug("Saved empty evaluation results to '{}'", path / "metrics.json")
    else:
        qa_cache_path = path / "qas.json"

        details = _answer_all_questions(kg, qas, qa_cache_path)

        # TODO IMPORTANT how do we ensure that answers are sourced only from the KG?

        # TODO also save preds and refs to file?

        predictions, references = _format_predictions(details, qas)
        metrics = squadv2.compute(predictions=predictions, references=references)
        logger.info("Evaluation results for topic '{}':", title)
        logger.info("{}", metrics)

        # save results to file
        (path / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        logger.debug("Saved evaluation results to '{}'", path / "metrics.json")

    # --- Neo4j evaluation ---
    _eval_neo4j_qa(kg, qas, neo4j_config, path)


def _format_predictions(details: list[ResponseDetail], qas: list[SquadQAPair]):
    """Format the predictions for squad_v2.py evaluation script"""

    predictions = [
        {
            "id": detail.qid,
            "prediction_text": detail.answer or "",
            "no_answer_probability": detail.no_answer_prob,
        }
        for detail in details
    ]

    # exclude two fields as we get an error when running SquadV2.compute else
    references = [qa.model_dump(exclude={"is_impossible", "question"}) for qa in qas]

    # sort by id to ensure correct order
    predictions.sort(key=lambda x: x["id"])
    references.sort(key=lambda x: x["id"])

    # ensure everything is aligned correctly!
    assert (
        len(predictions) == len(references)
    ), f"Number of predictions ({len(predictions)}) does not match number of references ({len(references)})"

    for i, (p, r) in enumerate(zip(predictions, references)):
        assert (
            p["id"] == r["id"]
        ), f"Formatting predictions failed as prediction ID '{p['id']}' does not match reference ID '{r['id']}' at index {i}."

    return predictions, references


def _merge_neo4j_configs(
    global_config: Neo4jConfig, scenario_config: Neo4jConfig
) -> Neo4jConfig:
    """Merge global Neo4j configuration with scenario-specific configuration.
    Scenario-specific settings override global settings."""

    # Start with global config
    merged_data = global_config.model_dump()

    # Override with scenario-specific settings if they are not default values
    scenario_data = scenario_config.model_dump()

    for key, value in scenario_data.items():
        # Only override if the scenario value is different from the default Neo4jConfig value
        default_config = Neo4jConfig()
        default_value = getattr(default_config, key)

        if value != default_value:
            merged_data[key] = value

    return Neo4jConfig(**merged_data)


def eval_scenario(
    scenario: EvalScenario, path: Path, global_neo4j_config: Neo4jConfig = None
):
    """Run evaluation for a single scenario"""

    logger.info(
        "Evaluating ontopipe on scenario '{}' with {} dataset",
        scenario.id,
        scenario.dataset_mode,
    )

    # Merge global and scenario-specific Neo4j configurations
    if global_neo4j_config is not None:
        merged_neo4j_config = _merge_neo4j_configs(global_neo4j_config, scenario.neo4j)
        logger.info(f"Using merged Neo4j configuration for scenario '{scenario.id}'")
        logger.debug(
            f"Global Neo4j enabled: {global_neo4j_config.enabled}, Scenario Neo4j enabled: {scenario.neo4j.enabled}, Merged enabled: {merged_neo4j_config.enabled}"
        )
    else:
        merged_neo4j_config = scenario.neo4j
        logger.info(
            f"Using scenario-specific Neo4j configuration for scenario '{scenario.id}'"
        )

    ontology = None
    ontology_runtime_stats = {}

    if scenario.domain is not None:
        logger.info("Generating ontology...")

        # Track runtime for ontology generation
        with MetadataTracker() as tracker:
            start_time = time.perf_counter()
            try:
                ontology = ontopipe(scenario.domain, cache_path=path)
            finally:
                end_time = time.perf_counter()

        # Save ontology runtime statistics
        ontology_runtime_stats = {
            "total_elapsed_time_seconds": end_time - start_time,
            "total_prompt_tokens": tracker.usage.get("prompt_tokens", 0)
            if hasattr(tracker, "usage")
            else 0,
            "total_completion_tokens": tracker.usage.get("completion_tokens", 0)
            if hasattr(tracker, "usage")
            else 0,
            "total_tokens": tracker.usage.get("total_tokens", 0)
            if hasattr(tracker, "usage")
            else 0,
            "total_calls": tracker.usage.get("calls", 0)
            if hasattr(tracker, "usage")
            else 0,
        }

        ontology_runtime_path = path / "ontology_runtime_stats.json"
        with open(ontology_runtime_path, "w") as f:
            json.dump(ontology_runtime_stats, f, indent=2)

        visualize_ontology(ontology, path / "ontology.html")
    else:
        logger.info("No domain specified, skipping ontology generation")

    topics_path = path / "topics"
    topics_path.mkdir(exist_ok=True, parents=True)

    for title in scenario.squad_titles:
        topic_path = topics_path / title
        topic_path.mkdir(exist_ok=True, parents=True)

        _eval_squad_topic(
            title,
            ontology,
            topic_path,
            merged_neo4j_config,
            scenario.dataset_mode,
            scenario.skip_qa,
            scenario.epochs,
        )


def _generate_run_metrics_and_stats(path: Path, config: EvalConfig):
    """Generate comprehensive metrics and runtime statistics for the entire evaluation run"""

    logger.info("Generating overall run metrics and runtime statistics...")

    # Initialize aggregated metrics
    run_metrics = {
        "total_scenarios": len(config.scenarios),
        "total_topics": sum(
            len(scenario.squad_titles) for scenario in config.scenarios
        ),
        "scenarios_with_ontology": sum(
            1 for scenario in config.scenarios if scenario.domain is not None
        ),
        "scenarios_without_ontology": sum(
            1 for scenario in config.scenarios if scenario.domain is None
        ),
        "scenarios": [],
    }

    # Calculate Neo4j enabled scenarios based on merged configuration
    neo4j_enabled_scenarios = 0
    for scenario in config.scenarios:
        if config.neo4j is not None:
            merged_config = _merge_neo4j_configs(config.neo4j, scenario.neo4j)
            if merged_config.enabled:
                neo4j_enabled_scenarios += 1
        else:
            if scenario.neo4j.enabled:
                neo4j_enabled_scenarios += 1

    run_metrics["neo4j_enabled_scenarios"] = neo4j_enabled_scenarios

    # Initialize aggregated runtime statistics
    total_runtime_info = RuntimeInfo(0, 0, 0, 0, 0, 0, 0, 0)

    # Process each scenario
    for scenario in config.scenarios:
        scenario_path = path / scenario.id

        if not scenario_path.exists():
            logger.warning(f"Scenario path {scenario_path} does not exist, skipping...")
            continue

        # Determine Neo4j enabled status based on merged configuration
        if config.neo4j is not None:
            merged_neo4j_config = _merge_neo4j_configs(config.neo4j, scenario.neo4j)
            neo4j_enabled = merged_neo4j_config.enabled
        else:
            neo4j_enabled = scenario.neo4j.enabled

        scenario_metrics = {
            "id": scenario.id,
            "domain": scenario.domain,
            "dataset_mode": scenario.dataset_mode,
            "neo4j_enabled": neo4j_enabled,
            "topics": [],
            "total_questions": 0,
            "total_kg_triplets": 0,
            "squad_metrics": {
                "exact_match": 0.0,
                "f1": 0.0,
                "no_answer_probability": 0.0,
            },
            "neo4j_metrics": {
                "successful_queries": 0,
                "total_queries": 0,
                "success_rate": 0.0,
                "queries_with_results": 0,
                "results_rate": 0.0,
                "correct_queries": 0,
                "accuracy": 0.0,
            },
        }

        topics_path = scenario_path / "topics"
        if not topics_path.exists():
            logger.warning(
                f"Topics path {topics_path} does not exist, skipping scenario {scenario.id}"
            )
            continue

        # Process each topic in the scenario
        for title in scenario.squad_titles:
            topic_path = topics_path / title

            if not topic_path.exists():
                logger.warning(
                    f"Topic path {topic_path} does not exist, skipping topic {title}"
                )
                continue

            topic_metrics = {
                "title": title,
                "questions": 0,
                "kg_triplets": 0,
                "squad_metrics": {},
                "neo4j_metrics": {},
            }

            # Load SQuAD metrics
            squad_metrics_path = topic_path / "metrics.json"
            if squad_metrics_path.exists():
                try:
                    squad_metrics = json.loads(
                        squad_metrics_path.read_text(encoding="utf-8", errors="ignore")
                    )
                    topic_metrics["squad_metrics"] = squad_metrics

                    # Aggregate SQuAD metrics for scenario
                    scenario_metrics["squad_metrics"]["exact_match"] += (
                        squad_metrics.get("exact_match", 0.0)
                    )
                    scenario_metrics["squad_metrics"]["f1"] += squad_metrics.get(
                        "f1", 0.0
                    )
                    scenario_metrics["squad_metrics"]["no_answer_probability"] += (
                        squad_metrics.get("no_answer_probability", 0.0)
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load SQuAD metrics from {squad_metrics_path}: {e}"
                    )

            # Load KG data
            kg_path = topic_path / "kg.json"
            if kg_path.exists():
                try:
                    kg_data = json.loads(
                        kg_path.read_text(encoding="utf-8", errors="ignore")
                    )
                    topic_metrics["kg_triplets"] = len(kg_data.get("triplets", []))
                    scenario_metrics["total_kg_triplets"] += topic_metrics[
                        "kg_triplets"
                    ]
                except Exception as e:
                    logger.warning(f"Failed to load KG data from {kg_path}: {e}")

            # Load Neo4j metrics if enabled
            if neo4j_enabled:
                neo4j_metrics_path = topic_path / "neo4j_eval" / "neo4j_metrics.json"
                if neo4j_metrics_path.exists():
                    try:
                        neo4j_metrics = json.loads(
                            neo4j_metrics_path.read_text(
                                encoding="utf-8", errors="ignore"
                            )
                        )
                        topic_metrics["neo4j_metrics"] = neo4j_metrics

                        # Aggregate Neo4j metrics for scenario
                        scenario_metrics["neo4j_metrics"]["successful_queries"] += (
                            neo4j_metrics.get("successful_queries", 0)
                        )
                        scenario_metrics["neo4j_metrics"]["total_queries"] += (
                            neo4j_metrics.get("total_queries", 0)
                        )
                        scenario_metrics["neo4j_metrics"]["queries_with_results"] += (
                            neo4j_metrics.get("queries_with_results", 0)
                        )
                        scenario_metrics["neo4j_metrics"]["correct_queries"] += (
                            neo4j_metrics.get("correct_queries", 0)
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to load Neo4j metrics from {neo4j_metrics_path}: {e}"
                        )

                # Load Neo4j runtime stats
                neo4j_runtime_path = (
                    topic_path / "neo4j_eval" / "neo4j_runtime_stats.csv"
                )
                if neo4j_runtime_path.exists():
                    try:
                        import pandas as pd

                        runtime_df = pd.read_csv(neo4j_runtime_path)

                        # Extract runtime values and add to total
                        for _, row in runtime_df.iterrows():
                            if row["metric"] == "total_elapsed_time_seconds":
                                total_runtime_info.total_elapsed_time += row["value"]
                            elif row["metric"] == "total_prompt_tokens":
                                total_runtime_info.prompt_tokens += row["value"]
                            elif row["metric"] == "total_completion_tokens":
                                total_runtime_info.completion_tokens += row["value"]
                            elif row["metric"] == "total_reasoning_tokens":
                                total_runtime_info.reasoning_tokens += row["value"]
                            elif row["metric"] == "total_cached_tokens":
                                total_runtime_info.cached_tokens += row["value"]
                            elif row["metric"] == "total_tokens":
                                total_runtime_info.total_tokens += row["value"]
                            elif row["metric"] == "total_calls":
                                total_runtime_info.total_calls += row["value"]
                            elif row["metric"] == "estimated_cost_usd":
                                total_runtime_info.cost_estimate += row["value"]
                    except Exception as e:
                        logger.warning(
                            f"Failed to load Neo4j runtime stats from {neo4j_runtime_path}: {e}"
                        )

            # Load KG runtime stats
            kg_runtime_path = topic_path / "kg_runtime_stats.json"
            if kg_runtime_path.exists():
                try:
                    kg_runtime_stats = json.loads(
                        kg_runtime_path.read_text(encoding="utf-8", errors="ignore")
                    )
                    total_runtime_info.total_elapsed_time += kg_runtime_stats.get(
                        "total_elapsed_time_seconds", 0
                    )
                    total_runtime_info.prompt_tokens += kg_runtime_stats.get(
                        "total_prompt_tokens", 0
                    )
                    total_runtime_info.completion_tokens += kg_runtime_stats.get(
                        "total_completion_tokens", 0
                    )
                    total_runtime_info.total_tokens += kg_runtime_stats.get(
                        "total_tokens", 0
                    )
                    total_runtime_info.total_calls += kg_runtime_stats.get(
                        "total_calls", 0
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load KG runtime stats from {kg_runtime_path}: {e}"
                    )

            # Load QA runtime stats
            qa_runtime_path = topic_path / "qa_runtime_stats.json"
            if qa_runtime_path.exists():
                try:
                    qa_runtime_stats = json.loads(
                        qa_runtime_path.read_text(encoding="utf-8", errors="ignore")
                    )
                    total_runtime_info.total_elapsed_time += qa_runtime_stats.get(
                        "total_elapsed_time_seconds", 0
                    )
                    total_runtime_info.prompt_tokens += qa_runtime_stats.get(
                        "total_prompt_tokens", 0
                    )
                    total_runtime_info.completion_tokens += qa_runtime_stats.get(
                        "total_completion_tokens", 0
                    )
                    total_runtime_info.total_tokens += qa_runtime_stats.get(
                        "total_tokens", 0
                    )
                    total_runtime_info.total_calls += qa_runtime_stats.get(
                        "total_calls", 0
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load QA runtime stats from {qa_runtime_path}: {e}"
                    )

            # Load question count from SQuAD dataset
            try:
                dataset = load_dataset(scenario.dataset_mode, title)
                topic = dataset.find_topic(title)
                if topic:
                    topic_metrics["questions"] = len(topic.qas)
                    scenario_metrics["total_questions"] += topic_metrics["questions"]
            except Exception as e:
                logger.warning(f"Failed to load question count for topic {title}: {e}")

            scenario_metrics["topics"].append(topic_metrics)

        # Load ontology runtime stats if domain was specified
        if scenario.domain is not None:
            ontology_runtime_path = scenario_path / "ontology_runtime_stats.json"
            if ontology_runtime_path.exists():
                try:
                    ontology_runtime_stats = json.loads(
                        ontology_runtime_path.read_text(
                            encoding="utf-8", errors="ignore"
                        )
                    )
                    total_runtime_info.total_elapsed_time += ontology_runtime_stats.get(
                        "total_elapsed_time_seconds", 0
                    )
                    total_runtime_info.prompt_tokens += ontology_runtime_stats.get(
                        "total_prompt_tokens", 0
                    )
                    total_runtime_info.completion_tokens += ontology_runtime_stats.get(
                        "total_completion_tokens", 0
                    )
                    total_runtime_info.total_tokens += ontology_runtime_stats.get(
                        "total_tokens", 0
                    )
                    total_runtime_info.total_calls += ontology_runtime_stats.get(
                        "total_calls", 0
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to load ontology runtime stats from {ontology_runtime_path}: {e}"
                    )

        # Calculate averages for scenario metrics
        num_topics = len(scenario_metrics["topics"])
        if num_topics > 0:
            scenario_metrics["squad_metrics"]["exact_match"] /= num_topics
            scenario_metrics["squad_metrics"]["f1"] /= num_topics
            scenario_metrics["squad_metrics"]["no_answer_probability"] /= num_topics

            if scenario_metrics["neo4j_metrics"]["total_queries"] > 0:
                scenario_metrics["neo4j_metrics"]["success_rate"] = (
                    scenario_metrics["neo4j_metrics"]["successful_queries"]
                    / scenario_metrics["neo4j_metrics"]["total_queries"]
                )
                scenario_metrics["neo4j_metrics"]["results_rate"] = (
                    scenario_metrics["neo4j_metrics"]["queries_with_results"]
                    / scenario_metrics["neo4j_metrics"]["total_queries"]
                )
                scenario_metrics["neo4j_metrics"]["accuracy"] = (
                    scenario_metrics["neo4j_metrics"]["correct_queries"]
                    / scenario_metrics["neo4j_metrics"]["total_queries"]
                )

        run_metrics["scenarios"].append(scenario_metrics)

    # Calculate overall run statistics
    total_scenarios = len(run_metrics["scenarios"])
    if total_scenarios > 0:
        run_metrics["overall_squad_metrics"] = {
            "exact_match": sum(
                s["squad_metrics"]["exact_match"] for s in run_metrics["scenarios"]
            )
            / total_scenarios,
            "f1": sum(s["squad_metrics"]["f1"] for s in run_metrics["scenarios"])
            / total_scenarios,
            "no_answer_probability": sum(
                s["squad_metrics"]["no_answer_probability"]
                for s in run_metrics["scenarios"]
            )
            / total_scenarios,
        }

        total_neo4j_queries = sum(
            s["neo4j_metrics"]["total_queries"] for s in run_metrics["scenarios"]
        )
        if total_neo4j_queries > 0:
            run_metrics["overall_neo4j_metrics"] = {
                "successful_queries": sum(
                    s["neo4j_metrics"]["successful_queries"]
                    for s in run_metrics["scenarios"]
                ),
                "total_queries": total_neo4j_queries,
                "success_rate": sum(
                    s["neo4j_metrics"]["successful_queries"]
                    for s in run_metrics["scenarios"]
                )
                / total_neo4j_queries,
                "queries_with_results": sum(
                    s["neo4j_metrics"]["queries_with_results"]
                    for s in run_metrics["scenarios"]
                ),
                "results_rate": sum(
                    s["neo4j_metrics"]["queries_with_results"]
                    for s in run_metrics["scenarios"]
                )
                / total_neo4j_queries,
                "correct_queries": sum(
                    s["neo4j_metrics"]["correct_queries"]
                    for s in run_metrics["scenarios"]
                ),
                "accuracy": sum(
                    s["neo4j_metrics"]["correct_queries"]
                    for s in run_metrics["scenarios"]
                )
                / total_neo4j_queries,
            }

    # Add runtime statistics
    run_metrics["runtime_stats"] = {
        "total_elapsed_time_seconds": total_runtime_info.total_elapsed_time,
        "total_prompt_tokens": total_runtime_info.prompt_tokens,
        "total_completion_tokens": total_runtime_info.completion_tokens,
        "total_reasoning_tokens": total_runtime_info.reasoning_tokens,
        "total_cached_tokens": total_runtime_info.cached_tokens,
        "total_tokens": total_runtime_info.total_tokens,
        "total_calls": total_runtime_info.total_calls,
        "estimated_cost_usd": total_runtime_info.cost_estimate,
    }

    # Save run metrics
    run_metrics_path = path / "run_metrics.json"
    with open(run_metrics_path, "w") as f:
        json.dump(run_metrics, f, indent=2)
    logger.info(f"Run metrics saved to: {run_metrics_path}")

    # Create and save runtime statistics DataFrame
    import pandas as pd

    runtime_stats_data = [
        {
            "metric": "total_elapsed_time_seconds",
            "value": total_runtime_info.total_elapsed_time,
            "formatted_value": f"{total_runtime_info.total_elapsed_time:.2f}",
        },
        {
            "metric": "total_prompt_tokens",
            "value": total_runtime_info.prompt_tokens,
            "formatted_value": f"{total_runtime_info.prompt_tokens:,}",
        },
        {
            "metric": "total_completion_tokens",
            "value": total_runtime_info.completion_tokens,
            "formatted_value": f"{total_runtime_info.completion_tokens:,}",
        },
        {
            "metric": "total_reasoning_tokens",
            "value": total_runtime_info.reasoning_tokens,
            "formatted_value": f"{total_runtime_info.reasoning_tokens:,}",
        },
        {
            "metric": "total_cached_tokens",
            "value": total_runtime_info.cached_tokens,
            "formatted_value": f"{total_runtime_info.cached_tokens:,}",
        },
        {
            "metric": "total_tokens",
            "value": total_runtime_info.total_tokens,
            "formatted_value": f"{total_runtime_info.total_tokens:,}",
        },
        {
            "metric": "total_calls",
            "value": total_runtime_info.total_calls,
            "formatted_value": str(total_runtime_info.total_calls),
        },
        {
            "metric": "estimated_cost_usd",
            "value": total_runtime_info.cost_estimate,
            "formatted_value": f"${total_runtime_info.cost_estimate:.4f}",
        },
    ]

    df_runtime_stats = pd.DataFrame(runtime_stats_data)
    runtime_stats_path = path / "run_runtime_stats.csv"
    df_runtime_stats.to_csv(runtime_stats_path, index=False)
    logger.info(f"Run runtime statistics saved to: {runtime_stats_path}")

    # Log summary
    logger.info("=" * 80)
    logger.info("OVERALL RUN SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total scenarios: {run_metrics['total_scenarios']}")
    logger.info(f"Total topics: {run_metrics['total_topics']}")
    logger.info(f"Scenarios with ontology: {run_metrics['scenarios_with_ontology']}")
    logger.info(
        f"Scenarios without ontology: {run_metrics['scenarios_without_ontology']}"
    )
    logger.info(f"Neo4j enabled scenarios: {run_metrics['neo4j_enabled_scenarios']}")

    if "overall_squad_metrics" in run_metrics:
        logger.info("Overall SQuAD Metrics:")
        logger.info(
            f"  Exact Match: {run_metrics['overall_squad_metrics']['exact_match']:.4f}"
        )
        logger.info(f"  F1 Score: {run_metrics['overall_squad_metrics']['f1']:.4f}")
        logger.info(
            f"  No Answer Probability: {run_metrics['overall_squad_metrics']['no_answer_probability']:.4f}"
        )

    if "overall_neo4j_metrics" in run_metrics:
        logger.info("Overall Neo4j Metrics:")
        logger.info(
            f"  Success Rate: {run_metrics['overall_neo4j_metrics']['success_rate']:.4f}"
        )
        logger.info(
            f"  Results Rate: {run_metrics['overall_neo4j_metrics']['results_rate']:.4f}"
        )
        logger.info(
            f"  Accuracy: {run_metrics['overall_neo4j_metrics']['accuracy']:.4f}"
        )

    logger.info("Runtime Statistics:")
    logger.info(
        f"  Total elapsed time: {total_runtime_info.total_elapsed_time:.2f} seconds"
    )
    logger.info(f"  Total tokens: {total_runtime_info.total_tokens:,}")
    logger.info(f"  Total calls: {total_runtime_info.total_calls}")
    logger.info(f"  Estimated cost: ${total_runtime_info.cost_estimate:.4f}")

    return run_metrics
