import json
from pathlib import Path

import openai
from loguru import logger
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from eval.squad_v2.data import SquadDataset, SquadQAPair
from eval.squad_v2.squad_v2 import SquadV2
from ontopipe import ontopipe
from ontopipe.kg import generate_kg
from ontopipe.models import KG, Ontology
from ontopipe.vis import visualize_kg, visualize_ontology

KG_BATCH_SIZE = 4
QA_BATCH_SIZE = 4
NO_ANSWER_THRESHOLD = 0.5

squadv2 = SquadV2()


SQUAD_TRAIN_DATASET_PATH = Path("eval/train-v2.0.json")

if not SQUAD_TRAIN_DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Could not find SQuAD training dataset at {SQUAD_TRAIN_DATASET_PATH}. "
        "Please follow the instructions in the README."
    )

SQUAD_TRAIN_DATASET = SquadDataset.model_validate_json(SQUAD_TRAIN_DATASET_PATH.read_text(encoding="utf-8"))


class EvalScenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str

    domain: str
    """The domain used for ontology creation"""

    squad_titles: tuple[str, ...]
    """Titles of topics in the SQuAD dataset to use for evaluation (title field)"""
    # intuition: we create an ontology for the domain, create KGs for each SQuAD topic based on the associated texts and the ontology and then evaluate using SQuAD questions


def _generate_kg(texts: list[str], domain: str, kg_path: Path, ontology: Ontology):
    """Generate a knowledge graph from a list of texts as well as the current ontology and domain"""

    if kg_path.exists():
        # load cached KG
        return KG.model_validate_json(kg_path.read_text(encoding="utf-8"))

    kg = generate_kg(
        texts,
        domain,
        ontology=ontology,
        batch_size=KG_BATCH_SIZE,  # TODO hyperparam
    )

    kg_path.write_text(kg.model_dump_json(indent=2), encoding="utf-8")

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

    details = response.choices[0].message.parsed

    if details is None:
        raise ValueError("Failed to generate QA answers")

    assert len(details.responses) == len(qas), (
        f"Number of responses ({len(details.responses)}) does not match number of questions ({len(qas)})"
    )

    for detail in details.responses:
        # reset relative question id
        detail.qid = qas[int(detail.qid)].id

    return details.responses


def _answer_all_questions(kg: KG, qas: list[SquadQAPair], cache_path: Path):
    """Answer all questions in the SQuAD dataset based on the knowledge graph"""

    details = list[ResponseDetail]()

    if cache_path.exists():
        # load cached responses
        details = ResponseDetails.model_validate_json(cache_path.read_text(encoding="utf-8")).responses
        logger.debug("Loaded {} cached answers from {}", len(details), cache_path)

    logger.debug("Answering {} questions", len(qas))

    # start at the end of the cached responses (should be the same, TODO maybe check that!)
    for i in tqdm(range(len(details), len(qas), QA_BATCH_SIZE), desc="Answering questions"):
        qas_batch = qas[i : i + QA_BATCH_SIZE]

        details.extend(_answer_questions(kg, qas_batch))

        # save intermediate results
        cache_path.write_text(
            ResponseDetails(responses=details).model_dump_json(indent=2),
            encoding="utf-8",
        )

    return details


def _eval_squad_topic(title: str, ontology: Ontology, path: Path):
    """Evaluate QA performance on a specific scenario consisting of multiple SQuAD topics using the generated ontology."""

    logger.info("Generating kg for '{}'", title)

    topic = SQUAD_TRAIN_DATASET.find_topic(title)

    assert topic is not None, f"Topic '{title}' not found in SQuAD training dataset"

    contexts = topic.contexts

    logger.debug(
        "Found {} contexts with {} words for topic '{}'",
        len(contexts),
        sum(len(c.split()) for c in contexts),
        title,
    )

    kg = _generate_kg(contexts, title, path / "kg.json", ontology)

    visualize_kg(kg, path / "kg.html")

    logger.debug("KG generated with {} triplets", len(kg.triplets))

    qas = topic.qas

    logger.debug("Found {} questions for topic '{}'", len(qas), title)

    qa_cache_path = path / "qas.json"

    details = _answer_all_questions(kg, qas, qa_cache_path)

    # TODO IMPORTANT how do we ensure that answers are sourced only from the KG?

    # TODO also save preds and refs to file?

    predictions, references = _format_predictions(details, qas)
    metrics = squadv2.compute(predictions=predictions, references=references)
    logger.info("Evaluation results for topic '{}':", title)
    logger.info("{}", metrics)

    # save results to file
    (path / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.debug("Saved evaluation results to '{}'", path / "metrics.json")


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
    assert len(predictions) == len(references), (
        f"Number of predictions ({len(predictions)}) does not match number of references ({len(references)})"
    )

    for i, (p, r) in enumerate(zip(predictions, references)):
        assert p["id"] == r["id"], (
            f"Formatting predictions failed as prediction ID '{p['id']}' does not match reference ID '{r['id']}' at index {i}."
        )

    return predictions, references


def eval_scenario(scenario: EvalScenario, path: Path):
    """Run evaluation for a single scenario"""

    logger.info(
        "Evaluating ontopipe on scenario '{}'",
        scenario.id,
    )

    logger.info("Generating ontology...")

    ontology = ontopipe(scenario.domain, path)

    visualize_ontology(ontology, path / "ontology.html")

    topics_path = path / "topics"
    topics_path.mkdir(exist_ok=True, parents=True)

    for title in scenario.squad_titles:
        topic_path = topics_path / title
        topic_path.mkdir(exist_ok=True, parents=True)

        _eval_squad_topic(title, ontology, topic_path)
