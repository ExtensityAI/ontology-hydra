from tqdm import tqdm
from loguru import logger
from pydantic import Field
import pandas as pd
import json

from symai import Expression, Import, Symbol
from symai.components import FileReader, MetadataTracker
from symai.models import LLMDataModel
from symai.strategy import contract

from chonkie import (RecursiveChunker, SDPMChunker, SemanticChunker,
                     SentenceChunker, TokenChunker)
from chonkie.embeddings.base import BaseEmbeddings
from tokenizers import Tokenizer

CHUNKER_MAPPING = {
    "TokenChunker": TokenChunker,
    "SentenceChunker": SentenceChunker,
    "RecursiveChunker": RecursiveChunker,
    "SemanticChunker": SemanticChunker,
    "SDPMChunker": SDPMChunker,
}

class ChonkieChunker(Expression):
    def __init__(
        self,
        tokenizer_name: str = "gpt2",
        embedding_model_name: str | BaseEmbeddings = "minishlab/potion-base-8M",
        **symai_kwargs,
    ):
        super().__init__(**symai_kwargs)
        self.tokenizer_name = tokenizer_name
        self.embedding_model_name = embedding_model_name

    def forward(self, data: Symbol[str | list[str]], chunker_name: str = "RecursiveChunker", **chunker_kwargs) -> Symbol[list[str]]:
        chunker = self._resolve_chunker(chunker_name, **chunker_kwargs)
        chunks = [self._clean_text(chunk.text) for chunk in chunker(data.value)]
        return self._to_symbol(chunks)

    def _resolve_chunker(self, chunker_name: str, **chunker_kwargs) -> TokenChunker | SentenceChunker | RecursiveChunker | SemanticChunker | SDPMChunker:
        if chunker_name in ["TokenChunker", "SentenceChunker", "RecursiveChunker"]:
            tokenizer = Tokenizer.from_pretrained(self.tokenizer_name)
            return CHUNKER_MAPPING[chunker_name](tokenizer, **chunker_kwargs)
        elif chunker_name in ["SemanticChunker", "SDPMChunker"]:
            return CHUNKER_MAPPING[chunker_name](embedding_model=self.embedding_model_name, **chunker_kwargs)
        else:
            raise ValueError(f"Chunker {chunker_name} not found. Available chunkers: {CHUNKER_MAPPING.keys()}. See docs (https://docs.chonkie.ai/getting-started/introduction) for more info.")

    def _clean_text(self, text: str) -> str:
        """Cleans text by removing problematic characters."""
        text = text.replace('\x00', '')                              # Remove null bytes (\x00)
        text = text.encode('utf-8', errors='ignore').decode('utf-8') # Replace invalid UTF-8 sequences
        return text

# Data Models
class Entity(LLMDataModel):
    """Represents an entity in the ontology"""
    name: str = Field(description="Name of the entity")
    type: str = Field(description="Type/category of the entity")

class Relationship(LLMDataModel):
    """Represents a relationship type in the ontology"""
    name: str = Field(description="Name of the relationship")

class OntologySchema(LLMDataModel):
    """Defines the ontology schema with allowed entities and relationships"""
    entities: list[Entity] = Field(description="List of valid entity types")
    relationships: list[Relationship] = Field(description="List of valid relationship types")

class TripletInput(LLMDataModel):
    """Input for triplet extraction"""
    text: str = Field(description="Text to extract triplets from")
    ontology: OntologySchema = Field(description="Ontology schema to use for extraction")

class Triplet(LLMDataModel):
    """A semantic triplet with typed entities and relationship"""
    subject: Entity
    predicate: Relationship
    object: Entity
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Confidence score for the extracted triplet [0, 1]")

class TripletOutput(LLMDataModel):
    """Collection of extracted triplets forming a knowledge graph"""
    triplets: list[Triplet] | None = Field(default=None, description="List of extracted triplets")

class QAPrediction(LLMDataModel):
    """Represents a prediction for a question-answer pair"""
    id: str = Field(description="Identification field of the question-answer pair")
    prediction_text: str = Field(description="The text of the answer")
    no_answer_probability: float = Field(default=0.0, ge=0.0, le=1.0, description="Probability that the question has no answer")

class QAReference(LLMDataModel):
    """Represents a reference for a question-answer pair"""
    id: str = Field(description="Identification field of the question-answer pair")
    answers: list[dict] = Field(description="List of answer dictionaries with text field")
    no_answer_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Probability threshold to decide a question has no answer")

class QAInput(LLMDataModel):
    """Input for question answering"""
    context: str = Field(description="Context text to extract answers from")
    questions: list[str] = Field(description="List of questions to answer")
    question_ids: list[str] = Field(description="List of question IDs")

class QAOutput(LLMDataModel):
    """Collection of predictions and references for question-answer pairs"""
    predictions: list[QAPrediction] = Field(description="List of predictions for question-answer pairs")
    references: list[QAReference] = Field(default=None, description="List of reference question-answers")

@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=True,
    remedy_retry_params=dict(
        tries=5,
        delay=0.5,
        max_delay=15,
        jitter=0.1,
        backoff=2,
        graceful=False
    )
)
class QuestionAnswerPredictor(Expression):
    """Predicts answers to questions based on context"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = kwargs.get('threshold', 0.5)

    def forward(self, input: QAInput, **kwargs) -> QAOutput:
        if self.contract_result is None:
            return QAOutput(predictions=[], references=None)
        return self.contract_result

    def pre(self, input: QAInput) -> bool:
        # Validate that we have matching number of questions and IDs
        return len(input.questions) == len(input.question_ids)

    def post(self, output: QAOutput) -> bool:
        # Validate predictions
        if not output.predictions:
            return False

        # Check that we have a prediction for each question
        # Check for duplicate and missing question IDs
        question_ids = set([pred.id for pred in output.predictions])
        if len(question_ids) != len(output.predictions):
            raise ValueError("Duplicate question IDs found in predictions")

        # Check if all questions were answered
        # input_ids = set(self.input.question_ids)
        # if len(question_ids) < len(input_ids):
        #     raise ValueError(f"Missing predictions for {len(input_ids) - len(question_ids)} questions")

        # Validate probability values
        for pred in output.predictions:
            if not (0 <= pred.no_answer_probability <= 1):
                raise ValueError(f"Invalid no_answer_probability: {pred.no_answer_probability}")

        return True

    @property
    def prompt(self) -> str:
        return (
            "You are an expert at answering questions based on provided context. "
            "For each question:\n"
            "1. Carefully read the context and question\n"
            "2. Extract the most relevant answer from the context\n"
            "3. If the answer cannot be found in the context, provide an empty string as prediction_text\n"
            "4. Assign a no_answer_probability between 0.0 and 1.0 (0 = certain answer exists, 1 = certain no answer)\n"
            "5. Ensure each question has exactly one prediction\n"
            "6. Keep answers concise and directly from the context\n"
            "JSON format:\n"
            '{"qa_pairs": [{"question": "string", "answer": "string"}]}'
        )

if __name__ == "__main__":
    reader = FileReader()
    chunker = ChonkieChunker()
    predictor = QuestionAnswerPredictor(tokenizer_name="Xenova/gpt-4o")

    DATA_PATH = f"data/squad_2.0_Normans"

    # Control the number of questions to process (set to None to process all)
    MAX_QUESTIONS = None

    # Toggle between using full context or only relevant context per question
    USE_RELEVANT_CONTEXT_ONLY = True  # Set to False to use full context for all questions

    # Title of the dataset to process
    DATASET = "Normans"

    # Load questions and context from JSON file
    with open("/Users/ryang/Work/ExtensityAI/Ontology/Evaluation/data/dev-v2.0.json", "r", encoding="utf-8") as f:
        squad_data = json.load(f)

    # Extract questions and context
    questions = []
    question_ids = []
    references = []
    context_paragraphs = []
    question_to_context = {}  # Map each question ID to its specific context

    for data_item in squad_data["data"]:
        if data_item["title"] == DATASET:
            for paragraph in data_item["paragraphs"]:
                # Collect all context paragraphs
                context_paragraphs.append(paragraph["context"])

                for qa in paragraph["qas"]:
                    questions.append(qa["question"])
                    question_ids.append(qa["id"])

                    # Store the specific context for this question
                    question_to_context[qa["id"]] = paragraph["context"]

                    # Create reference object for evaluation
                    reference = QAReference(
                        id=qa["id"],
                        answers=[{"text": ans["text"], "answer_start": ans["answer_start"]} for ans in qa["answers"]],
                        no_answer_threshold=0.5  # Default threshold
                    )
                    references.append(reference)

    # Combine all context paragraphs into a single text
    full_context_text = " ".join(context_paragraphs)
    logger.info(f"Full context length: {len(full_context_text)} characters")

    # Limit the number of questions if MAX_QUESTIONS is set
    if MAX_QUESTIONS is not None:
        questions = questions[:MAX_QUESTIONS]
        question_ids = question_ids[:MAX_QUESTIONS]
        references = references[:MAX_QUESTIONS]
        logger.info(f"Processing a subset of {MAX_QUESTIONS} questions")

    logger.info(f"Loaded {len(questions)} questions from the {DATASET} dataset")
    logger.info(f"Using {'relevant context only' if USE_RELEVANT_CONTEXT_ONLY else 'full context'} for each question")

    # Process each question individually
    all_predictions = []

    for i, (question, question_id) in enumerate(zip(questions, question_ids)):
        logger.info(f"Processing question {i+1}/{len(questions)}: {question}")

        # Determine which context to use
        if USE_RELEVANT_CONTEXT_ONLY:
            context_text = question_to_context[question_id]
            logger.info(f"Using relevant context of length: {len(context_text)} characters")
        else:
            context_text = full_context_text
            logger.info(f"Using full context of length: {len(full_context_text)} characters")

        # Create input data for a single question
        input_data = QAInput(
            context=context_text,
            questions=[question],
            question_ids=[question_id]
        )

        # Track usage and get prediction for this question
        with MetadataTracker() as tracker:
            try:
                result = predictor(input=input_data)
                if result.predictions:
                    prediction = result.predictions[0]
                    all_predictions.append(prediction)

                    # Display result for this question
                    logger.info(f"\n-------\nQuestion ID: {prediction.id}")
                    logger.info(f"Prediction: {prediction.prediction_text}")
                    logger.info(f"No Answer Probability: {prediction.no_answer_probability}\n-------\n")
                else:
                    logger.warning(f"No prediction returned for question: {question}")
            except Exception as e:
                logger.error(f"Error generating prediction for question {question_id}: {str(e)}")

            usage = tracker.usage
            logger.info(f"API Usage for this question:\n{usage}")

    # Create final result with all predictions
    final_result = QAOutput(
        predictions=all_predictions,
        references=references
    )

    # Display final stats
    predictor.contract_perf_stats()
    logger.info(f"\nProcessed {len(all_predictions)} questions successfully")

    # Save predictions to a file
    output_file = f"{DATA_PATH}/qa_predictions_baseline_context_parts-mini.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "predictions": [pred.dict() for pred in all_predictions],
                "references": [ref.dict() for ref in references]
            },
            f,
            indent=2
        )
    logger.info(f"\nPredictions saved to {output_file}")
