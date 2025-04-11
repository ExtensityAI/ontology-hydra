from pydantic import BaseModel, Field
from symai import Expression
from symai.components import MetadataTracker
from symai.models import LLMDataModel
from symai.strategy import contract

from ontopipe.models import KG


class Question(BaseModel):
    id: str
    text: str
    answers: list[str]


class QuestionInput(BaseModel):
    id: str
    text: str


class QAInput(LLMDataModel):
    """Input for question answering"""

    kg: KG = Field(description="Knowledge graph to extract answers from")
    questions: list[QuestionInput] = Field(description="List of questions to answer")


class QAPrediction(LLMDataModel):
    """Represents a prediction for a question-answer pair"""

    id: str = Field(description="ID of the question")

    cant_answer: bool = Field(
        description="True if the model cannot answer the question based on the KG. THIS DOES NOT MEAN THAT THE QUESTION HAS NO ANSWER."
    )

    prediction_text: str | None = Field(description="The text of the answer")
    no_answer_probability: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Probability that the question has no answer",
    )


class QAOutput(LLMDataModel):
    """Collection of predictions and references for question-answer pairs"""

    predictions: list[QAPrediction] = Field(description="List of predictions for question-answer pairs")


@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=True,
    remedy_retry_params=dict(tries=5, delay=0.5, max_delay=15, jitter=0.1, backoff=2, graceful=False),
)
class QuestionAnswerPredictor(Expression):
    """Predicts answers to questions based on context"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = kwargs.get("threshold", 0.5)

    def forward(self, input: QAInput, **kwargs) -> QAOutput:
        if self.contract_result is None:
            raise ValueError("Contract failed. Cannot proceed with predictions.")
        return self.contract_result

    def pre(self, input: QAInput) -> bool:
        # Validate that we have matching number of questions and IDs
        return len(input.questions) == len(input.questions)

    def post(self, output: QAOutput) -> bool:
        # Validate predictions
        if not output.predictions:
            return False

        # Check that we have a prediction for each question
        # Check for duplicate and missing question IDs
        question_ids = set([pred.id for pred in output.predictions])
        if len(question_ids) != len(output.predictions):
            raise ValueError("Duplicate question IDs found in predictions")

        ## Check if all questions were answered
        # input_ids = set(self.input.question_ids)
        # if len(input_ids.difference(set(pred.id for pred in output.predictions))) > 0:
        #    raise ValueError("Missing predictions! Not all questions have been answered")

        # Validate probability values
        for pred in output.predictions:
            if not (0 <= pred.no_answer_probability <= 1):
                raise ValueError(f"Invalid no_answer_probability: {pred.no_answer_probability}")

        return True

    @property
    def prompt(self) -> str:
        return (
            "Answer questions based on the provided knowledge graph. For each question:\n"
            "1. Extract the precise answer from the context\n"
            "2. If no answer exists in the context, set prediction_text to null and cant_answer to true\n"
            "3. Assign no_answer_probability (0=certain answer exists, 1=certain no answer)\n"
            "4. Keep answers concise and ONLY USE THE INFORMATION IN THE KNOWLEDGE GRAPH to answer.\n"
            "5. Provide exactly one prediction per question ID"
        )


def predict_squad(kg: KG, questions: list[Question], batch_size: int = 10):
    predictor = QuestionAnswerPredictor()
    with MetadataTracker() as tracker:
        for i in range(0, len(questions), batch_size):
            question_batch = questions[i : i + batch_size]
            input_data = QAInput(kg=kg, questions=[QuestionInput(id=q.id, text=q.text) for q in question_batch])

            yield from predictor(input=input_data).predictions
