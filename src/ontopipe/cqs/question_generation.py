from logging import getLogger

from pydantic import Field
from symai import Expression
from symai.components import MetadataTracker
from symai.strategy import LLMDataModel, contract

from ontopipe.cqs.comittee import ComitteeMember
from ontopipe.prompts import prompt_registry

logger = getLogger("ontopipe.cqs")


class QuestionGenerationInput(LLMDataModel):
    domain: str = Field(..., description="The domain of the ontology")
    group: list[ComitteeMember] = Field(..., description="The committee members generating questions")
    scope_document: str = Field(..., description="The scope document containing domain information")


class QuestionGeneratorOutput(LLMDataModel):
    items: list[str] = Field(..., description="List of generated questions")


class Question(LLMDataModel):
    index: int
    text: str


class Questions(LLMDataModel):
    items: list[Question] = Field(..., description="List of questions")


class Duplicate(LLMDataModel):
    question: str = Field(..., description="The question text that is duplicated")
    indexes: list[int] = Field(
        ..., description="List of indexes of duplicates (including the original! thus, length is at least 2)"
    )


class Duplicates(LLMDataModel):
    duplicates: list[Duplicate] = Field(..., description="List of duplicates found in the questions")


@contract(
    pre_remedy=False,
    post_remedy=True,
    accumulate_errors=False,
    verbose=True,
    remedy_retry_params=dict(tries=25, delay=0.5, max_delay=15, jitter=0.1, backoff=2, graceful=False),
)
class QuestionGenerator(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: QuestionGenerationInput, **kwargs) -> QuestionGeneratorOutput:
        if self.contract_result is None:
            raise ValueError("Contract failed!")
        return self.contract_result

    def post(self, output: QuestionGeneratorOutput) -> bool:
        # Ensure we have at least one question (TODO in the future improve this massively, and maybe skip the scoping step as well!)
        if not output.items or len(output.items) == 0:
            return False
        return True

    @property
    def prompt(self) -> str:
        return prompt_registry.instruction("generate_questions")


def generate_questions(domain: str, group: list[ComitteeMember], scope_document: str) -> list[str]:
    generator = QuestionGenerator()

    with MetadataTracker() as tracker:
        result: QuestionGeneratorOutput = generator(
            input=QuestionGenerationInput(domain=domain, group=group, scope_document=scope_document)
        )

        generator.contract_perf_stats()
        logger.debug("API Usage: %s", tracker.usage)

    return result.items


@contract(
    pre_remedy=False,
    post_remedy=True,
    accumulate_errors=False,
    verbose=True,
    remedy_retry_params=dict(tries=25, delay=0.5, max_delay=15, jitter=0.1, backoff=2, graceful=False),
)
class QuestionDeduplicator(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: Questions, **kwargs) -> Duplicates:
        if self.contract_result is None:
            raise ValueError("Contract failed!")
        return self.contract_result

    def post(self, output: Duplicates) -> bool:
        return True

    @property
    def prompt(self) -> str:
        return prompt_registry.instruction("deduplicate_questions")
