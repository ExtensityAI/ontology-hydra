from logging import getLogger
from typing import List

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


class Questions(LLMDataModel):
    items: List[str] = Field(..., description="List of generated questions")


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

    def forward(self, input: QuestionGenerationInput, **kwargs) -> Questions:
        if self.contract_result is None:
            raise ValueError("Contract failed!")
        return self.contract_result

    def post(self, output: Questions) -> bool:
        # Ensure we have at least one question (TODO in the future improve this massively, and maybe skip the scoping step as well!)
        if not output.items or len(output.items) == 0:
            return False
        return True

    @property
    def prompt(self) -> str:
        return prompt_registry.instruction("generate_questions")


def generate_questions(domain: str, group: list[ComitteeMember], scope_document: str) -> List[str]:
    generator = QuestionGenerator()

    with MetadataTracker() as tracker:
        result = generator(input=QuestionGenerationInput(domain=domain, group=group, scope_document=scope_document))

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

    def forward(self, input: Questions, **kwargs) -> Questions:
        if self.contract_result is None:
            raise ValueError("Contract failed!")
        return self.contract_result

    def post(self, output: Questions) -> bool:
        # Ensure we have at least one question after deduplication
        if not output.items or len(output.items) == 0:
            return False
        return True

    @property
    def prompt(self) -> str:
        return prompt_registry.instruction("deduplicate_questions")
