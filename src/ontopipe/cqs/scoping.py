from logging import getLogger

from pydantic import Field
from symai import Expression
from symai.components import MetadataTracker
from symai.strategy import LLMDataModel, contract

from ontopipe.cqs.personas import Persona
from ontopipe.prompts import prompt_registry

logger = getLogger("ontopipe.cqs")


class ScopeDocumentGenerationInput(LLMDataModel):
    domain: str = Field(..., description="The domain of the ontology")
    personas: list[Persona] = Field(..., description="The personas for which to generate a scope document")


class ScopeDocument(LLMDataModel):
    content: str = Field(..., description="The generated scope document content")


class ScopeDocumentMergeInput(LLMDataModel):
    domain: str = Field(..., description="The domain of the ontology")
    documents: list[str] = Field(..., description="The scope documents to merge")


@contract(
    pre_remedy=False,
    post_remedy=True,
    accumulate_errors=False,
    verbose=True,
    remedy_retry_params=dict(tries=25, delay=0.5, max_delay=15, jitter=0.1, backoff=2, graceful=False),
)
class ScopeDocumentGenerator(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: ScopeDocumentGenerationInput, **kwargs) -> ScopeDocument:
        if self.contract_result is None:
            raise ValueError("Contract failed!")
        return self.contract_result

    def post(self, output: ScopeDocument) -> bool:
        if not output.content or len(output.content) < 100:
            return False
        return True

    @property
    def prompt(self) -> str:
        return prompt_registry.instruction("generate_scope_document")


@contract(
    pre_remedy=False,
    post_remedy=True,
    accumulate_errors=False,
    verbose=True,
    remedy_retry_params=dict(tries=25, delay=0.5, max_delay=15, jitter=0.1, backoff=2, graceful=False),
)
class ScopeDocumentMerger(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: ScopeDocumentMergeInput, **kwargs) -> ScopeDocument:
        if self.contract_result is None:
            raise ValueError("Contract failed!")
        return self.contract_result

    def post(self, output: ScopeDocument) -> bool:
        if not output.content or len(output.content) < 100:
            return False
        return True

    @property
    def prompt(self) -> str:
        return prompt_registry.instruction("merge_scope_documents")


def generate_scope_document(domain: str, personas: list[Persona]) -> str:
    """Generate a scope document for a given domain and group of personas."""
    generator = ScopeDocumentGenerator()

    with MetadataTracker() as tracker:
        result = generator(input=ScopeDocumentGenerationInput(domain=domain, personas=personas))

        generator.contract_perf_stats()
        logger.debug("API Usage: %s", tracker.usage)

    return result.content


CHUNK_SIZE = 6


def merge_scope_documents(domain: str, documents: list[str], chunk_size=CHUNK_SIZE) -> str:
    """Merge scope documents into a single document by splitting them into chunks and merging them recursively."""

    if len(documents) == 1:
        return documents[0]

    if len(documents) <= chunk_size:
        return _do_merge(domain, documents)

    # Split into chunks of size chunk_size
    chunks = [documents[i : i + chunk_size] for i in range(0, len(documents), chunk_size)]
    merged_chunks = [_do_merge(domain, chunk) for chunk in chunks]

    return merge_scope_documents(domain, merged_chunks)


def _do_merge(domain: str, documents: list[str]) -> str:
    """Merge a small set of documents using the ScopeDocumentMerger contract."""
    merger = ScopeDocumentMerger()

    with MetadataTracker() as tracker:
        result = merger(input=ScopeDocumentMergeInput(domain=domain, documents=documents))

        merger.contract_perf_stats()
        logger.debug("API Usage: %s", tracker.usage)

    return result.content
