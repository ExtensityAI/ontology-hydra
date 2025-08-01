import logging
from pathlib import Path

from pydantic import Field
from symai import Expression
from symai.components import MetadataTracker
from symai.strategy import LLMDataModel, contract
from tqdm import tqdm

from ontopipe.ontology.models import Concept, Ontology
from ontopipe.ontology.validator import try_add_concepts
from ontopipe.prompts import prompt_registry

# from ontopipe.vis import visualize_ontology

logger = logging.getLogger("ontopipe.ontology.generator")


class OntologyGeneratorInput(LLMDataModel):
    cqs: list[str] = Field(
        description="A list of competency questions discovered during an interview process by the ontology engineer. Extract a list of relevant concepts."  # TODO mention that the extracted concepts should be GENERAL but also useful to answer the questions.
    )
    ontology: Ontology = Field(
        description="A dynamic state of the ontology that evolves with each iteration. Use this state to expand the ontology with new concepts."
    )


class OntologyGeneratorOutput(LLMDataModel):
    additions: list[Concept] = Field(description="List of new concepts that should be added to the ontology.")


# TODO: make sure that properties are not applied to a class and to its superclass (could be autofixed, but rather not have the model generate that. Also, maybe allow moving properties around or EXTENDING properties?)


# =========================================#
# ----Contract-----------------------------#
# =========================================#
@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=True,
    remedy_retry_params=dict(tries=25, delay=0.5, max_delay=15, jitter=0.1, backoff=2, graceful=False),
)
class OntologyGenerator(Expression):
    def __init__(self, ontology: Ontology, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ontology = ontology

    @property
    def prompt(self) -> str:
        return prompt_registry.instruction("ontology_generator")

    def forward(self, input: OntologyGeneratorInput, **kwargs) -> OntologyGeneratorOutput:
        if self.contract_result is None:
            raise ValueError("Contract failed!")
        return self.contract_result

    def pre(self, input: OntologyGeneratorInput):
        return True

    def post(self, output: OntologyGeneratorOutput):
        is_valid, issues = try_add_concepts(self._ontology, output.additions)

        if not is_valid:
            raise ValueError(
                "Ontology validation failed with the following errors:\n- " + "\n- ".join(map(str, issues))
            )

        return True


def generate_ontology(
    cqs: list[str],
    cache_path: Path,
    cqs_per_batch: int = 1,
) -> Ontology:
    # TODO consider providing scope document
    # TODO do this iteratively, i.e. generate until done. Then, critique ontology and regenerate from there.

    ontology = Ontology()
    generator = OntologyGenerator(ontology)

    partial_json_cache_path = cache_path.with_suffix(".partial.json")
    partial_html_cache_path = cache_path.with_suffix(".partial.html")

    usage = None
    with MetadataTracker() as tracker:  # For gpt-* models
        for i in tqdm(range(0, len(cqs), cqs_per_batch)):
            batch_cqs = cqs[i : i + cqs_per_batch]

            input = OntologyGeneratorInput(cqs=batch_cqs, ontology=ontology)

            try:
                output: OntologyGeneratorOutput = generator(input=input)
                # TODO do not like that the ontology is just implicitly mutated here, change this in the future again
            except Exception as e:
                logger.error(f"Error getting state update for batch: {e}")
                continue

            partial_json_cache_path.write_text(
                ontology.model_dump_json(indent=2),
                encoding="utf-8",
            )

            # visualize_ontology(ontology, partial_html_cache_path, open_browser=False)

        generator.contract_perf_stats()
        usage = tracker.usage

    logger.debug("API Usage:\n%s", usage)

    cache_path.write_text(
        ontology.model_dump_json(indent=2),
        encoding="utf-8",
    )

    # remove partial cache files once again
    partial_json_cache_path.unlink(missing_ok=True)
    partial_html_cache_path.unlink(missing_ok=True)

    logger.debug("Ontology creation completed!")

    return ontology
