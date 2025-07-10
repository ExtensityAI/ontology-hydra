import logging
from pathlib import Path

from symai import Expression
from symai.components import MetadataTracker
from symai.strategy import contract
from tqdm import tqdm

from ontopipe.models import (
    Ontology,
    OntologyState,
    OWLBuilderInput,
)
from ontopipe.ontology.ontology_validation import try_add_concepts
from ontopipe.prompts import prompt_registry
from ontopipe.vis import visualize_ontology

logger = logging.getLogger("ontopipe.ontology_generation")


# =========================================#
# ----Contract-----------------------------#
# =========================================#
@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=True,
    remedy_retry_params=dict(tries=25, delay=0.5, max_delay=15, jitter=0.1, backoff=2, graceful=False),
)
class OWLBuilder(Expression):
    def __init__(self, ontology: Ontology, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ontology = ontology

    @property
    def prompt(self) -> str:
        return prompt_registry.instruction("owl_builder")

    def forward(self, input: OWLBuilderInput, **kwargs) -> OntologyState:
        if self.contract_result is None:
            raise ValueError("Contract failed!")
        return self.contract_result

    def pre(self, input: OWLBuilderInput) -> bool:
        return True

    def post(self, output: OntologyState) -> bool:
        is_valid, issues, _ = try_add_concepts(self._ontology, output.concepts)

        if not is_valid:
            raise ValueError(
                "Ontology validation failed with the following errors:\n- " + "\n- ".join(map(str, issues))
            )

        return True


def generate_ontology(
    cqs: list[str],
    ontology_name: str,
    cache_path: Path,
    batch_size: int = 1,
) -> Ontology:
    ontology = Ontology(
        name=ontology_name,
        classes=[],
        subclass_relations=[],
        object_properties=[],
        data_properties=[],
    )
    # TODO consider providing scope document
    # TODO do this iteratively, i.e. generate until done. Then, critique ontology and regenerate from there.

    builder = OWLBuilder(ontology)

    usage = None
    state = OntologyState(concepts=[])
    concepts = []
    with MetadataTracker() as tracker:  # For gpt-* models
        for i in tqdm(range(0, len(cqs), batch_size)):
            batch_cqs = cqs[i : i + batch_size]

            input_data = OWLBuilderInput(competency_question=batch_cqs, ontology_state=state)

            try:
                new_state = builder(input=input_data)
            except Exception as e:
                logger.error(f"Error getting state update for batch: {e}")
                continue

            concepts.extend(new_state.concepts)
            ontology.extend(new_state.concepts)

            cache_path.with_suffix(".partial.json").write_text(
                ontology.model_dump_json(indent=2),
                encoding="utf-8",
            )

            visualize_ontology(ontology, cache_path.with_suffix(".partial.html"), open_browser=False)

            state = OntologyState(concepts=concepts)
        builder.contract_perf_stats()
        usage = tracker.usage

    logger.debug("API Usage:\n%s", usage)
    cache_path.write_text(
        ontology.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.debug("Ontology creation completed!")

    return ontology
