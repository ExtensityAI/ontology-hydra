import json
from pathlib import Path

from loguru import logger
from symai import Expression
from symai.components import MetadataTracker
from symai.strategy import contract
from tqdm import tqdm

from ontopipe.models import (
    DataProperty,
    ObjectProperty,
    Ontology,
    OntologyState,
    OWLBuilderInput,
    OwlClass,
    SubClassRelation,
)
from ontopipe.prompts import prompt_registry


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
    def __init__(self, name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.name = name
        self._classes = set()
        self._subclass_relations = set()
        self._object_properties = set()
        self._data_properties = set()

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
        # @TODO: 3rd party validation of the ontology (something like OOPS!)
        for concept in output.concepts:
            if isinstance(concept, OwlClass):
                if concept in self._classes:
                    raise ValueError(
                        f"You've generated a duplicate concept: {concept}. It is already defined. Please focus on new and unique concepts while taking the history into account."
                    )
        return True

    def extend_concepts(self, concepts: list):
        for concept in concepts:
            if isinstance(concept, SubClassRelation):
                self._classes.add(concept.subclass)
                self._classes.add(concept.superclass)
                self._subclass_relations.add(concept)
            elif isinstance(concept, ObjectProperty):
                for domain in concept.domain:
                    self._classes.add(domain)
                for range in concept.range:
                    self._classes.add(range)
                self._object_properties.add(concept)
            elif isinstance(concept, DataProperty):
                for domain in concept.domain:
                    self._classes.add(domain)
                self._data_properties.add(concept)

    def get_ontology(self) -> Ontology:
        return Ontology(
            name=self.name,
            subclass_relations=self._subclass_relations,
            object_properties=self._object_properties,
            data_properties=self._data_properties,
        )

    def dump_ontology(self, folder: Path, fname: str = "ontology.json"):
        ontology = self.get_ontology()
        if not folder.exists():
            folder.mkdir(parents=True)
        with open(folder / fname, "w") as f:
            json.dump(ontology.model_dump(), f, indent=4)

    def to_rdf(self, folder: Path, fname: str = "ontology.rdf"):
        raise NotImplementedError("to_rdf method not implemented")


def generate_ontology(
    cqs: list[str],
    ontology_name: str,
    folder: Path,
    fname: str = "ontology.json",
    batch_size: int = 1,
) -> Ontology:
    builder = OWLBuilder(name=ontology_name)

    usage = None
    state = OntologyState(concepts=None)
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
            builder.extend_concepts(concepts)
            state = OntologyState(concepts=concepts)
        builder.contract_perf_stats()
        usage = tracker.usage

    logger.debug(f"API Usage:\n{usage}")
    builder.dump_ontology(folder, fname)
    logger.debug("Ontology creation completed!")

    return builder.get_ontology()


if __name__ == "__main__":
    cqs = ["What is the capital of France?", "What is the population of New York City?"]
    ontology_name = "example_ontology"
    folder = Path("/tmp/output")
    generate_ontology(cqs, ontology_name, folder)
