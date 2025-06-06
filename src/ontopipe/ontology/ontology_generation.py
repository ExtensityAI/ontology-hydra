import logging
from pathlib import Path

from symai import Expression
from symai.components import MetadataTracker
from symai.strategy import contract
from tqdm import tqdm

from ontopipe.models import (
    Class,
    DataProperty,
    ObjectProperty,
    Ontology,
    OntologyState,
    OWLBuilderInput,
    SubClassRelation,
)
from ontopipe.prompts import prompt_registry

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
        # @TODO: 3rd party validation of the ontology (something like OOPS!)
        """
        for concept in output.concepts:
                if concept in self._classes:
                    raise ValueError(
                        f"You've generated a duplicate concept: {concept}. It is already defined. Please focus on new and unique concepts while taking the history into account."
                    )"""

        # TODO check if classes exist etc.

        errors = []

        all_superclasses = self._ontology.superclasses

        for concept in output.concepts:
            if isinstance(concept, Class):
                # check if class already exists
                if self._ontology.has_class(concept.name):
                    errors.append(
                        f"Class '{concept.name}' already exists in the ontology. Please ensure unique class names."
                    )

                    continue

            elif isinstance(concept, SubClassRelation):
                # check if subclass and superclass exist
                if not self._ontology.has_class(concept.subclass):
                    errors.append(f"Subclass {concept.subclass} does not exist in the ontology.")
                    continue

                if not self._ontology.has_class(concept.superclass):
                    errors.append(f"Superclass {concept.superclass} does not exist in the ontology.")
                    continue

                # check if prospective subclass is already defined as a subclass
                if (superclass := self._ontology.get_superclass_of(concept.subclass)) is not None:
                    errors.append(
                        f"Subclass '{concept.subclass}' is already defined as a subclass of '{superclass}'. Please ensure that each subclass has only one direct superclass."
                    )

                    continue

                # ensure no circular subclass relations
                su_superclasses = all_superclasses[concept.superclass]
                sc_superclasses = all_superclasses[concept.subclass]

                if any(sc in su_superclasses for sc in sc_superclasses):
                    errors.append(
                        f"Circular subclass relation detected: '{concept.subclass}' cannot be a subclass of '{concept.superclass}' as it is already a superclass of one of its subclasses."
                    )
                    continue

            elif isinstance(concept, ObjectProperty):
                # check if object property already exists
                if self._ontology.has_property(concept.name):
                    errors.append(
                        f"Property '{concept.name}' already exists in the ontology. Please ensure unique object property names."
                    )
                    continue

                # check if domains and ranges are valid classes
                for domain in concept.domain:
                    if not self._ontology.has_class(domain):
                        errors.append(f"Domain class '{domain}' of object property '{concept.name}' does not exist.")
                        continue

                for range in concept.range:
                    if not self._ontology.has_class(range):
                        errors.append(f"Range class '{range}' of object property '{concept.name}' does not exist.")
                        continue

            elif isinstance(concept, DataProperty):
                # check if data property already exists
                if self._ontology.has_property(concept.name):
                    errors.append(
                        f"Property '{concept.name}' already exists in the ontology. Please ensure unique data property names."
                    )
                    continue

                # check if domains are valid classes
                for domain in concept.domain:
                    if not self._ontology.has_class(domain):
                        errors.append(f"Domain class '{domain}' of data property '{concept.name}' does not exist.")
                        continue

        return True

    def dump_ontology(self, folder: Path, fname: str = "ontology.json"):
        """Dumps the current ontology to a JSON file."""
        if not folder.exists():
            folder.mkdir(parents=True, exist_ok=True)

        ontology_path = folder / fname
        with ontology_path.open("w", encoding="utf-8") as f:
            f.write(self._ontology.model_dump_json(indent=2))
        logger.debug("Ontology dumped to %s", ontology_path)

    def extend_concepts(self, concepts: list):
        for concept in concepts:
            if isinstance(concept, Class):
                self._ontology.classes.append(concept)
            elif isinstance(concept, SubClassRelation):
                self._ontology.subclass_relations.append(concept)
            elif isinstance(concept, ObjectProperty):
                self._ontology.object_properties.append(concept)
            elif isinstance(concept, DataProperty):
                self._ontology.data_properties.append(concept)
            else:
                logger.warning(f"Unknown concept type: {type(concept)}. Skipping.")


def generate_ontology(
    cqs: list[str],
    ontology_name: str,
    folder: Path,
    fname: str = "ontology.json",
    batch_size: int = 1,
) -> Ontology:
    builder = OWLBuilder(name=ontology_name)

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
            builder.extend_concepts(concepts)
            state = OntologyState(concepts=concepts)
        builder.contract_perf_stats()
        usage = tracker.usage

    logger.debug("API Usage:\n%s", usage)
    builder.dump_ontology(folder, fname)
    logger.debug("Ontology creation completed!")

    return builder.get_ontology()


if __name__ == "__main__":
    cqs = ["What is the capital of France?", "What is the population of New York City?"]
    ontology_name = "example_ontology"
    folder = Path("/tmp/output")
    generate_ontology(cqs, ontology_name, folder)
