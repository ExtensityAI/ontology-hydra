import json
from collections import defaultdict, deque
from copy import deepcopy
from pathlib import Path

from loguru import logger
from symai import Expression
from symai.components import MetadataTracker
from symai.strategy import contract

from ontopipe.models import (
    Bridge,
    Cluster,
    DataProperty,
    Merge,
    ObjectProperty,
    Ontology,
    Operation,
    Prune,
    WeaverInput,
)
from ontopipe.prompts import prompt_registry
from ontopipe.utils import load_ontology


# =========================================#
# ----Contract-----------------------------#
# =========================================#
@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=True,
    remedy_retry_params=dict(tries=25, delay=0.5, max_delay=15, jitter=0.1, backoff=2, graceful=False),
)
class Weaver(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dynamic_ontology = None
        self._history = []
        self._cluster_indexes = []

    @property
    def prompt(self) -> str:
        return prompt_registry.instruction("weaver")

    def forward(self, input: WeaverInput, **kwargs) -> Operation:
        if self.contract_result is None:
            raise ValueError("Contract failed!")
        return self.contract_result

    def pre(self, input: WeaverInput) -> bool:
        return True

    def post(self, output: Operation) -> bool:
        op = output.type

        if isinstance(op, Merge):
            for relation in op.relations:
                if not self._class_exists(relation.subclass) or not self._class_exists(relation.superclass):
                    raise ValueError(
                        f"Merge operation contains non-existent class: subclass={relation.subclass}, "
                        f"superclass={relation.superclass}. Please generate a relation that uses existing classes."
                    )
            for idx in op.indexes:
                if idx not in self._cluster_indexes:
                    raise ValueError(f"Invalid cluster index: {idx}. Valid indices are: {self._cluster_indexes}")

        if isinstance(op, Prune):
            for cls in op.classes:
                if not self._class_exists(cls):
                    raise ValueError(
                        f"Prune operation contains non-existent class: {cls}. Please generate a relation that uses existing classes."
                    )
            if op.indexes[0] not in self._cluster_indexes:
                raise ValueError(f"Invalid cluster index: {op.indexes[0]}. Valid indices are: {self._cluster_indexes}")

        if isinstance(op, Bridge):
            for idx in op.indexes:
                if idx not in self._cluster_indexes:
                    raise ValueError(f"Invalid cluster index: {idx}. Valid indices are: {self._cluster_indexes}")

        # Simulate applying the op to see if we decrease the number of clusters
        old_cluster_count = len(self._cluster_indexes)
        current_clusters = Weaver.find_isolated_clusters(self._dynamic_ontology)
        simulated_ontology = Weaver.apply_operation(self._dynamic_ontology, output, current_clusters)
        new_clusters = Weaver.find_isolated_clusters(simulated_ontology)
        new_cluster_count = len(new_clusters)
        logger.debug(f"Old cluster count: {old_cluster_count}, New cluster count: {new_cluster_count}")

        if new_cluster_count >= old_cluster_count:
            raise ValueError(
                f"Operation {op} did not reduce clusters: before={old_cluster_count} after={new_cluster_count}. "
                "Please generate an operation that reduces the number of clusters."
            )

        return True

    def _class_exists(self, cls: str) -> bool:
        if self._dynamic_ontology is None:
            raise ValueError("The dynamic ontology was not set!")
        for relation in self._dynamic_ontology.subclass_relations:
            if cls == relation.subclass or cls == relation.superclass:
                return True
        return False

    def set_cluster_indexes(self, indexes: list[Cluster]):
        self._cluster_indexes = [cluster.index for cluster in indexes]

    def set_dynamic_ontology(self, ontology: Ontology):
        self._dynamic_ontology = ontology

    def update_history(self, operation: Operation):
        self._history.append(operation)

    def clear_history(self):
        self._history.clear()

    def get_history(self):
        return self._history

    @staticmethod
    def dump_transformation_history(
        fname: Path, history: list[Operation], original_ontology: Ontology, original_clusters: list[Cluster]
    ):
        json_data = {
            "original_ontology": original_ontology.model_dump(),
            "original_clusters": [cluster.model_dump() for cluster in original_clusters],
            "history": [(operation.type.__class__.__name__, operation.model_dump()) for operation in history],
        }
        with open(fname, "w") as f:
            json.dump(json_data, f, indent=4)
        logger.success(f"Transformation history saved to {fname}")

    @staticmethod
    def find_isolated_clusters(ontology: Ontology) -> list[Cluster]:
        """
        Identifies isolated clusters of classes in an ontology using subclass relationships.
        The largest connected component is considered the main ontology.
        Returns a list of Cluster objects, each containing an index and the subclass relations
        within that cluster.
        """
        graph = defaultdict(list)
        all_classes = set()

        for rel in ontology.subclass_relations:
            subclass = rel.subclass
            superclass = rel.superclass
            graph[subclass].append(superclass)
            graph[superclass].append(subclass)
            all_classes.add(subclass)
            all_classes.add(superclass)

        if not all_classes:
            return []

        visited = set()
        components = []

        for cls in all_classes:
            if cls not in visited:
                queue = deque([cls])
                visited.add(cls)
                component = {cls}
                while queue:
                    current = queue.popleft()
                    for neighbor in graph.get(current, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                            component.add(neighbor)
                components.append(component)

        if len(components) <= 1:
            return []

        components.sort(key=len, reverse=True)
        clusters = []
        for idx, component_classes in enumerate(components, start=1):
            component_relations = [
                rel
                for rel in ontology.subclass_relations
                if rel.subclass in component_classes and rel.superclass in component_classes
            ]
            clusters.append(Cluster(index=idx, relations=component_relations))
        return clusters

    @staticmethod
    def apply_operation(ontology: Ontology, operation: Operation, clusters: list[Cluster]) -> Ontology:
        new_ontology = deepcopy(ontology)
        op = operation.type

        if isinstance(op, (Merge, Bridge)):
            cluster1 = next(c for c in clusters if c.index == op.indexes[0])
            cluster2 = next(c for c in clusters if c.index == op.indexes[1])
            for relation in op.relations:
                if relation not in new_ontology.subclass_relations:
                    new_ontology.subclass_relations.append(relation)

            valid_classes = set()
            for rel in new_ontology.subclass_relations:
                valid_classes.add(rel.subclass)
                valid_classes.add(rel.superclass)

            filtered_object_properties = []
            for prop in new_ontology.object_properties:
                new_domain = [cls for cls in prop.domain if cls in valid_classes]
                new_range = [cls for cls in prop.range if cls in valid_classes]
                if new_domain and new_range:
                    filtered_object_properties.append(
                        ObjectProperty(
                            name=prop.name,
                            description=prop.description,
                            usage_guideline=prop.usage_guideline,
                            domain=new_domain,
                            range=new_range,
                            characteristics=prop.characteristics,
                        )
                    )
            new_ontology.object_properties = filtered_object_properties

            filtered_data_properties = []
            for prop in new_ontology.data_properties:
                new_domain = [cls for cls in prop.domain if cls in valid_classes]
                if new_domain:
                    filtered_data_properties.append(
                        DataProperty(
                            name=prop.name,
                            description=prop.description,
                            usage_guideline=prop.usage_guideline,
                            domain=new_domain,
                            range=prop.range,
                            characteristics=prop.characteristics,
                        )
                    )
            new_ontology.data_properties = filtered_data_properties

            return new_ontology

        if isinstance(op, Prune):
            cluster = next(c for c in clusters if c.index == op.indexes[0])
            cluster_classes = set()
            for rel in cluster.relations:
                cluster_classes.add(rel.subclass)
                cluster_classes.add(rel.superclass)

            classes_to_prune = {cls for cls in op.classes}
            new_relations = [
                rel
                for rel in new_ontology.subclass_relations
                if rel.subclass not in classes_to_prune and rel.superclass not in classes_to_prune
            ]
            new_ontology.subclass_relations = new_relations

            new_object_properties = []
            for prop in new_ontology.object_properties:
                new_domain = [cls for cls in prop.domain if cls not in classes_to_prune]
                new_range = [cls for cls in prop.range if cls not in classes_to_prune]
                if new_domain and new_range:
                    new_object_properties.append(
                        ObjectProperty(
                            name=prop.name,
                            description=prop.description,
                            usage_guideline=prop.usage_guideline,
                            domain=new_domain,
                            range=new_range,
                            characteristics=prop.characteristics,
                        )
                    )
            new_ontology.object_properties = new_object_properties

            new_data_properties = []
            for prop in new_ontology.data_properties:
                new_domain = [cls for cls in prop.domain if cls not in classes_to_prune]
                if new_domain:
                    new_data_properties.append(
                        DataProperty(
                            name=prop.name,
                            description=prop.description,
                            usage_guideline=prop.usage_guideline,
                            domain=new_domain,
                            range=prop.range,
                            characteristics=prop.characteristics,
                        )
                    )
            new_ontology.data_properties = new_data_properties

            return new_ontology


def fix_ontology(
    ontology: Ontology | Path, folder: Path, fnames: str = "fixed_ontology", dump: bool = True
) -> Ontology:
    if isinstance(ontology, Path):
        ontology = Ontology.model_validate(load_ontology(ontology))

    weaver = Weaver()
    # Init
    clusters = weaver.find_isolated_clusters(ontology)
    weaver.set_dynamic_ontology(ontology)
    weaver.set_cluster_indexes(clusters)
    weaver_input = WeaverInput(ontology=ontology, clusters=clusters, history=None)
    # Preserve original
    original_ontology = deepcopy(ontology)
    original_clusters = deepcopy(clusters)

    with MetadataTracker() as tracker:
        while len(clusters) > 0:
            op = weaver(input=weaver_input)
            updated_ontology = weaver.apply_operation(ontology, op, clusters)
            updated_clusters = weaver.find_isolated_clusters(updated_ontology)

            weaver.update_history(op)
            weaver.set_dynamic_ontology(updated_ontology)
            weaver.set_cluster_indexes(updated_clusters)

            clusters = updated_clusters
            ontology = updated_ontology

            weaver_input = WeaverInput(ontology=ontology, clusters=clusters, history=weaver.get_history())

    logger.info("Usage:")
    print(tracker.usage)
    weaver.contract_perf_stats()

    # Dump stuff
    if dump:
        weaver.dump_transformation_history(
            folder / f"{fnames}_transformation_history.json", weaver.get_history(), original_ontology, original_clusters
        )

    return ontology
