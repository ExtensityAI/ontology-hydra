import json
from pathlib import Path

from loguru import logger
from pyvis.network import Network

COLOR_MAP = {
    "CLASS": "#F1C40F",  # Golden for ontology classes
    "OBJECT_PROPERTY": "#45B7D1",  # Blue
    "DATA_PROPERTY": "#9B59B6",  # Purple
    "Unknown": "#808080",  # Gray fallback
}


def chunked(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def load_ontology(json_file: Path) -> dict:
    with open(json_file, "r") as f:
        ontology = json.load(f)
    logger.info(f"Ontology loaded from {json_file}")
    return ontology


def save_ontology(js: dict, json_file: Path):
    with open(json_file, "w") as f:
        json.dump(js, f, indent=4)
    logger.success(f"Ontology saved to {json_file}")


def save_graph(net: Network, fname: str | Path):
    if isinstance(fname, str):
        net.save_graph(fname)
    else:
        net.save_graph(fname.as_posix())
    logger.success(f"Graph saved to {fname}")


def build_ontology_graph(data: dict) -> Network:
    net = Network(height="100vh", width="100%", bgcolor="#ffffff", font_color="black")
    net.toggle_physics(True)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)

    # Visualize subclass relations as edges between classes
    for relation in data.get("subclass_relations", []):
        subclass = relation["subclass"]
        superclass = relation["superclass"]
        # Both subclass and superclass are ontology classes.
        net.add_node(subclass, label=subclass, title="CLASS", color=COLOR_MAP["CLASS"])
        net.add_node(superclass, label=superclass, title="CLASS", color=COLOR_MAP["CLASS"])
        net.add_edge(subclass, superclass, label="subClassOf", color="#34495E", arrows="to")

    # Visualize object properties: create edges from each domain to each range using the property name.
    for prop in data.get("object_properties", []):
        prop_name = prop["name"]
        domains = prop.get("domain", [])
        ranges = prop.get("range", [])
        for d in domains:
            net.add_node(d, label=d, title="CLASS", color=COLOR_MAP["CLASS"])
        for r in ranges:
            net.add_node(r, label=r, title="CLASS", color=COLOR_MAP["CLASS"])
        for d in domains:
            for r in ranges:
                net.add_edge(
                    d,
                    r,
                    label=prop_name,
                    color=COLOR_MAP["OBJECT_PROPERTY"],
                    arrows="to",
                )

    # Visualize data properties: edges from domain classes to a literal node representing the datatype.
    for prop in data.get("data_properties", []):
        prop_name = prop["name"]
        domains = prop.get("domain", [])
        datatype = prop["range"]["value"]
        dt_node = f"datatype: {datatype}"
        net.add_node(dt_node, label=datatype, title="Datatype", color=COLOR_MAP["DATA_PROPERTY"])
        for d in domains:
            net.add_node(d, label=d, title="CLASS", color=COLOR_MAP["CLASS"])
            net.add_edge(
                d,
                dt_node,
                label=prop_name,
                color=COLOR_MAP["DATA_PROPERTY"],
                arrows="to",
            )

    return net


def build_kg_graph(kg_data: dict) -> Network:
    net = Network(height="100vh", width="100%", bgcolor="#ffffff", font_color="black")
    net.toggle_physics(True)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)

    entity_color = "#3498DB"  # Blue for entities

    for triplet in kg_data.get("triplets", []):
        subject = triplet["subject"]
        predicate = triplet["predicate"]
        obj = triplet["object"]
        confidence = triplet.get("confidence", 1.0)

        net.add_node(subject, label=subject, title="Entity", color=entity_color)
        net.add_node(obj, label=obj, title="Entity", color=entity_color)

        width = 1 + 4 * confidence  # Scale confidence to width between 1-5
        net.add_edge(
            subject,
            obj,
            label=predicate,
            title=f"Confidence: {confidence:.2f}",
            width=width,
            arrows="to",
        )

    return net
