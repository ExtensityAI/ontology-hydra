import json

from ontopipe.models import Characteristic, Ontology


def characteristic_color(characteristics: list[Characteristic]):
    """Assign colors based on property characteristics."""
    if not characteristics:
        return "#7F8C8D"  # default gray
    char_set = {c.value for c in characteristics}
    if "functional" in char_set:
        return "#3498DB"  # blue
    if "transitive" in char_set:
        return "#E67E22"  # orange
    if "symmetric" in char_set:
        return "#2ECC71"  # green
    return "#9B59B6"  # fallback purple


def generate_visjs_code(ontology: Ontology):
    node_set = set()
    datatype_nodes = set()
    edges = []

    # === Subclass Relations ===
    for rel in ontology.subclass_relations:
        subclass = rel.subclass.name
        superclass = rel.superclass.name

        node_set.update([subclass, superclass])
        edges.append(
            {
                "from": subclass,
                "to": superclass,
                "label": "subClassOf",
                "arrows": "to",
                "color": "#34495E",
            }
        )

    # === Object Properties ===
    for prop in ontology.object_properties:
        color = characteristic_color(prop.characteristics)

        for domain in prop.domain:
            for rng in prop.range:
                node_set.update([domain.name, rng.name])
                edges.append(
                    {
                        "from": domain.name,
                        "to": rng.name,
                        "label": prop.name,
                        "arrows": "to",
                        "color": color,
                    }
                )

    # === Node Definitions ===
    nodes = []

    for name in sorted(node_set):
        nodes.append(
            {
                "id": name,
                "label": name,
                "title": "CLASS",
                "shape": "dot",
                "color": "#F1C40F",
                "font": {"color": "black"},
            }
        )

    for dtype in sorted(datatype_nodes):
        nodes.append(
            {
                "id": f"datatype: {dtype}",
                "label": dtype,
                "title": "Datatype",
                "shape": "dot",
                "color": "#9B59B6",
                "font": {"color": "black"},
            }
        )

    # === Final Output ===
    js_code = f"nodes = new vis.DataSet({json.dumps(nodes, indent=2)});\n"
    js_code += f"edges = new vis.DataSet({json.dumps(edges, indent=2)});"

    return js_code
