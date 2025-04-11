import json


def characteristic_color(characteristics):
    """Assign colors based on property characteristics."""
    if not characteristics:
        return "#7F8C8D"  # default gray
    char_set = {c["value"] for c in characteristics}
    if "functional" in char_set:
        return "#3498DB"  # blue
    if "transitive" in char_set:
        return "#E67E22"  # orange
    if "symmetric" in char_set:
        return "#2ECC71"  # green
    return "#9B59B6"  # fallback purple


def generate_visjs_code(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    node_set = set()
    datatype_nodes = set()
    edges = []

    # === Subclass Relations ===
    for rel in data.get("subclass_relations", []):
        subclass = rel["subclass"]["name"]
        superclass = rel["superclass"]["name"]

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
    for prop in data.get("object_properties", []):
        prop_name = prop["name"]
        color = characteristic_color(prop.get("characteristics", []))

        for domain in prop["domain"]:
            for rng in prop["range"]:
                domain_name = domain["name"]
                range_name = rng["name"]

                node_set.update([domain_name, range_name])
                edges.append(
                    {
                        "from": domain_name,
                        "to": range_name,
                        "label": prop_name,
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


# Example usage
if __name__ == "__main__":
    input_file = "eval/runs/cmdZlV9O/To_Kill_A_Mockingbird/ontology/ontology.json"  # <-- your JSON file
    visjs_output = generate_visjs_code(input_file)

    with open("output_vis.js", "w") as out_file:
        out_file.write(visjs_output)

    print("vis.js network data written to output_vis.js ✅")
