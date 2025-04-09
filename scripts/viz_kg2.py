import json


def get_confidence_color(confidence):
    if confidence >= 0.9:
        return "#2ECC71"  # green
    elif confidence >= 0.8:
        return "#E67E22"  # orange
    else:
        return "#E74C3C"  # red


def generate_visjs_from_triplets(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    triplets = data.get("triplets", [])
    nodes = {}
    edges = []

    for triplet in triplets:
        subj = triplet["subject"]["name"]
        pred = triplet["predicate"]["name"]
        obj = triplet["object"]["name"]
        confidence = triplet.get("confidence", 0.5)

        # Add subject and object as nodes (if not already added)
        for entity in [subj, obj]:
            if entity not in nodes:
                nodes[entity] = {
                    "id": entity,
                    "label": entity,
                    "title": "ENTITY",
                    "shape": "dot",
                    "color": "#F1C40F",
                    "font": {"color": "black"},
                }

        # Add edge
        edges.append(
            {
                "from": subj,
                "to": obj,
                "label": pred,
                "arrows": "to",
                "color": get_confidence_color(confidence),
            }
        )

    js_code = (
        f"nodes = new vis.DataSet({json.dumps(list(nodes.values()), indent=2)});\n"
    )
    js_code += f"edges = new vis.DataSet({json.dumps(edges, indent=2)});"

    return js_code


# Example usage
if __name__ == "__main__":
    input_path = (
        "eval/runs/nNZq1Fy6/To_Kill_A_Mockingbird/kg/kg.json"  # your JSON filename
    )
    output_path = "mockingbird_vis.js"

    js_output = generate_visjs_from_triplets(input_path)

    with open(output_path, "w") as f:
        f.write(js_output)

    print(f"✅ vis.js code saved to {output_path}")
