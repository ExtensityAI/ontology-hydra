import json
from pathlib import Path

from pyvis.network import Network

from ontopipe.kg.schema import generate_kg_schema
from ontopipe.ontology.models import Ontology

o = Ontology.model_validate_json(
    Path("/home/adrian/Desktop/extensity/ontology-hydra/temp/fiction/output/cache/ontology.partial.json").read_text()
)

schema = generate_kg_schema(o)

print(json.dumps(schema.model_json_schema(), indent=2))

# Create ontology graph visualization
net = Network(height="100vh", width="100vw", bgcolor="#222222", font_color="white")

# Add class nodes - purple circles
for class_name, cls in o.classes.items():
    label = f"{class_name}"
    title = cls.description.description if cls.description and cls.description.description else f"Class: {class_name}"

    net.add_node(
        class_name,
        label=label,
        title=title,
        color="#9C7FCE",
        shape="circle",
        size=20,
        font={"size": 12, "color": "white"},
    )

# Add data type nodes for data properties
data_types_used = set()
for prop in o.data_properties.values():
    data_types_used.add(prop.range)

for data_type in data_types_used:
    net.add_node(
        f"dt_{data_type}",
        label=data_type,
        title=f"Data Type: {data_type}",
        color="#FFD54F",
        shape="diamond",
        size=15,
        font={"size": 10, "color": "black"},
    )

# Add superclass relationships (inheritance edges) - grey dashed
for class_name, cls in o.classes.items():
    if cls.superclass:
        net.add_edge(
            cls.superclass,
            class_name,
            label="inherits",
            color={"color": "#808080", "inherit": False},
            width=2,
            dashes=True,
            arrows={"to": {"enabled": True, "scaleFactor": 0.8}},
            font={"size": 8, "color": "#808080", "strokeWidth": 0, "strokeColor": "transparent"},
        )

# Add data property edges (from domain classes to data type nodes) - light blue
for prop_name, prop in o.data_properties.items():
    edge_title = f"Data Property: {prop_name}"
    if prop.description and prop.description.description:
        edge_title += f"\n{prop.description.description}"

    for domain_class in prop.domain:
        if domain_class in o.classes:
            net.add_edge(
                domain_class,
                f"dt_{prop.range}",
                label=prop_name,
                title=edge_title,
                color={"color": "#87CEEB", "inherit": False},
                width=2,
                arrows={"to": {"enabled": True, "scaleFactor": 0.6}},
                font={"size": 8, "color": "#87CEEB", "strokeWidth": 0, "strokeColor": "transparent"},
            )

# Add object property edges (from domain classes to range classes) - light green
for prop_name, prop in o.object_properties.items():
    edge_title = f"Object Property: {prop_name}"
    if prop.description and prop.description.description:
        edge_title += f"\n{prop.description.description}"

    for domain_class in prop.domain:
        if domain_class in o.classes:
            for range_class in prop.range:
                if range_class in o.classes:
                    net.add_edge(
                        domain_class,
                        range_class,
                        label=prop_name,
                        title=edge_title,
                        color={"color": "#90EE90", "inherit": False},
                        width=2,
                        arrows={"to": {"enabled": True, "scaleFactor": 0.6}},
                        font={"size": 8, "color": "#90EE90", "strokeWidth": 0, "strokeColor": "transparent"},
                    )

# Configure better physics and layout
net.set_options("""
var options = {
  "physics": {
    "enabled": true,
    "forceAtlas2Based": {
      "gravitationalConstant": -80,
      "centralGravity": 0.01,
      "springLength": 250,
      "springConstant": 0.05,
      "damping": 0.4,
      "avoidOverlap": 1
    },
    "maxVelocity": 50,
    "minVelocity": 0.1,
    "solver": "forceAtlas2Based",
    "stabilization": {
      "enabled": true,
      "iterations": 1000,
      "updateInterval": 25
    }
  },
  "layout": {
    "improvedLayout": true,
    "hierarchical": {
      "enabled": false
    }
  },
  "interaction": {
    "dragNodes": true,
    "dragView": true,
    "zoomView": true
  },
  "edges": {
    "smooth": {
      "enabled": true,
      "type": "dynamic"
    }
  }
}
""")

# Save and show the graph
net.show("ontology_graph.html", notebook=False)
print("Graph saved as ontology_graph.html")
