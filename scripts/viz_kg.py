import networkx as nx
from pyvis.network import Network
import re
import yaml
import json

def create_knowledge_graph(text, fname):
    # Create a new network
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    net.toggle_physics(True)

    # Set some physics options for better visualization
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)

    # Parse the YAML-like text and extract triplets
    triplets = []

    # Split text into blocks separated by "-------"
    blocks = text.split("-------")

    for block in blocks:
        if "subject:" in block and "predicate:" in block and "object:" in block:
            try:
                # Parse YAML block
                triplet_data = yaml.safe_load(block)

                subject = triplet_data['subject']['name']
                predicate = triplet_data['predicate']['name']
                obj = triplet_data['object']['name']
                # Assign default types if missing
                subj_type = triplet_data['subject'].get('type', 'Unknown')
                obj_type = triplet_data['object'].get('type', 'Unknown')
                confidence = triplet_data.get('confidence', 0.5)

                triplets.append({
                    'subject': (subject, subj_type),
                    'predicate': predicate,
                    'object': (obj, obj_type),
                    'confidence': confidence
                })
            except Exception as e:
                continue

    # Add nodes and edges to the network
    add_triplets_to_network(net, triplets)

    # Save the network
    net.save_graph(fname)
    return fname

def create_knowledge_graph_from_json(json_text, fname):
    # Create a new network
    net = Network(height="750px", width="100%", bgcolor="#ffffff", font_color="black")
    net.toggle_physics(True)

    # Set physics options for better visualization
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)

    data = json.loads(json_text)
    triplets = []

    for key, triple in data.items():
        try:
            # Extract triple information. If no 'type' provided, default to "CONTENT"
            subject = triple['subject']['name']
            predicate = triple['predicate']['name']
            obj = triple['object']['name']
            subj_type = triple['subject'].get('type', 'CONTENT')  # Default value changed here
            obj_type = triple['object'].get('type', 'CONTENT')      # Default value changed here
            confidence = triple.get('confidence', 0.5)

            triplets.append({
                'subject': (subject, subj_type),
                'predicate': predicate,
                'object': (obj, obj_type),
                'confidence': confidence
            })
        except Exception as e:
            # In case any triple can't be parsed, skip it
            continue

    # Add nodes and edges to the network
    add_triplets_to_network(net, triplets)

    # Save the network
    net.save_graph(fname)
    return fname

def add_triplets_to_network(net, triplets):
    # Define a color map for different node types
    color_map = {
        'ORG': '#FF6B6B',      # Red
        'PERSON': '#4ECDC4',    # Teal
        'SERVICE': '#45B7D1',   # Blue
        'AGREEMENT': '#96CEB4',  # Green
        'POLICY': '#FFEEAD',    # Yellow
        'RIGHT': '#D4A5A5',     # Pink
        'OBLIGATION': '#9B59B6', # Purple
        'CONTENT': '#F1C40F',   # Golden (default for JSON input)
        'LOC': '#E67E22',       # Orange
        'Unknown': '#808080'    # Gray fallback for unknown types (used for YAML if missing)
    }

    for triplet in triplets:
        subj, subj_type = triplet['subject']
        obj, obj_type = triplet['object']
        pred = triplet['predicate']
        conf = triplet['confidence']

        # Add nodes with their respective colors based on type
        net.add_node(subj, label=subj, title=subj_type, color=color_map.get(subj_type, '#808080'))
        net.add_node(obj, label=obj, title=obj_type, color=color_map.get(obj_type, '#808080'))

        # Add edges with width scaled from confidence
        width = conf * 3
        net.add_edge(subj, obj, label=pred, width=width)

# Example usage for JSON input:
with open('artifacts/kg.json', 'r') as f:
    json_text = f.read()
create_knowledge_graph_from_json(json_text, fname="kg.html")
