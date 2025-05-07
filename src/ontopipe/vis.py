from pathlib import Path

from pyvis.network import Network

from ontopipe.models import KG, Ontology


def visualize_ontology(ontology: Ontology, output_html_path: Path):
    net = Network(directed=True, height="100vh", width="100%")
    net.toggle_physics(True)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)

    # Basic color assignments
    CLASS_COLOR = "#ADD8E6"  # Light blue for classes
    DATATYPE_COLOR = "#D3D3D3"  # Light gray for datatypes

    # Track which nodes we've already added to the graph
    added_nodes = set()

    # -------------------------------------------------------------------------
    # 1. Add nodes for all classes discovered in the ontology
    #    (from subclass relations, object property domains/ranges, data property domains)
    # -------------------------------------------------------------------------

    # From subclass relations
    for sub_rel in ontology.subclass_relations:
        if sub_rel.subclass.name not in added_nodes:
            net.add_node(sub_rel.subclass.name, label=sub_rel.subclass.name, color=CLASS_COLOR, shape="ellipse")
            added_nodes.add(sub_rel.subclass.name)

        if sub_rel.superclass.name not in added_nodes:
            net.add_node(sub_rel.superclass.name, label=sub_rel.superclass.name, color=CLASS_COLOR, shape="ellipse")
            added_nodes.add(sub_rel.superclass.name)

    # From object properties (domain + range)
    for obj_prop in ontology.object_properties:
        for dom in obj_prop.domain:
            if dom.name not in added_nodes:
                net.add_node(dom.name, label=dom.name, color=CLASS_COLOR, shape="ellipse")
                added_nodes.add(dom.name)
        for rng in obj_prop.range:
            if rng.name not in added_nodes:
                net.add_node(rng.name, label=rng.name, color=CLASS_COLOR, shape="ellipse")
                added_nodes.add(rng.name)

    # From data properties (domain) - we'll add the datatype as a node below
    for data_prop in ontology.data_properties:
        for dom in data_prop.domain:
            if dom.name not in added_nodes:
                net.add_node(dom.name, label=dom.name, color=CLASS_COLOR, shape="ellipse")
                added_nodes.add(dom.name)

    # -------------------------------------------------------------------------
    # 2. Add edges for subclass relationships
    #    (subclass -> superclass) with label "isA"
    # -------------------------------------------------------------------------
    for sub_rel in ontology.subclass_relations:
        net.add_edge(sub_rel.subclass.name, sub_rel.superclass.name, label="isA")

    # -------------------------------------------------------------------------
    # 3. Add edges for object properties
    #    Domain -> Range with label = property name
    # -------------------------------------------------------------------------
    for obj_prop in ontology.object_properties:
        prop_label = obj_prop.name
        for dom in obj_prop.domain:
            for rng in obj_prop.range:
                net.add_edge(dom.name, rng.name, label=prop_label)

    # -------------------------------------------------------------------------
    # 4. Add edges for data properties
    #    Domain -> Datatype node with label = property name
    #    We'll represent datatypes as separate nodes to make them visible.
    # -------------------------------------------------------------------------
    for data_prop in ontology.data_properties:
        prop_label = data_prop.name

        # Ensure we have a node for the datatype
        datatype_node = data_prop.range.value
        if datatype_node not in added_nodes:
            net.add_node(datatype_node, label=datatype_node, color=DATATYPE_COLOR, shape="database")
            added_nodes.add(datatype_node)

        # Create an edge from domain -> datatype with property name as label
        for dom in data_prop.domain:
            net.add_edge(dom.name, datatype_node, label=prop_label)

    # -------------------------------------------------------------------------
    # 5. Generate the HTML and save the interactive network
    # -------------------------------------------------------------------------
    net.save_graph(str(output_html_path))


def visualize_kg(kg: KG, output_html_path: Path):
    net = Network(directed=True, height="100vh", width="100%")
    net.toggle_physics(True)
    net.force_atlas_2based(gravity=-50, central_gravity=0.01, spring_length=100, spring_strength=0.08)

    PALETTE = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#aec7e8",
        "#ffbb78",
        "#98df8a",
        "#ff9896",
        "#c5b0d5",
    ]

    # Default color for entities with no known type
    DEFAULT_COLOR = "#D3D3D3"

    # Step 1. Identify all 'isA' relationships to map entities to types
    #         Also record which nodes themselves are recognized as 'types'
    entity_to_types = {}  # e.g., "Harold" -> {"Person", ...}
    recognized_types = set()  # e.g., {"Person", "City", "Organization", ...}

    for triplet in kg.triplets or []:
        if triplet.predicate.lower() == "isa":
            subj_name = triplet.subject
            obj_name = triplet.object
            recognized_types.add(obj_name)
            entity_to_types.setdefault(subj_name, set()).add(obj_name)

    # Step 2. Assign each recognized type a unique color from the palette
    type_color_map = {}

    def get_type_color(type_name: str) -> str:
        if type_name not in type_color_map:
            # Cycle through the palette if we have more types than colors
            index = len(type_color_map) % len(PALETTE)
            type_color_map[type_name] = PALETTE[index]
        return type_color_map[type_name]

    # Step 3. Collect all entities (subjects + objects)
    all_entities = set()
    for triplet in kg.triplets or []:
        all_entities.add(triplet.subject)
        all_entities.add(triplet.object)

    # Step 4. Add nodes (with colors) to the PyVis network
    added_nodes = set()

    for entity_name in all_entities:
        if entity_name in recognized_types:
            # This entity is itself a 'type' (i.e., appears as the object in 'isA')
            # Give it an outline + fill color
            color_val = get_type_color(entity_name)
            net.add_node(
                entity_name,
                label=entity_name,
                shape="ellipse",
                borderWidth=2,
                borderWidthSelected=3,
                color=color_val,
            )
            added_nodes.add(entity_name)

        elif entity_name in entity_to_types and len(entity_to_types[entity_name]) > 0:
            # This entity has at least one type; pick the first (arbitrary) type's color
            first_type = next(iter(entity_to_types[entity_name]))
            color_val = get_type_color(first_type)
            net.add_node(entity_name, label=entity_name, shape="ellipse", color=color_val)
            added_nodes.add(entity_name)

        else:
            # Entity has no recognized type -> default color
            net.add_node(entity_name, label=entity_name, shape="ellipse", color=DEFAULT_COLOR)
            added_nodes.add(entity_name)

    # Step 5. Add edges for each triplet. The arrow is from subject -> object, labeled by predicate
    for triplet in kg.triplets or []:
        net.add_edge(triplet.subject, triplet.object, label=triplet.predicate)
    # 6. Output the interactive graph to an HTML file
    net.save_graph(str(output_html_path))
