import datetime
import json
import webbrowser
from pathlib import Path
from typing import Any

from ontopipe.models import KG
from ontopipe.ontology.models import Ontology

# Import models from your existing code


# Global backward compatibility functions to match the original API
def visualize_ontology(ontology: Ontology, output_html_path: Path, open_browser: bool = True):
    """
    Backwards compatible function that matches the original API.
    Creates an interactive visualization of an ontology.

    Parameters:
    -----------
    ontology : Ontology
        The ontology to visualize
    output_html_path : Path
        Path to save the HTML visualization
    """
    viz = KnowledgeGraphViz(output_dir=str(output_html_path.parent), auto_open=open_browser)
    return viz.visualize_ontology(ontology, filename=output_html_path.name)


def visualize_kg(
    kg: KG,
    output_html_path: Path,
    ontology: Ontology | None = None,
    open_browser: bool = True,
):
    """
    Backwards compatible function that matches the original API.
    Creates an interactive visualization of a knowledge graph.

    Parameters:
    -----------
    kg : KG
        The knowledge graph to visualize
    output_html_path : Path
        Path to save the HTML visualization
    ontology : Ontology | None
        Optional ontology for enhanced visualization
    """
    viz = KnowledgeGraphViz(output_dir=str(output_html_path.parent), auto_open=open_browser)
    return viz.visualize_kg(kg, filename=output_html_path.name, ontology=ontology)


class AdvancedGraphVisualizer:
    """
    Advanced graph visualization library for knowledge graphs and ontologies
    with interactive, GPU-accelerated visualizations.
    """

    def __init__(
        self,
        dark_mode: bool = True,
        physics_enabled: bool = True,
        render_mode: str = "webgl",  # Ensure WebGL is used for GPU acceleration
        height: str = "100vh",
        width: str = "100%",
        auto_open: bool = True,
        layout_algorithm: str = "forceAtlas2Based",
        enable_history: bool = True,
        enable_minimap: bool = True,
        enable_advanced_search: bool = True,
        initial_stabilization: bool = True,  # NEW: Allow disabling initial stabilization
        max_nodes_full_render: int = 500,  # NEW: Performance threshold
    ):
        """
        Initialize the visualizer with customizable options.

        Parameters:
        -----------
        dark_mode : bool
            Whether to use dark mode theme (default: True)
        physics_enabled : bool
            Whether to enable physics simulation (default: True)
        render_mode : str
            Rendering mode ("webgl" for GPU acceleration, "canvas" for compatibility)
        height : str
            Height of the visualization container
        width : str
            Width of the visualization container
        auto_open : bool
            Whether to automatically open the visualization in a browser
        initial_stabilization : bool
            Whether to run stabilization on startup (disable for large graphs)
        max_nodes_full_render : int
            Maximum number of nodes to render with full details
        """
        self.dark_mode = dark_mode
        self.physics_enabled = physics_enabled
        self.render_mode = render_mode
        self.height = height
        self.width = width
        self.auto_open = auto_open
        self.initial_stabilization = initial_stabilization
        self.max_nodes_full_render = max_nodes_full_render

        # Palette for entity types (high contrast for dark mode)
        self.color_palette = [
            "#00BFFF",  # Deep Sky Blue
            "#FF4500",  # Orange Red
            "#32CD32",  # Lime Green
            "#FFD700",  # Gold
            "#FF1493",  # Deep Pink
            "#9370DB",  # Medium Purple
            "#00CED1",  # Dark Turquoise
            "#FF8C00",  # Dark Orange
            "#00FA9A",  # Medium Spring Green
            "#FF00FF",  # Magenta
            "#1E90FF",  # Dodger Blue
            "#DC143C",  # Crimson
            "#7FFF00",  # Chartreuse
            "#FF69B4",  # Hot Pink
            "#4169E1",  # Royal Blue
        ]

        # Alternate palette for light mode
        self.light_palette = [
            "#1f77b4",  # Blue
            "#ff7f0e",  # Orange
            "#2ca02c",  # Green
            "#d62728",  # Red
            "#9467bd",  # Purple
            "#8c564b",  # Brown
            "#e377c2",  # Pink
            "#7f7f7f",  # Gray
            "#bcbd22",  # Yellow-green
            "#17becf",  # Teal
        ]

        # Node shapes for different entity types - SMALLER shapes by default
        self.shapes = {
            "class": "circle",
            "datatype": "diamond",
            "property": "triangleDown",
            "individual": "dot",
            "default": "dot",
        }

        # Store additional parameters
        self.layout_algorithm = layout_algorithm
        self.enable_history = enable_history
        self.enable_minimap = enable_minimap
        self.enable_advanced_search = enable_advanced_search

        # Default settings with optimized physics
        self.default_settings = {
            "physics": {
                "enabled": self.physics_enabled,
                "barnesHut": {  # More optimized for large graphs
                    "gravitationalConstant": -2000,  # Reduced from -30000
                    "centralGravity": 0.1,  # Reduced from 0.3
                    "springLength": 150,  # Increased from 95
                    "springConstant": 0.02,  # Reduced from 0.04
                    "damping": 0.09,
                    "avoidOverlap": 0.2,  # Increased from 0.1
                },
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springConstant": 0.04,  # Reduced from 0.08
                    "springLength": 150,  # Increased from 100
                    "damping": 0.3,  # Reduced from 0.4
                    "avoidOverlap": 0.5,
                },
                "hierarchicalRepulsion": {
                    "nodeDistance": 150,  # Increased from 120
                    "centralGravity": 0.0,
                    "springLength": 150,  # Increased from 100
                    "springConstant": 0.01,
                    "damping": 0.09,
                },
                "maxVelocity": 30,  # Reduced from 50
                "minVelocity": 0.75,  # Increased from 0.1
                "solver": self.layout_algorithm,
                "stabilization": {
                    "enabled": self.initial_stabilization,
                    "iterations": 300,  # Reduced from 1000
                    "updateInterval": 50,  # Increased from 25
                    "fit": True,
                },
            },
            "edges": {
                "smooth": {
                    "type": "dynamic",
                    "forceDirection": "none",
                    "roundness": 0.5,
                },
                "color": {"inherit": "from", "opacity": 0.7},
                "arrows": {"to": {"enabled": True, "scaleFactor": 0.5}},
                "shadow": {
                    "enabled": False,  # Disabled for performance
                },
                "width": 1.5,  # Slightly thicker for better visibility
            },
            "nodes": {
                "shadow": {
                    "enabled": False,  # Disabled for performance
                },
                "scaling": {
                    "min": 8,  # Reduced from 10
                    "max": 20,  # Reduced from 30
                    "label": {
                        "enabled": True,
                        "min": 12,  # Reduced from 14
                        "max": 18,  # Reduced from 30
                        "maxVisible": 18,  # Reduced from 30
                        "drawThreshold": 5,
                    },
                },
                "font": {
                    "size": 14,  # Reduced from 16
                    "face": "Inter, system-ui, Avenir, Helvetica, Arial, sans-serif",
                    "color": "#FFFFFF" if self.dark_mode else "#000000",
                },
            },
            "interaction": {
                "hover": True,
                "hoverConnectedEdges": True,
                "selectConnectedEdges": True,
                "multiselect": True,
                "dragNodes": True,
                "dragView": True,
                "zoomView": True,
                "navigationButtons": True,
                "keyboard": {
                    "enabled": True,
                    "speed": {"x": 10, "y": 10, "zoom": 0.02},
                    "bindToWindow": True,
                },
            },
            "layout": {
                "randomSeed": 42,
                "improvedLayout": True,
                "hierarchical": {"enabled": False},
            },
            "configure": {
                "enabled": False,  # Disable configure button for cleaner UI
            },
            "renderer": {
                "renderingMode": self.render_mode,  # Ensure WebGL is used
            },
        }

    def generate_graph_data(self, nodes, edges):
        """
        Convert nodes and edges to the format required by the visualization library.
        Applies performance optimizations for large graphs.

        Parameters:
        -----------
        nodes : list
            List of node dictionaries
        edges : list
            List of edge dictionaries

        Returns:
        --------
        dict
            Graph data in the format required by the visualization library
        """
        # Performance optimizations for large graphs
        num_nodes = len(nodes)
        settings = self.default_settings.copy()

        if num_nodes > self.max_nodes_full_render:
            # For large graphs, apply performance optimizations
            settings["physics"]["stabilization"]["iterations"] = 100
            settings["edges"]["smooth"]["enabled"] = False
            settings["edges"]["smooth"]["type"] = "continuous"

            # Reduce visual complexity
            for node in nodes:
                # Simplify node appearance
                if "shadow" in node:
                    node["shadow"] = False
                if "size" in node and node["size"] > 15:
                    node["size"] = 15

            for edge in edges:
                # Simplify edge appearance
                if "shadow" in edge:
                    edge["shadow"] = False
                if "smooth" in edge:
                    edge["smooth"] = False

        return {"nodes": nodes, "edges": edges, "options": settings}

    def visualize_ontology(self, ontology: Ontology, output_path: Path):
        """
        Generate an interactive visualization of an ontology.

        Parameters:
        -----------
        ontology : Ontology
            The ontology to visualize
        output_path : Path
            Path to save the HTML visualization
        """
        # Node styling constants - reduced sizes for better performance
        CLASS_COLOR = "#4FC3F7"  # Light blue for classes
        DATATYPE_COLOR = "#BBDEFB"  # Lighter blue for datatypes
        PROPERTY_COLOR = "#FF9800"  # Orange for properties

        # Track which nodes we've already added to the graph
        added_nodes = set()
        nodes = []
        edges = []

        # Generate a unique ID for node names to avoid conflicts
        node_id_map = {}
        next_id = 1

        def get_node_id(name):
            if name not in node_id_map:
                nonlocal next_id
                node_id_map[name] = next_id
                next_id += 1
            return node_id_map[name]

        # Add a node with visual styling - optimized for performance
        def add_node(name, node_type="class", group=None, size=None):
            if name in added_nodes:
                return get_node_id(name)

            # Determine color based on node type
            if node_type == "class":
                color = CLASS_COLOR
                shape = self.shapes["class"]
                font_size = 14  # Reduced from 16
            elif node_type == "datatype":
                color = DATATYPE_COLOR
                shape = self.shapes["datatype"]
                font_size = 12  # Reduced from 14
            else:  # property
                color = PROPERTY_COLOR
                shape = self.shapes["property"]
                font_size = 12  # Reduced from 14

            # Set node size (either provided or based on node type) - reduced sizes
            if size is None:
                if node_type == "class":
                    size = 15  # Reduced from 20
                elif node_type == "datatype":
                    size = 10  # Reduced from 15
                else:
                    size = 8  # Reduced from 12

            # Create node with simplified styling for performance
            node = {
                "id": get_node_id(name),
                "label": name,
                "color": color,
                "shape": shape,
                "size": size,
                "font": {
                    "size": font_size,
                    "color": "#FFFFFF" if self.dark_mode else "#000000",
                },
                "title": f"<div style='max-width: 250px;'><h3>{name}</h3><p>Type: {node_type.capitalize()}</p></div>",
            }

            # Add group if provided (for clustering)
            if group:
                node["group"] = group

            nodes.append(node)
            added_nodes.add(name)
            return get_node_id(name)

        for cls in ontology.classes.values():
            add_node(cls.name, "class")
            if cls.superclass is not None:
                add_node(cls.superclass, "class")

                edges.append(
                    {
                        "from": get_node_id(cls.superclass),
                        "to": get_node_id(cls.name),
                        "label": "isA",
                        "font": {
                            "size": 9,  # Reduced from 10
                            "align": "middle",
                            "background": "#2E2E2E" if self.dark_mode else "#FFFFFF",
                        },
                        "arrows": {"to": {"enabled": True, "type": "arrow"}},
                        "dashes": [5, 5],  # Dashed line for isA relationships
                        "color": {"color": "#AAAAAA", "opacity": 0.7},
                    }
                )

        # From data properties
        for data_prop in ontology.data_properties.values():
            # Add datatype node
            datatype_name = data_prop.range
            add_node(datatype_name, "datatype")

        # 3. Add edges for object properties - with optimized styling
        for obj_prop in ontology.object_properties.values():
            for dom in obj_prop.domain:
                for rng in obj_prop.range:
                    source_id = get_node_id(dom)
                    target_id = get_node_id(rng)

                    edges.append(
                        {
                            "from": source_id,
                            "to": target_id,
                            "label": obj_prop.name,
                            "font": {
                                "size": 9,  # Reduced from 10
                                "align": "middle",
                                "background": "#2E2E2E" if self.dark_mode else "#FFFFFF",
                            },
                            "arrows": {"to": {"enabled": True, "type": "arrow"}},
                            "color": {
                                "color": "#FF9800",  # Orange for object properties
                                "opacity": 0.8,
                            },
                        }
                    )

        # 4. Add edges for data properties - with optimized styling
        for data_prop in ontology.data_properties.values():
            for dom in data_prop.domain:
                source_id = get_node_id(dom)
                target_id = get_node_id(data_prop.range)

                edges.append(
                    {
                        "from": source_id,
                        "to": target_id,
                        "label": data_prop.name,
                        "font": {
                            "size": 9,  # Reduced from 10
                            "align": "middle",
                            "background": "#2E2E2E" if self.dark_mode else "#FFFFFF",
                        },
                        "arrows": {"to": {"enabled": True, "type": "arrow"}},
                        "color": {
                            "color": "#4CAF50",  # Green for data properties
                            "opacity": 0.8,
                        },
                    }
                )

        # Generate graph data - with performance optimizations
        graph_data = self.generate_graph_data(nodes, edges)

        # Create the visualization
        self._create_visualization(
            graph_data,
            output_path,
            title=f"Ontology Visualization: {len(nodes)} classes & properties",
        )

    def visualize_kg(self, kg: KG, output_path: Path, ontology: Ontology | None = None):
        """
        Generate an interactive visualization of a knowledge graph.
        If an ontology is provided, entities will be colored according to their ontology class.

        Parameters:
        -----------
        kg : KG
            The knowledge graph to visualize
        output_path : Path
            Path to save the HTML visualization
        ontology : Ontology | None, optional
            Ontology to use for coloring entities based on their class
        """
        # High contrast palette for dark mode visualization
        palette = self.color_palette if self.dark_mode else self.light_palette
        default_color = "#D3D3D3"  # Light gray for entities with no known type

        # Track which nodes we've already added to the graph
        nodes = []
        edges = []
        added_nodes = set()

        # Generate a unique ID for node names to avoid conflicts
        node_id_map = {}
        next_id = 1

        def get_node_id(name):
            if name not in node_id_map:
                nonlocal next_id
                node_id_map[name] = next_id
                next_id += 1
            return node_id_map[name]

        # Step 1. Identify all 'isA' relationships to map entities to types
        entity_to_types = {}  # e.g., "Harold" -> {"Person", ...}
        recognized_types = set()  # e.g., {"Person", "City", "Organization", ...}

        for triplet in kg.triplets or []:
            if triplet.predicate.lower() == "isa":
                subj_name = triplet.subject
                obj_name = triplet.object
                recognized_types.add(obj_name)
                entity_to_types.setdefault(subj_name, set()).add(obj_name)

        # Step 2. Create color map for ontology classes if provided
        type_color_map = {}

        def get_type_color(type_name: str) -> str:
            if type_name not in type_color_map:
                # Cycle through the palette if we have more types than colors
                index = len(type_color_map) % len(palette)
                type_color_map[type_name] = palette[index]
            return type_color_map[type_name]

        # Step 3. Collect all entities (subjects + objects)
        all_entities = set()
        predicate_counts = {}  # Track frequency of each predicate type

        for triplet in kg.triplets or []:
            # Skip "isA" relationships if we have an ontology
            if ontology and triplet.predicate.lower() == "isa":
                continue

            all_entities.add(triplet.subject)
            all_entities.add(triplet.object)

            # Count predicates
            predicate_counts[triplet.predicate] = predicate_counts.get(triplet.predicate, 0) + 1

        # Calculate node sizes based on connections (degree centrality) - skip isA when considering
        node_connections = {}
        for triplet in kg.triplets or []:
            # Skip "isA" relationships if we have an ontology
            if ontology and triplet.predicate.lower() == "isa":
                continue

            node_connections[triplet.subject] = node_connections.get(triplet.subject, 0) + 1
            node_connections[triplet.object] = node_connections.get(triplet.object, 0) + 1

        # Normalize node sizes - with more reasonable scaling
        max_connections = max(node_connections.values()) if node_connections else 1

        # Step 4. Add nodes with optimized styling for performance
        for entity_name in all_entities:
            if entity_name in added_nodes:
                continue

            # Determine node size based on connectivity - reduced sizing
            connections = node_connections.get(entity_name, 1)

            # More reasonable sizing formula - cap at 25 instead of 40
            size = 6 + min((connections / max_connections) * 20, 20)

            # Determine node color and shape based on type
            color = default_color
            shape = self.shapes["default"]
            title = f"<div style='max-width: 250px;'><h3>{entity_name}</h3><p>Unclassified Entity</p><p>Connections: {connections}</p></div>"
            group = "unclassified"
            type_description = ""

            if entity_name in entity_to_types and len(entity_to_types[entity_name]) > 0:
                types = list(entity_to_types[entity_name])
                first_type = types[0]
                color = get_type_color(first_type)
                shape = self.shapes["individual"]
                group = first_type
                type_list = ", ".join(types)
                type_description = f"Types: {type_list}"
                title = f"<div style='max-width: 250px;'><h3>{entity_name}</h3><p>{type_description}</p><p>Connections: {connections}</p></div>"

            # Create node with optimized styling
            node = {
                "id": get_node_id(entity_name),
                "label": entity_name,
                "color": color,
                "shape": shape,
                "size": size,
                "group": group,
                "font": {
                    "size": 12,  # Reduced from 14
                    "color": "#FFFFFF" if self.dark_mode else "#000000",
                },
                "title": title,
            }

            nodes.append(node)
            added_nodes.add(entity_name)

        # Step 5. Add edges with styling based on predicate - optimized
        # Assign colors to predicates
        predicate_color_map = {}
        predicate_index = 0

        for triplet in kg.triplets or []:
            # Skip "isA" relationships if we have an ontology
            if ontology and triplet.predicate.lower() == "isa":
                continue

            source_id = get_node_id(triplet.subject)
            target_id = get_node_id(triplet.object)
            predicate = triplet.predicate

            # Assign color to predicate if not already assigned
            if predicate not in predicate_color_map:
                predicate_color_map[predicate] = palette[predicate_index % len(palette)]
                predicate_index += 1

            # Edge thickness based on predicate frequency - reduced range for performance
            frequency = predicate_counts[predicate]
            max_freq = max(predicate_counts.values())
            width = 0.5 + (frequency / max_freq) * 3  # Width between 0.5 and 3.5

            # Create edge with optimized styling
            edge = {
                "from": source_id,
                "to": target_id,
                "label": predicate,
                "width": width,
                "font": {
                    "size": 9,  # Reduced from 10
                    "align": "middle",
                    "background": "#2E2E2E" if self.dark_mode else "#FFFFFF",
                },
                "arrows": {
                    "to": {
                        "enabled": True,
                        "type": "arrow",
                        "scaleFactor": 0.4,  # Reduced from 0.5
                    }
                },
                "color": {
                    "color": predicate_color_map[predicate],
                    "opacity": 0.8,
                    "highlight": "#FFFFFF",
                },
                "title": f"<div>{triplet.subject} - <b>{predicate}</b> → {triplet.object}</div>",
            }

            edges.append(edge)

        # Generate graph data with performance optimizations
        graph_data = self.generate_graph_data(nodes, edges)

        # Create the visualization
        self._create_visualization(
            graph_data,
            output_path,
            title=f"Knowledge Graph: {len(nodes)} entities, {len(edges)} relationships",
        )

    def _create_visualization(self, graph_data, output_path, title="Graph Visualization"):
        """
        Create an HTML file with an interactive visualization - optimized for performance.

        Parameters:
        -----------
        graph_data : dict
            Graph data (nodes, edges, options)
        output_path : Path
            Path to save the HTML visualization
        title : str
            Title of the visualization
        """
        # JavaScript libraries to include - using specific versions for better compatibility
        vis_js = "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.js"
        vis_css = "https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/vis-network.min.css"
        chart_js = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"
        fuse_js = "https://cdnjs.cloudflare.com/ajax/libs/fuse.js/6.6.2/fuse.min.js"

        # Create the HTML content - optimized UI
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="{vis_css}">
    <script src="{vis_js}"></script>
    <script src="{chart_js}"></script>
    <script src="{fuse_js}"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: Inter, system-ui, Avenir, Helvetica, Arial, sans-serif;
        }}

        body {{
            background-color: {("#1E1E1E" if self.dark_mode else "#FFFFFF")};
            color: {("#FFFFFF" if self.dark_mode else "#000000")};
            overflow: hidden;
            height: 100vh;
            width: 100vw;
        }}

        #app {{
            display: flex;
            flex-direction: column;
            height: 100vh;
            width: 100vw;
        }}

        .navbar {{
            background-color: {("#333333" if self.dark_mode else "#F0F0F0")};
            padding: 8px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            z-index: 10;
        }}

        .title {{
            font-size: 16px;
            font-weight: bold;
        }}

        .dropdown {{
            position: relative;
            display: inline-block;
        }}

        .dropdown-content {{
            display: none;
            position: absolute;
            background-color: {("#444444" if self.dark_mode else "#FFFFFF")};
            min-width: 160px;
            box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
            z-index: 100;
            border-radius: 4px;
        }}

        .dropdown-content a {{
            color: {("#FFFFFF" if self.dark_mode else "#000000")};
            padding: 8px 12px;
            text-decoration: none;
            display: block;
            cursor: pointer;
        }}

        .dropdown-content a:hover {{
            background-color: {("#555555" if self.dark_mode else "#F0F0F0")};
        }}

        .dropdown:hover .dropdown-content {{
            display: block;
        }}

        .controls {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}

        .btn {{
            background-color: {("#444444" if self.dark_mode else "#E0E0E0")};
            color: {("#FFFFFF" if self.dark_mode else "#000000")};
            border: none;
            padding: 5px 10px;
            border-radius: 4px;
            cursor: pointer;
            transition: background-color 0.2s;
            font-size: 13px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}

        .btn:hover {{
            background-color: {("#555555" if self.dark_mode else "#D0D0D0")};
        }}

        .active {{
            background-color: {("#7E57C2" if self.dark_mode else "#9C27B0")};
            color: white;
        }}

        .main-container {{
            display: flex;
            flex: 1;
            overflow: hidden;
            position: relative;
        }}

        #graph-container {{
            flex: 1;
            position: relative;
        }}

        .sidebar {{
            width: 280px;
            background-color: {("#2D2D2D" if self.dark_mode else "#F5F5F5")};
            padding: 12px;
            overflow-y: auto;
            box-shadow: -2px 0 8px rgba(0,0,0,0.2);
            position: absolute;
            top: 0;
            right: 0;
            height: 100%;
            z-index: 5;
            transform: translateX(100%);
            transition: transform 0.3s ease;
        }}

        .sidebar.open {{
            transform: translateX(0);
        }}

        .panel {{
            margin-bottom: 12px;
            background-color: {("#3D3D3D" if self.dark_mode else "#FFFFFF")};
            border-radius: 5px;
            padding: 10px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }}

        .panel-header {{
            font-weight: bold;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 14px;
        }}

        .slider-container {{
            margin: 8px 0;
        }}

        .slider-label {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
            font-size: 12px;
        }}

        .slider {{
            width: 100%;
            background-color: {("#555555" if self.dark_mode else "#DDDDDD")};
            -webkit-appearance: none;
            height: 4px;
            border-radius: 4px;
            outline: none;
        }}

        .slider::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: {("#7E57C2" if self.dark_mode else "#9C27B0")};
            cursor: pointer;
        }}

        .checkbox-container {{
            display: flex;
            align-items: center;
            margin: 4px 0;
            font-size: 13px;
        }}

        .checkbox-container input {{
            margin-right: 8px;
        }}

        .stats-container {{
            margin-top: 12px;
            font-size: 13px;
        }}

        .stat {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }}

        .chart-container {{
            margin-top: 12px;
            height: 130px;
        }}

        .search-container {{
            margin-bottom: 12px;
        }}

        .search-input {{
            width: 100%;
            padding: 6px 8px;
            border-radius: 4px;
            border: 1px solid {("#555555" if self.dark_mode else "#DDDDDD")};
            background-color: {("#3D3D3D" if self.dark_mode else "#FFFFFF")};
            color: {("#FFFFFF" if self.dark_mode else "#000000")};
            font-size: 13px;
        }}

        .selection-details {{
            margin-top: 12px;
        }}

        .loading-screen {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: {("rgba(30, 30, 30, 0.8)" if self.dark_mode else "rgba(255, 255, 255, 0.8)")};
            display: flex;
            justify-content: center;
            align-items: center;
            z-index: 1000;
            flex-direction: column;
        }}

        .spinner {{
            border: 4px solid {("#3D3D3D" if self.dark_mode else "#f3f3f3")};
            border-top: 4px solid {("#7E57C2" if self.dark_mode else "#9C27B0")};
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin-bottom: 16px;
        }}

        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}

        .loading-text {{
            font-size: 16px;
        }}

        .legend {{
            position: absolute;
            bottom: 16px;
            left: 16px;
            background-color: {("rgba(45, 45, 45, 0.9)" if self.dark_mode else "rgba(245, 245, 245, 0.9)")};
            padding: 8px 12px;
            border-radius: 5px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            z-index: 5;
            max-width: 250px;
            max-height: 130px;
            overflow-y: auto;
            font-size: 12px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 4px;
        }}

        .legend-color {{
            width: 12px;
            height: 12px;
            border-radius: 3px;
            margin-right: 8px;
        }}

        .tooltip {{
            position: absolute;
            background-color: {("rgba(45, 45, 45, 0.9)" if self.dark_mode else "rgba(245, 245, 245, 0.9)")};
            padding: 8px;
            border-radius: 4px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            z-index: 100;
            max-width: 200px;
            display: none;
            font-size: 12px;
        }}

        .toggle-controls {{
            position: absolute;
            bottom: 16px;
            right: 16px;
            background-color: {("rgba(45, 45, 45, 0.9)" if self.dark_mode else "rgba(245, 245, 245, 0.9)")};
            padding: 6px;
            border-radius: 4px;
            z-index: 5;
            display: flex;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
        }}

        .mini-btn {{
            width: 24px;
            height: 24px;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            margin: 0 2px;
            border-radius: 3px;
            background-color: {("#444444" if self.dark_mode else "#E0E0E0")};
            color: {("#FFFFFF" if self.dark_mode else "#000000")};
            transition: background-color 0.2s;
            font-size: 14px;
        }}

        .mini-btn:hover {{
            background-color: {("#555555" if self.dark_mode else "#D0D0D0")};
        }}

        .network-tooltip {{
            background-color: {("rgba(45, 45, 45, 0.9)" if self.dark_mode else "rgba(245, 245, 245, 0.9)")};
            border: 1px solid {("#555555" if self.dark_mode else "#DDDDDD")};
            border-radius: 4px;
            padding: 8px;
            color: {("#FFFFFF" if self.dark_mode else "#000000")};
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            font-size: 13px;
            max-width: 250px;
        }}

        .filter-group {{
            margin-bottom: 10px;
        }}

        .filter-title {{
            font-weight: bold;
            margin-bottom: 5px;
            font-size: 13px;
        }}

        .filter-options {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }}

        .filter-option {{
            display: flex;
            align-items: center;
            background-color: {("#444444" if self.dark_mode else "#EEEEEE")};
            padding: 3px 6px;
            border-radius: 3px;
            font-size: 12px;
            cursor: pointer;
        }}

        .filter-option.active {{
            background-color: {("#7E57C2" if self.dark_mode else "#9C27B0")};
            color: white;
        }}

        .filter-color {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }}

        .search-options {{
            display: flex;
            margin-top: 5px;
            gap: 5px;
        }}

        .search-option {{
            padding: 3px 6px;
            background-color: {("#444444" if self.dark_mode else "#EEEEEE")};
            border-radius: 3px;
            font-size: 12px;
            cursor: pointer;
        }}

        .search-option.active {{
            background-color: {("#7E57C2" if self.dark_mode else "#9C27B0")};
            color: white;
        }}

        .search-results {{
            background-color: {("#3D3D3D" if self.dark_mode else "#FFFFFF")};
            border: 1px solid {("#555555" if self.dark_mode else "#DDDDDD")};
            border-radius: 4px;
            margin-top: 5px;
            max-height: 150px;
            overflow-y: auto;
        }}

        .search-result {{
            padding: 5px 8px;
            cursor: pointer;
            font-size: 12px;
            border-bottom: 1px solid {("#444444" if self.dark_mode else "#EEEEEE")};
        }}

        .search-result:hover {{
            background-color: {("#444444" if self.dark_mode else "#F0F0F0")};
        }}

        /* Hide the default vis.js tooltip */
        div.vis-tooltip {{
            display: none !important;
        }}

        /* Custom controls for navigation - better visibility */
        .vis-navigation-button {{
            background-color: {("rgba(45, 45, 45, 0.9)" if self.dark_mode else "rgba(245, 245, 245, 0.9)")};
            color: {("#FFFFFF" if self.dark_mode else "#000000")};
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            border: none !important;
            border-radius: 4px !important;
            font-size: 14px !important;
            width: 28px !important;
            height: 28px !important;
            margin: 2px !important;
        }}

        /* Fix for broken navigation bars */
        .vis-navigation div {{
            transform: none !important;
            filter: invert({("0%" if self.dark_mode else "100%")}) !important;
        }}
    </style>
</head>
<body>
    <div id="app">
        <div class="navbar">
            <div class="title">{title}</div>
            <div class="controls">
                <button id="btn-layout" class="btn">Auto Layout</button>
                <button id="btn-clusters" class="btn">Show Clusters</button>
                <div class="dropdown">
                    <button id="btn-algorithm" class="btn">Layout Options</button>
                    <div class="dropdown-content">
                        <a id="layout-force">Force-Directed</a>
                        <a id="layout-hierarchical">Hierarchical</a>
                        <a id="layout-radial">Radial</a>
                        <a id="layout-circular">Circular</a>
                    </div>
                </div>
                <button id="btn-filter" class="btn">Filter</button>
                <button id="btn-path" class="btn">Find Path</button>
                <button id="btn-insights" class="btn">Insights</button>
                <button id="btn-settings" class="btn">Settings</button>
                <button id="btn-export" class="btn">Export</button>
                <button id="btn-fullscreen" class="btn">Fullscreen</button>
            </div>
        </div>

        <div class="main-container">
            <div id="graph-container"></div>

            <div class="sidebar" id="sidebar">
                <div class="search-container">
                    <input type="text" class="search-input" id="search-input" placeholder="Search nodes and edges...">
                    <div class="search-options">
                        <div class="search-option active" data-type="all">All</div>
                        <div class="search-option" data-type="nodes">Nodes</div>
                        <div class="search-option" data-type="edges">Edges</div>
                        <div class="search-option" data-type="advanced">Advanced</div>
                    </div>
                    <div id="search-results" class="search-results" style="display: none;"></div>
                </div>

                <div class="panel filter-panel" id="filter-panel" style="display: none;">
                    <div class="panel-header">Filter Graph</div>

                    <div class="filter-group">
                        <div class="filter-title">Node Types</div>
                        <div class="filter-options" id="node-type-filters">
                            <!-- Will be filled dynamically -->
                        </div>
                    </div>

                    <div class="filter-group">
                        <div class="filter-title">Edge Types</div>
                        <div class="filter-options" id="edge-type-filters">
                            <!-- Will be filled dynamically -->
                        </div>
                    </div>

                    <div class="filter-group">
                        <div class="filter-title">Node Degree</div>
                        <div class="slider-container">
                            <div class="slider-label">
                                <span>Min Connections</span>
                                <span id="min-degree-value">1</span>
                            </div>
                            <input type="range" min="1" max="50" value="1" class="slider" id="min-degree-slider">
                        </div>
                    </div>

                    <button id="apply-filters" class="btn" style="width: 100%; margin-top: 10px;">Apply Filters</button>
                    <button id="reset-filters" class="btn" style="width: 100%; margin-top: 5px;">Reset Filters</button>
                </div>

                <div class="panel" id="path-finder-panel" style="display: none;">
                    <div class="panel-header">Find Shortest Path</div>

                    <div style="margin-bottom: 10px;">
                        <div style="margin-bottom: 5px; font-size: 13px;">Source Node</div>
                        <input type="text" class="search-input" id="source-node-input" placeholder="Select source node...">
                        <div id="source-node-results" class="search-results" style="display: none; max-height: 80px;"></div>
                    </div>

                    <div style="margin-bottom: 10px;">
                        <div style="margin-bottom: 5px; font-size: 13px;">Target Node</div>
                        <input type="text" class="search-input" id="target-node-input" placeholder="Select target node...">
                        <div id="target-node-results" class="search-results" style="display: none; max-height: 80px;"></div>
                    </div>

                    <button id="find-path" class="btn" style="width: 100%; margin-top: 10px;">Find Path</button>
                    <button id="clear-path" class="btn" style="width: 100%; margin-top: 5px; display: none;">Clear Path</button>

                    <div id="path-results" style="margin-top: 10px; display: none;">
                        <div class="filter-title">Path Found:</div>
                        <div id="path-results-content" style="margin-top: 5px; font-size: 12px;"></div>
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header">Physics Settings</div>

                    <div class="checkbox-container">
                        <input type="checkbox" id="physics-enabled" checked>
                        <label for="physics-enabled">Enable Physics</label>
                    </div>

                    <div class="slider-container">
                        <div class="slider-label">
                            <span>Gravitational Constant</span>
                            <span id="gravity-value">-50</span>
                        </div>
                        <input type="range" min="-500" max="0" value="-50" class="slider" id="gravity-slider">
                    </div>

                    <div class="slider-container">
                        <div class="slider-label">
                            <span>Spring Length</span>
                            <span id="spring-length-value">150</span>
                        </div>
                        <input type="range" min="50" max="500" value="150" class="slider" id="spring-length-slider">
                    </div>

                    <div class="slider-container">
                        <div class="slider-label">
                            <span>Spring Strength</span>
                            <span id="spring-strength-value">0.04</span>
                        </div>
                        <input type="range" min="0" max="1" step="0.01" value="0.04" class="slider" id="spring-strength-slider">
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header">Display Settings</div>

                    <div class="checkbox-container">
                        <input type="checkbox" id="show-labels" checked>
                        <label for="show-labels">Show Labels</label>
                    </div>

                    <div class="checkbox-container">
                        <input type="checkbox" id="smooth-edges" checked>
                        <label for="smooth-edges">Smooth Edges</label>
                    </div>

                    <div class="slider-container">
                        <div class="slider-label">
                            <span>Node Size</span>
                            <span id="node-size-value">1</span>
                        </div>
                        <input type="range" min="0.5" max="2" step="0.1" value="1" class="slider" id="node-size-slider">
                    </div>

                    <div class="slider-container">
                        <div class="slider-label">
                            <span>Edge Width</span>
                            <span id="edge-width-value">1</span>
                        </div>
                        <input type="range" min="0.5" max="3" step="0.1" value="1" class="slider" id="edge-width-slider">
                    </div>
                </div>

                <div class="panel">
                    <div class="panel-header">Graph Statistics</div>

                    <div class="stats-container">
                        <div class="stat">
                            <span>Nodes:</span>
                            <span id="stat-nodes">{len(graph_data["nodes"])}</span>
                        </div>
                        <div class="stat">
                            <span>Edges:</span>
                            <span id="stat-edges">{len(graph_data["edges"])}</span>
                        </div>
                        <div class="stat">
                            <span>Average Degree:</span>
                            <span id="stat-avg-degree">-</span>
                        </div>
                        <div class="stat">
                            <span>Edge Types:</span>
                            <span id="stat-edge-types">-</span>
                        </div>
                        <div class="stat">
                            <span>Node Types:</span>
                            <span id="stat-node-types">-</span>
                        </div>
                    </div>

                    <div class="chart-container">
                        <canvas id="degree-distribution-chart"></canvas>
                    </div>
                </div>

                <div class="panel selection-details" id="selection-details">
                    <div class="panel-header">Selection Details</div>
                    <div id="selection-content">Select a node or edge to see details</div>
                </div>
            </div>
        </div>
    </div>

    <div class="legend" id="legend"></div>

    <!-- Removed duplicate tooltip, keeping only the transparent one -->

    <div class="toggle-controls">
        <div class="mini-btn" id="btn-zoom-in" title="Zoom In">+</div>
        <div class="mini-btn" id="btn-zoom-out" title="Zoom Out">−</div>
        <div class="mini-btn" id="btn-fit" title="Fit View">⊡</div>
    </div>

    <div class="loading-screen" id="loading-screen">
        <div class="spinner"></div>
        <div class="loading-text">Initializing Graph Visualization...</div>
    </div>

    <script>
        // Graph data from Python
        const graphData = {json.dumps(graph_data)};

        // Utility functions
        function getNodeDegree(nodeId) {{
            let degree = 0;
            graphData.edges.forEach(edge => {{
                if (edge.from === nodeId || edge.to === nodeId) {{
                    degree++;
                }}
            }});
            return degree;
        }}

        function getConnectedNodes(nodeId) {{
            const connectedNodes = new Set();
            graphData.edges.forEach(edge => {{
                if (edge.from === nodeId) {{
                    connectedNodes.add(edge.to);
                }} else if (edge.to === nodeId) {{
                    connectedNodes.add(edge.from);
                }}
            }});
            return Array.from(connectedNodes);
        }}

        function getNodeGroups() {{
            const groups = new Map();
            graphData.nodes.forEach(node => {{
                if (node.group) {{
                    if (!groups.has(node.group)) {{
                        groups.set(node.group, []);
                    }}
                    groups.get(node.group).push(node.id);
                }}
            }});
            return groups;
        }}

        function getEdgeTypes() {{
            const types = new Set();
            graphData.edges.forEach(edge => {{
                if (edge.label) {{
                    types.add(edge.label);
                }}
            }});
            return Array.from(types);
        }}

        function calculateDegreeDistribution() {{
            const distribution = [];
            graphData.nodes.forEach(node => {{
                const degree = getNodeDegree(node.id);
                distribution.push(degree);
            }});
            return distribution;
        }}

        function calculateAverageDegree() {{
            const distribution = calculateDegreeDistribution();
            return distribution.reduce((sum, degree) => sum + degree, 0) / distribution.length;
        }}

        function getNodeById(id) {{
            return graphData.nodes.find(node => node.id === id);
        }}

        function getEdgeByNodes(fromId, toId) {{
            return graphData.edges.find(edge => edge.from === fromId && edge.to === toId);
        }}

        // DOM elements
        const container = document.getElementById('graph-container');
        const sidebar = document.getElementById('sidebar');
        const btnSettings = document.getElementById('btn-settings');
        const btnLayout = document.getElementById('btn-layout');
        const btnClusters = document.getElementById('btn-clusters');
        const btnExport = document.getElementById('btn-export');
        const btnFullscreen = document.getElementById('btn-fullscreen');
        const btnFilter = document.getElementById('btn-filter');
        const btnPath = document.getElementById('btn-path');
        const btnInsights = document.getElementById('btn-insights');
        const searchInput = document.getElementById('search-input');
        const searchResults = document.getElementById('search-results');
        const searchOptions = document.querySelectorAll('.search-option');
        const filterPanel = document.getElementById('filter-panel');
        const pathFinderPanel = document.getElementById('path-finder-panel');
        const sourceNodeInput = document.getElementById('source-node-input');
        const targetNodeInput = document.getElementById('target-node-input');
        const sourceNodeResults = document.getElementById('source-node-results');
        const targetNodeResults = document.getElementById('target-node-results');
        const findPathBtn = document.getElementById('find-path');
        const clearPathBtn = document.getElementById('clear-path');
        const pathResults = document.getElementById('path-results');
        const pathResultsContent = document.getElementById('path-results-content');
        const loadingScreen = document.getElementById('loading-screen');
        const legend = document.getElementById('legend');
        const selectionContent = document.getElementById('selection-content');
        const btnZoomIn = document.getElementById('btn-zoom-in');
        const btnZoomOut = document.getElementById('btn-zoom-out');
        const btnFit = document.getElementById('btn-fit');

        // Layout options elements
        const layoutForce = document.getElementById('layout-force');
        const layoutHierarchical = document.getElementById('layout-hierarchical');
        const layoutRadial = document.getElementById('layout-radial');
        const layoutCircular = document.getElementById('layout-circular');

        // Physics settings
        const physicsEnabled = document.getElementById('physics-enabled');
        const gravitySlider = document.getElementById('gravity-slider');
        const gravityValue = document.getElementById('gravity-value');
        const springLengthSlider = document.getElementById('spring-length-slider');
        const springLengthValue = document.getElementById('spring-length-value');
        const springStrengthSlider = document.getElementById('spring-strength-slider');
        const springStrengthValue = document.getElementById('spring-strength-value');

        // Display settings
        const showLabels = document.getElementById('show-labels');
        const smoothEdges = document.getElementById('smooth-edges');
        const nodeSizeSlider = document.getElementById('node-size-slider');
        const nodeSizeValue = document.getElementById('node-size-value');
        const edgeWidthSlider = document.getElementById('edge-width-slider');
        const edgeWidthValue = document.getElementById('edge-width-value');

        // Statistics elements
        const statNodes = document.getElementById('stat-nodes');
        const statEdges = document.getElementById('stat-edges');
        const statAvgDegree = document.getElementById('stat-avg-degree');
        const statEdgeTypes = document.getElementById('stat-edge-types');
        const statNodeTypes = document.getElementById('stat-node-types');

        // Initialize the network
        let network;
        let nodes;
        let edges;

        // For search functionality
        let fuseNodes;
        let fuseEdges;
        let searchType = 'all';

        // For path finding functionality
        let shortestPath = [];
        let pathHighlightedEdges = new Set();
        let selectedSourceNode = null;
        let selectedTargetNode = null;

        // Initialize network when document is fully loaded
        document.addEventListener('DOMContentLoaded', initNetwork);

        function initNetwork() {{
            try {{
                // Create datasets
                nodes = new vis.DataSet(graphData.nodes);
                edges = new vis.DataSet(graphData.edges);

                // Create network with enhanced performance options
                const options = graphData.options;

                // Set correct renderer
                options.renderer = {{ renderingMode: '{self.render_mode}' }};

                // Enhanced interaction settings
                options.interaction = {{
                    ...options.interaction,
                    hideEdgesOnDrag: true,    // Hide edges while dragging
                    hideNodesOnDrag: false,   // Keep nodes visible
                    multiselect: true,
                    navigationButtons: true,
                    tooltipDelay: 300,
                    zoomSpeed: 0.5,           // Slower zoom for more control
                }};

                // Create the network
                network = new vis.Network(container, {{ nodes, edges }}, options);

                // Initialize search functionality
                initSearch();

                // Event listeners
                network.on('stabilizationProgress', function(params) {{
                    const progress = Math.round(params.iterations / params.total * 100);
                    document.querySelector('.loading-text').textContent = `Stabilizing layout: ${{progress}}%`;
                }});

                network.on('stabilizationIterationsDone', function() {{
                    loadingScreen.style.display = 'none';

                    // After stabilization, disable physics for performance if there are many nodes
                    if (graphData.nodes.length > 300) {{
                        setTimeout(() => {{
                            network.setOptions({{ physics: {{ enabled: false }} }});
                            physicsEnabled.checked = false;
                        }}, 500);
                    }}

                    updateStatistics();
                    createLegend();
                }});

                network.on('click', function(params) {{
                    if (params.nodes.length > 0) {{
                        const nodeId = params.nodes[0];
                        const node = getNodeById(nodeId);
                        showNodeDetails(node);
                    }} else if (params.edges.length > 0) {{
                        const edgeId = params.edges[0];
                        const edge = edges.get(edgeId);
                        showEdgeDetails(edge);
                    }} else {{
                        selectionContent.innerHTML = 'Select a node or edge to see details';
                    }}
                }});

                // Custom tooltip handling to avoid duplicate tooltips
                network.on('hoverNode', function(params) {{
                    const nodeId = params.node;
                    const node = getNodeById(nodeId);
                    const position = network.getPositions([nodeId])[nodeId];
                    const canvasPosition = network.canvasToDOM(position);

                    // Create custom tooltip
                    const tooltip = document.createElement('div');
                    tooltip.className = 'network-tooltip';
                    tooltip.innerHTML = node.title || node.label;
                    tooltip.style.position = 'absolute';
                    tooltip.style.left = `${{canvasPosition.x + 10}}px`;
                    tooltip.style.top = `${{canvasPosition.y + 10}}px`;
                    tooltip.style.zIndex = 1000;

                    // Remove any existing tooltips
                    const existingTooltips = document.querySelectorAll('.network-tooltip');
                    existingTooltips.forEach(el => el.remove());

                    document.body.appendChild(tooltip);
                }});

                network.on('blurNode', function() {{
                    // Remove custom tooltips
                    const existingTooltips = document.querySelectorAll('.network-tooltip');
                    existingTooltips.forEach(el => el.remove());
                }});

                // Initialize charts
                initCharts();

                // Attach event handlers
                attachEventHandlers();

            }} catch (error) {{
                console.error('Error initializing network:', error);
                document.querySelector('.loading-text').textContent =
                    'Error initializing visualization. Please check the console for details.';
            }}
        }}

        function initSearch() {{
            // Initialize Fuse.js for fuzzy search
            fuseNodes = new Fuse(graphData.nodes, {{
                keys: ['label', 'title', 'group'],
                threshold: 0.3,
                ignoreLocation: true,
                includeScore: true
            }});

            fuseEdges = new Fuse(graphData.edges, {{
                keys: ['label', 'title', 'from', 'to'],
                threshold: 0.3,
                ignoreLocation: true,
                includeScore: true
            }});
        }}

        function attachEventHandlers() {{
            // Settings button
            btnSettings.addEventListener('click', function() {{
                sidebar.classList.toggle('open');
            }});

            // Auto layout button
            btnLayout.addEventListener('click', function() {{
                loadingScreen.style.display = 'flex';
                document.querySelector('.loading-text').textContent = 'Optimizing layout...';

                // Use setTimeout to allow UI to update before heavy operation
                setTimeout(() => {{
                    // Enable physics temporarily for layout
                    network.setOptions({{ physics: {{ enabled: true }} }});
                    physicsEnabled.checked = true;

                    // Run stabilization with limited iterations for speed
                    network.stabilize(100);

                    // After stabilization, reset to user's physics preference
                    setTimeout(() => {{
                        loadingScreen.style.display = 'none';
                    }}, 1000);
                }}, 50);
            }});

            // Physics toggle
            physicsEnabled.addEventListener('change', function() {{
                network.setOptions({{ physics: {{ enabled: this.checked }} }});
            }});

            // Physics sliders
            gravitySlider.addEventListener('input', function() {{
                const value = parseInt(this.value);
                gravityValue.textContent = value;
                network.setOptions({{ physics: {{ forceAtlas2Based: {{ gravitationalConstant: value }} }} }});
            }});

            springLengthSlider.addEventListener('input', function() {{
                const value = parseInt(this.value);
                springLengthValue.textContent = value;
                network.setOptions({{ physics: {{ forceAtlas2Based: {{ springLength: value }} }} }});
            }});

            springStrengthSlider.addEventListener('input', function() {{
                const value = parseFloat(this.value);
                springStrengthValue.textContent = value.toFixed(2);
                network.setOptions({{ physics: {{ forceAtlas2Based: {{ springConstant: value }} }} }});
            }});

            // Display settings
            showLabels.addEventListener('change', function() {{
                if (this.checked) {{
                    // When enabling labels, restore original font sizes
                    nodes.forEach(node => {{
                        nodes.update({{
                            id: node.id,
                            font: {{
                                size: node.originalFontSize || 14
                            }}
                        }});
                    }});
                }} else {{
                    // When disabling, store original sizes and set to 0
                    nodes.forEach(node => {{
                        if (!node.originalFontSize && node.font) {{
                            nodes.update({{
                                id: node.id,
                                originalFontSize: node.font.size || 14
                            }});
                        }}
                        nodes.update({{
                            id: node.id,
                            font: {{ size: 0 }}
                        }});
                    }});
                }}
            }});

            smoothEdges.addEventListener('change', function() {{
                network.setOptions({{
                    edges: {{
                        smooth: {{
                            enabled: this.checked,
                            type: this.checked ? 'dynamic' : 'continuous'
                        }}
                    }}
                }});
            }});

            // Node size slider with optimized scaling
            nodeSizeSlider.addEventListener('input', function() {{
                const value = parseFloat(this.value);
                nodeSizeValue.textContent = value.toFixed(1);

                // Batch updates for better performance
                const updates = [];
                nodes.forEach(node => {{
                    const originalSize = node.originalSize || node.size || 10;
                    updates.push({{
                        id: node.id,
                        size: originalSize * value
                    }});

                    // Store original size if not already stored
                    if (!node.originalSize) {{
                        nodes.update({{ id: node.id, originalSize: node.size || 10 }});
                    }}
                }});

                // Apply all updates at once
                nodes.update(updates);
            }});

            // Edge width slider with optimized scaling
            edgeWidthSlider.addEventListener('input', function() {{
                const value = parseFloat(this.value);
                edgeWidthValue.textContent = value.toFixed(1);

                // Batch updates for performance
                const updates = [];
                edges.forEach(edge => {{
                    const originalWidth = edge.originalWidth || edge.width || 1;
                    updates.push({{
                        id: edge.id,
                        width: originalWidth * value
                    }});

                    // Store original width if not already stored
                    if (!edge.originalWidth) {{
                        edges.update({{ id: edge.id, originalWidth: edge.width || 1 }});
                    }}
                }});

                // Apply all updates at once
                edges.update(updates);
            }});

            // Zoom controls
            btnZoomIn.addEventListener('click', function() {{
                network.zoom(1.2, {{
                    animation: {{
                        duration: 200,
                        easingFunction: 'easeInOutQuad'
                    }}
                }});
            }});

            btnZoomOut.addEventListener('click', function() {{
                network.zoom(0.8, {{
                    animation: {{
                        duration: 200,
                        easingFunction: 'easeInOutQuad'
                    }}
                }});
            }});

            btnFit.addEventListener('click', function() {{
                network.fit({{
                    animation: {{
                        duration: 500,
                        easingFunction: 'easeInOutQuad'
                    }}
                }});
            }});

            // Search functionality
            searchInput.addEventListener('input', performSearch);

            // Layout algorithm options
            layoutForce.addEventListener('click', function() {{
                applyLayout('forceAtlas2Based');
            }});

            layoutHierarchical.addEventListener('click', function() {{
                applyLayout('hierarchical');
            }});

            layoutRadial.addEventListener('click', function() {{
                applyRadialLayout();
            }});

            layoutCircular.addEventListener('click', function() {{
                applyCircularLayout();
            }});

            // Filter button
            btnFilter.addEventListener('click', function() {{
                // Close path panel if open
                pathFinderPanel.style.display = 'none';

                // Toggle filter panel
                if (filterPanel.style.display === 'none') {{
                    filterPanel.style.display = 'block';
                    populateFilterPanel();
                }} else {{
                    filterPanel.style.display = 'none';
                }}

                // Open sidebar if closed
                if (!sidebar.classList.contains('open')) {{
                    sidebar.classList.add('open');
                }}
            }});

            // Path finding button
            btnPath.addEventListener('click', function() {{
                // Close filter panel if open
                filterPanel.style.display = 'none';

                // Toggle path panel
                if (pathFinderPanel.style.display === 'none') {{
                    pathFinderPanel.style.display = 'block';
                }} else {{
                    pathFinderPanel.style.display = 'none';
                }}

                // Open sidebar if closed
                if (!sidebar.classList.contains('open')) {{
                    sidebar.classList.add('open');
                }}
            }});

            // Export button
            btnExport.addEventListener('click', function() {{
                // Get network as image
                const dataUrl = network.canvas.frame.canvas.toDataURL('image/png');

                // Create download link
                const link = document.createElement('a');
                link.href = dataUrl;
                link.download = 'graph_visualization.png';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            }});

            // Fullscreen button
            btnFullscreen.addEventListener('click', function() {{
                if (!document.fullscreenElement) {{
                    container.requestFullscreen().catch(err => {{
                        console.error('Error attempting to enable fullscreen:', err);
                    }});
                }} else {{
                    document.exitFullscreen();
                }}
            }});

            // Clusters button
            btnClusters.addEventListener('click', function() {{
                this.classList.toggle('active');

                if (this.classList.contains('active')) {{
                    // Show loading screen during clustering
                    loadingScreen.style.display = 'flex';
                    document.querySelector('.loading-text').textContent = 'Creating clusters...';

                    // Defer clustering to allow UI update
                    setTimeout(() => {{
                        try {{
                            createClusters();
                            loadingScreen.style.display = 'none';
                        }} catch (error) {{
                            console.error('Error creating clusters:', error);
                            loadingScreen.style.display = 'none';
                        }}
                    }}, 50);
                }} else {{
                    // Show loading screen during opening clusters
                    loadingScreen.style.display = 'flex';
                    document.querySelector('.loading-text').textContent = 'Opening clusters...';

                    // Defer operation to allow UI update
                    setTimeout(() => {{
                        try {{
                            network.openCluster('*');
                            loadingScreen.style.display = 'none';
                        }} catch (error) {{
                            console.error('Error opening clusters:', error);
                            loadingScreen.style.display = 'none';
                        }}
                    }}, 50);
                }}
            }});
        }}

        function createClusters() {{
            // Group nodes by their group property
            const nodeGroups = getNodeGroups();

            // Don't cluster if no groups
            if (nodeGroups.size === 0) {{
                alert('No node groups found to create clusters');
                btnClusters.classList.remove('active');
                return;
            }}

            nodeGroups.forEach((nodeIds, groupName) => {{
                // Find the average position of all nodes in this group
                const positions = network.getPositions(nodeIds);
                let avgX = 0, avgY = 0;
                let count = 0;

                nodeIds.forEach(id => {{
                    if (positions[id]) {{
                        avgX += positions[id].x;
                        avgY += positions[id].y;
                        count++;
                    }}
                }});

                if (count > 0) {{
                    avgX /= count;
                    avgY /= count;

                    // Create cluster
                    network.cluster({{
                        joinCondition: function(nodeOptions) {{
                            return nodeOptions.group === groupName;
                        }},
                        processProperties: function(clusterOptions, childNodes, childEdges) {{
                            const sampleNode = getNodeById(childNodes[0].id);

                            clusterOptions.label = `${{groupName}} (${{childNodes.length}})`;
                            clusterOptions.color = childNodes[0].color || sampleNode.color;
                            clusterOptions.shape = 'dot';
                            clusterOptions.size = 20 + Math.min(childNodes.length, 20); // Cap size for very large clusters
                            clusterOptions.font = {{
                                size: 14,
                                color: '{("#FFFFFF" if self.dark_mode else "#000000")}',
                                face: 'Inter, system-ui, Avenir, Helvetica, Arial, sans-serif',
                            }};
                            clusterOptions.mass = Math.min(childNodes.length, 50); // Cap mass for stability
                            clusterOptions.borderWidth = 2;
                            clusterOptions.borderWidthSelected = 3;

                            return clusterOptions;
                        }},
                        clusterNodeProperties: {{
                            x: avgX,
                            y: avgY,
                            fixed: {{ x: false, y: false }} // Allow cluster to move with physics
                        }}
                    }});
                }}
            }});
        }}

        function applyLayout(layoutType) {{
            loadingScreen.style.display = 'flex';
            document.querySelector('.loading-text').textContent = `Applying ${{layoutType}} layout...`;

            setTimeout(() => {{
                try {{
                    if (layoutType === 'hierarchical') {{
                        network.setOptions({{
                            layout: {{
                                hierarchical: {{
                                    enabled: true,
                                    direction: 'UD',
                                    sortMethod: 'directed',
                                    nodeSpacing: 150,
                                    levelSeparation: 150
                                }}
                            }},
                            physics: {{ enabled: false }}
                        }});
                    }} else {{
                        // Reset to force-directed layout
                        network.setOptions({{
                            layout: {{
                                hierarchical: {{
                                    enabled: false
                                }},
                                improvedLayout: true,
                                randomSeed: 42
                            }},
                            physics: {{
                                enabled: true,
                                solver: layoutType
                            }}
                        }});

                        // Stabilize with limited iterations
                        network.stabilize(100);
                    }}

                    loadingScreen.style.display = 'none';
                }} catch (error) {{
                    console.error('Error applying layout:', error);
                    loadingScreen.style.display = 'none';
                }}
            }}, 50);
        }}

        function applyRadialLayout() {{
            loadingScreen.style.display = 'flex';
            document.querySelector('.loading-text').textContent = 'Applying radial layout...';

            setTimeout(() => {{
                try {{
                    // Find node with highest degree for center
                    let centerNodeId = 1;
                    let maxDegree = 0;

                    nodes.forEach(node => {{
                        const degree = getNodeDegree(node.id);
                        if (degree > maxDegree) {{
                            maxDegree = degree;
                            centerNodeId = node.id;
                        }}
                    }});

                    // Get all nodes and calculate positions in a radial layout
                    const allNodes = nodes.get();
                    const radius = 300; // Base radius
                    const angleStep = (2 * Math.PI) / allNodes.length;

                    // Position center node
                    network.moveNode(centerNodeId, 0, 0);

                    // Arrange other nodes in concentric circles
                    let currentAngle = 0;
                    let updatedPositions = [];

                    // Create levels based on distance from center node
                    const nodeDistances = calculateNodeDistances(centerNodeId);
                    const maxLevel = Math.max(...Object.values(nodeDistances));

                    allNodes.forEach(node => {{
                        if (node.id === centerNodeId) return; // Skip center node

                        const level = nodeDistances[node.id] || maxLevel;
                        const levelRadius = radius * (level / maxLevel);

                        const x = levelRadius * Math.cos(currentAngle);
                        const y = levelRadius * Math.sin(currentAngle);

                        updatedPositions.push({{
                            id: node.id,
                            x: x,
                            y: y
                        }});

                        currentAngle += angleStep;
                    }});

                    // Batch update positions
                    network.setOptions({{ physics: {{ enabled: false }} }});

                    // Apply positions in batches to avoid UI freeze
                    const batchSize = 50;
                    for (let i = 0; i < updatedPositions.length; i += batchSize) {{
                        const batch = updatedPositions.slice(i, i + batchSize);
                        batch.forEach(pos => {{
                            network.moveNode(pos.id, pos.x, pos.y);
                        }});
                    }}

                    loadingScreen.style.display = 'none';
                }} catch (error) {{
                    console.error('Error applying radial layout:', error);
                    loadingScreen.style.display = 'none';
                }}
            }}, 50);
        }}

        function calculateNodeDistances(startNodeId) {{
            // Simplified BFS to calculate distances from start node
            const distances = {{}};
            const queue = [[startNodeId, 0]];
            const visited = new Set([startNodeId]);

            while (queue.length > 0) {{
                const [nodeId, distance] = queue.shift();
                distances[nodeId] = distance;

                const neighbors = getConnectedNodes(nodeId);
                for (const neighbor of neighbors) {{
                    if (!visited.has(neighbor)) {{
                        visited.add(neighbor);
                        queue.push([neighbor, distance + 1]);
                    }}
                }}
            }}

            return distances;
        }}

        function applyCircularLayout() {{
            loadingScreen.style.display = 'flex';
            document.querySelector('.loading-text').textContent = 'Applying circular layout...';

            setTimeout(() => {{
                try {{
                    // Get all nodes
                    const allNodes = nodes.get();
                    const nodeCount = allNodes.length;

                    // Calculate circle parameters
                    const radius = Math.min(500, 20 * Math.sqrt(nodeCount));
                    const angleStep = (2 * Math.PI) / nodeCount;

                    // Disable physics during layout
                    network.setOptions({{ physics: {{ enabled: false }} }});

                    // Position nodes in a circle
                    let currentAngle = 0;
                    let updatedPositions = [];

                    allNodes.forEach(node => {{
                        const x = radius * Math.cos(currentAngle);
                        const y = radius * Math.sin(currentAngle);

                        updatedPositions.push({{
                            id: node.id,
                            x: x,
                            y: y
                        }});

                        currentAngle += angleStep;
                    }});

                    // Apply positions in batches
                    const batchSize = 50;
                    for (let i = 0; i < updatedPositions.length; i += batchSize) {{
                        const batch = updatedPositions.slice(i, i + batchSize);
                        batch.forEach(pos => {{
                            network.moveNode(pos.id, pos.x, pos.y);
                        }});
                    }}

                    loadingScreen.style.display = 'none';
                }} catch (error) {{
                    console.error('Error applying circular layout:', error);
                    loadingScreen.style.display = 'none';
                }}
            }}, 50);
        }}

        function performSearch() {{
            const query = searchInput.value.toLowerCase();

            if (query.length < 2) {{
                searchResults.style.display = 'none';

                // Reset all nodes to original state
                nodes.forEach(node => {{
                    if (node.originalColor) {{
                        nodes.update({{ id: node.id, color: node.originalColor }});
                    }}
                }});

                return;
            }}

            // Perform search based on type
            let results = [];

            if (searchType === 'nodes' || searchType === 'all') {{
                const nodeResults = fuseNodes.search(query);
                nodeResults.forEach(result => {{
                    results.push({{
                        type: 'node',
                        item: result.item,
                        score: result.score
                    }});
                }});
            }}

            if (searchType === 'edges' || searchType === 'all') {{
                const edgeResults = fuseEdges.search(query);
                edgeResults.forEach(result => {{
                    results.push({{
                        type: 'edge',
                        item: result.item,
                        score: result.score
                    }});
                }});
            }}

            // Sort by score
            results.sort((a, b) => a.score - b.score);

            // Limit results
            results = results.slice(0, 10);

            // Display results
            if (results.length > 0) {{
                let html = '';

                results.forEach(result => {{
                    if (result.type === 'node') {{
                        html += `<div class="search-result" data-type="node" data-id="${{result.item.id}}">
                            <b>Node:</b> ${{result.item.label || 'Node ' + result.item.id}}
                        </div>`;
                    }} else {{
                        const fromNode = getNodeById(result.item.from);
                        const toNode = getNodeById(result.item.to);

                        html += `<div class="search-result" data-type="edge" data-from="${{result.item.from}}" data-to="${{result.item.to}}">
                            <b>Edge:</b> ${{fromNode.label || 'Node ' + result.item.from}} →
                            ${{result.item.label || ''}} →
                            ${{toNode.label || 'Node ' + result.item.to}}
                        </div>`;
                    }}
                }});

                searchResults.innerHTML = html;
                searchResults.style.display = 'block';

                // Add click handlers
                const resultItems = searchResults.querySelectorAll('.search-result');
                resultItems.forEach(item => {{
                    item.addEventListener('click', function() {{
                        const type = this.dataset.type;

                        if (type === 'node') {{
                            const nodeId = parseInt(this.dataset.id);
                            highlightNode(nodeId);
                        }} else {{
                            const fromId = parseInt(this.dataset.from);
                            const toId = parseInt(this.dataset.to);
                            highlightEdge(fromId, toId);
                        }}

                        searchResults.style.display = 'none';
                    }});
                }});

                // Highlight matching nodes
                highlightMatchingNodes(results);
            }} else {{
                searchResults.innerHTML = '<div class="search-result">No results found</div>';
                searchResults.style.display = 'block';
            }}
        }}

        function highlightMatchingNodes(results) {{
            // Store original colors if not already stored
            nodes.forEach(node => {{
                if (!node.originalColor) {{
                    nodes.update({{ id: node.id, originalColor: node.color }});
                }}
            }});

            // Dim all nodes
            nodes.forEach(node => {{
                const dimmedColor = typeof node.originalColor === 'object'
                    ? {{ background: '#333333', border: '#555555' }}
                    : '#333333';
                nodes.update({{ id: node.id, color: dimmedColor }});
            }});

            // Highlight matching nodes
            const matchingNodeIds = results
                .filter(result => result.type === 'node')
                .map(result => result.item.id);

            matchingNodeIds.forEach(nodeId => {{
                const node = getNodeById(nodeId);
                nodes.update({{ id: nodeId, color: node.originalColor }});
            }});

            // If there are results, focus on first match
            if (matchingNodeIds.length > 0) {{
                network.focus(matchingNodeIds[0], {{
                    scale: 1.2,
                    animation: {{
                        duration: 500,
                        easingFunction: 'easeInOutQuad'
                    }}
                }});
            }}
        }}

        function highlightNode(nodeId) {{
            // Clear any previous highlights
            clearHighlights();

            // Highlight the node
            const node = getNodeById(nodeId);

            // Focus on the node
            network.focus(nodeId, {{
                scale: 1.5,
                animation: {{
                    duration: 500,
                    easingFunction: 'easeInOutQuad'
                }}
            }});

            // Select the node
            network.selectNodes([nodeId]);

            // Show node details
            showNodeDetails(node);
        }}

        function highlightEdge(fromId, toId) {{
            // Clear any previous highlights
            clearHighlights();

            // Find the edge
            const edge = edges.get().find(e => e.from === fromId && e.to === toId);

            if (edge) {{
                // Select the edge
                network.selectEdges([edge.id]);

                // Focus on the edge
                const fromPos = network.getPositions([fromId])[fromId];
                const toPos = network.getPositions([toId])[toId];

                const centerX = (fromPos.x + toPos.x) / 2;
                const centerY = (fromPos.y + toPos.y) / 2;

                network.moveTo({{
                    position: {{ x: centerX, y: centerY }},
                    scale: 1.2,
                    animation: {{
                        duration: 500,
                        easingFunction: 'easeInOutQuad'
                    }}
                }});

                // Show edge details
                showEdgeDetails(edge);
            }}
        }}

        function clearHighlights() {{
            // Reset node colors if changed
            nodes.forEach(node => {{
                if (node.originalColor) {{
                    nodes.update({{ id: node.id, color: node.originalColor }});
                }}
            }});

            // Reset edge colors if changed
            edges.forEach(edge => {{
                if (edge.originalColor) {{
                    edges.update({{ id: edge.id, color: edge.originalColor }});
                }}
            }});

            // Clear selection
            network.unselectAll();
        }}

        function populateFilterPanel() {{
            // Get all node groups
            const nodeGroups = getNodeGroups();
            let nodeTypeHtml = '';

            nodeGroups.forEach((nodeIds, groupName) => {{
                // Get color from first node in group
                let color = '#AAAAAA';
                if (nodeIds.length > 0) {{
                    const node = getNodeById(nodeIds[0]);
                    color = node.color;
                    if (typeof color === 'object') {{
                        color = color.background || color.border || '#AAAAAA';
                    }}
                }}

                nodeTypeHtml += `
                    <div class="filter-option active" data-group="${{groupName}}">
                        <div class="filter-color" style="background-color: ${{color}}"></div>
                        ${{groupName}} (${{nodeIds.length}})
                    </div>
                `;
            }});

            document.getElementById('node-type-filters').innerHTML = nodeTypeHtml || 'No groups found';

            // Get all edge types
            const edgeTypes = getEdgeTypes();
            let edgeTypeHtml = '';

            edgeTypes.forEach(type => {{
                edgeTypeHtml += `
                    <div class="filter-option active" data-edge-type="${{type}}">
                        ${{type}}
                    </div>
                `;
            }});

            document.getElementById('edge-type-filters').innerHTML = edgeTypeHtml || 'No edge types found';

            // Add event listeners
            const filterOptions = document.querySelectorAll('.filter-option');
            filterOptions.forEach(option => {{
                option.addEventListener('click', function() {{
                    this.classList.toggle('active');
                }});
            }});

            // Apply filters button
            document.getElementById('apply-filters').addEventListener('click', applyFilters);

            // Reset filters button
            document.getElementById('reset-filters').addEventListener('click', resetFilters);
        }}

        function applyFilters() {{
            // Get selected node groups
            const activeNodeGroups = [];
            document.querySelectorAll('#node-type-filters .filter-option.active').forEach(option => {{
                activeNodeGroups.push(option.dataset.group);
            }});

            // Get selected edge types
            const activeEdgeTypes = [];
            document.querySelectorAll('#edge-type-filters .filter-option.active').forEach(option => {{
                activeEdgeTypes.push(option.dataset.edgeType);
            }});

            // Get minimum degree
            const minDegree = parseInt(document.getElementById('min-degree-slider').value);

            // Apply filters to nodes
            nodes.forEach(node => {{
                let visible = true;

                // Check group filter
                if (activeNodeGroups.length > 0 && node.group) {{
                    if (!activeNodeGroups.includes(node.group)) {{
                        visible = false;
                    }}
                }}

                // Check degree filter
                if (visible && minDegree > 1) {{
                    const degree = getNodeDegree(node.id);
                    if (degree < minDegree) {{
                        visible = false;
                    }}
                }}

                // Update node visibility
                nodes.update({{ id: node.id, hidden: !visible }});
            }});

            // Apply filters to edges
            edges.forEach(edge => {{
                let visible = true;

                // Check edge type filter
                if (activeEdgeTypes.length > 0 && edge.label) {{
                    if (!activeEdgeTypes.includes(edge.label)) {{
                        visible = false;
                    }}
                }}

                // Check if connected nodes are visible
                const fromNode = nodes.get(edge.from);
                const toNode = nodes.get(edge.to);

                if (fromNode.hidden || toNode.hidden) {{
                    visible = false;
                }}

                // Update edge visibility
                edges.update({{ id: edge.id, hidden: !visible }});
            }});
        }}

        function resetFilters() {{
            // Reset node visibility
            nodes.forEach(node => {{
                nodes.update({{ id: node.id, hidden: false }});
            }});

            // Reset edge visibility
            edges.forEach(edge => {{
                edges.update({{ id: edge.id, hidden: false }});
            }});

            // Reset filter UI
            document.querySelectorAll('.filter-option').forEach(option => {{
                option.classList.add('active');
            }});

            // Reset degree slider
            document.getElementById('min-degree-slider').value = 1;
            document.getElementById('min-degree-value').textContent = 1;
        }}

        function updateStatistics() {{
            statNodes.textContent = graphData.nodes.length;
            statEdges.textContent = graphData.edges.length;
            statAvgDegree.textContent = calculateAverageDegree().toFixed(2);

            const edgeTypes = getEdgeTypes();
            statEdgeTypes.textContent = edgeTypes.length;

            const nodeGroups = getNodeGroups();
            statNodeTypes.textContent = nodeGroups.size;

            // Update degree distribution chart
            updateDegreeDistributionChart();
        }}

        function showNodeDetails(node) {{
            const degree = getNodeDegree(node.id);
            const connectedNodes = getConnectedNodes(node.id);

            let html = `
                <h3>${{node.label || 'Node ' + node.id}}</h3>
                <div class="stat">
                    <span>ID:</span>
                    <span>${{node.id}}</span>
                </div>
                <div class="stat">
                    <span>Degree:</span>
                    <span>${{degree}}</span>
                </div>
            `;

            if (node.group) {{
                html += `
                    <div class="stat">
                        <span>Group:</span>
                        <span>${{node.group}}</span>
                    </div>
                `;
            }}

            html += '<h4>Connected Nodes:</h4>';

            if (connectedNodes.length > 0) {{
                // Limit to showing top 10 connections for performance
                const showNodes = connectedNodes.slice(0, 10);

                html += '<ul style="margin-top: 5px; padding-left: 15px; font-size: 12px;">';
                showNodes.forEach(connId => {{
                    const connNode = getNodeById(connId);
                    html += `<li>${{connNode.label || 'Node ' + connId}}</li>`;
                }});

                if (connectedNodes.length > 10) {{
                    html += `<li>... and ${{connectedNodes.length - 10}} more</li>`;
                }}

                html += '</ul>';
            }} else {{
                html += '<div style="margin-top: 5px; font-size: 12px;">No connections</div>';
            }}

            selectionContent.innerHTML = html;
        }}

        function showEdgeDetails(edge) {{
            const fromNode = getNodeById(edge.from);
            const toNode = getNodeById(edge.to);

            let html = `
                <h3>Edge Details</h3>
                <div class="stat">
                    <span>From:</span>
                    <span>${{fromNode.label || 'Node ' + edge.from}}</span>
                </div>
                <div class="stat">
                    <span>To:</span>
                    <span>${{toNode.label || 'Node ' + edge.to}}</span>
                </div>
            `;

            if (edge.label) {{
                html += `
                    <div class="stat">
                        <span>Relationship:</span>
                        <span>${{edge.label}}</span>
                    </div>
                `;
            }}

            selectionContent.innerHTML = html;
        }}

        function createLegend() {{
            // Create legend based on node groups
            const nodeGroups = getNodeGroups();

            if (nodeGroups.size === 0) {{
                legend.style.display = 'none';
                return;
            }}

            let html = '<div class="panel-header">Legend</div>';

            // Convert to array for sorting
            const groupsArray = Array.from(nodeGroups.entries());

            // Sort by size (largest first)
            groupsArray.sort((a, b) => b[1].length - a[1].length);

            // Take only the top 10 for readability
            const topGroups = groupsArray.slice(0, 10);

            topGroups.forEach(([groupName, nodeIds]) => {{
                const sampleNode = getNodeById(nodeIds[0]);
                let color = '#AAAAAA';

                if (sampleNode.color) {{
                    color = sampleNode.color;
                    if (typeof color === 'object') {{
                        color = color.background || color.border || '#AAAAAA';
                    }}
                }}

                html += `
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: ${{color}}"></div>
                        <span>${{groupName}} (${{nodeIds.length}})</span>
                    </div>
                `;
            }});

            if (groupsArray.length > 10) {{
                html += `<div class="legend-item">...and ${{groupsArray.length - 10}} more groups</div>`;
            }}

            legend.innerHTML = html;
        }}

        function initCharts() {{
            // Degree distribution chart with optimized settings
            const degreeCtx = document.getElementById('degree-distribution-chart').getContext('2d');
            window.degreeChart = new Chart(degreeCtx, {{
                type: 'bar',
                data: {{
                    labels: [],
                    datasets: [{{
                        label: 'Node Count',
                        data: [],
                        backgroundColor: '{("#7E57C2" if self.dark_mode else "#9C27B0")}',
                        borderColor: '{("#9575CD" if self.dark_mode else "#BA68C8")}',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {{
                        duration: 400  // Reduced from default
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: false  // Removed to save space
                            }},
                            ticks: {{
                                color: '{("#CCCCCC" if self.dark_mode else "#555555")}',
                                font: {{
                                    size: 10  // Smaller font
                                }}
                            }}
                        }},
                        x: {{
                            title: {{
                                display: false  // Removed to save space
                            }},
                            ticks: {{
                                color: '{("#CCCCCC" if self.dark_mode else "#555555")}',
                                font: {{
                                    size: 10  // Smaller font
                                }},
                                maxRotation: 0  // No rotation to save space
                            }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            display: false
                        }}
                    }}
                }}
            }});

            updateDegreeDistributionChart();
        }}

        function updateDegreeDistributionChart() {{
            try {{
                // Calculate degree distribution more efficiently
                const degrees = Array(graphData.nodes.length);
                for (let i = 0; i < graphData.nodes.length; i++) {{
                    degrees[i] = getNodeDegree(graphData.nodes[i].id);
                }}

                const maxDegree = Math.max(...degrees);

                // Optimize bin sizes for large graphs
                let binSize = 1;
                if (maxDegree > 30) {{
                    binSize = Math.ceil(maxDegree / 30);  // Aim for about 30 bins maximum
                }}

                // Count occurrences of each degree using bins
                const numBins = Math.ceil(maxDegree / binSize);
                const degreeFrequency = Array(numBins).fill(0);

                degrees.forEach(d => {{
                    const binIndex = Math.floor(d / binSize);
                    degreeFrequency[binIndex]++;
                }});

                // Create bin labels
                const labels = [];
                for (let i = 0; i < numBins; i++) {{
                    if (binSize === 1) {{
                        labels.push(i);
                    }} else {{
                        const start = i * binSize;
                        const end = Math.min((i + 1) * binSize - 1, maxDegree);
                        if (start === end) {{
                            labels.push(`${{start}}`);
                        }} else {{
                            labels.push(`${{start}}-${{end}}`);
                        }}
                    }}
                }}

                // Update chart
                window.degreeChart.data.labels = labels;
                window.degreeChart.data.datasets[0].data = degreeFrequency;
                window.degreeChart.update('none');  // No animation for performance
            }} catch (error) {{
                console.error('Error updating degree chart:', error);
            }}
        }}
    </script>
</body>
</html>
        """

        # Save the HTML file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Open the visualization in a browser
        if self.auto_open:
            webbrowser.open("file://" + str(output_path.absolute()))

        return str(output_path)

    def visualize_combined(
        self,
        ontology: Ontology,
        kg: KG,
        output_path: Path,
        title: str = "Combined Visualization",
    ):
        """
        Generate a visualization that combines both the ontology and knowledge graph.
        This shows how instances relate to their ontology classes.

        Parameters:
        -----------
        ontology : Ontology
            The ontology to visualize
        kg : KG
            The knowledge graph with instances
        output_path : Path
            Path to save the HTML visualization
        title : str
            Title of the visualization
        """
        # Styling constants - with reduced sizes
        CLASS_COLOR = "#4FC3F7"  # Light blue for classes
        DATATYPE_COLOR = "#BBDEFB"  # Lighter blue for datatypes
        PROPERTY_COLOR = "#FF9800"  # Orange for properties
        INSTANCE_COLOR = "#F06292"  # Pink for instances

        # Track nodes and edges
        added_nodes = set()
        nodes = []
        edges = []

        # Generate a unique ID for node names to avoid conflicts
        node_id_map = {}
        next_id = 1

        def get_node_id(name):
            if name not in node_id_map:
                nonlocal next_id
                node_id_map[name] = next_id
                next_id += 1
            return node_id_map[name]

        # Add a node with visual styling - optimized for performance
        def add_node(name, node_type="class", group=None, size=None):
            if name in added_nodes:
                return get_node_id(name)

            # Determine color and shape based on node type
            if node_type == "class":
                color = CLASS_COLOR
                shape = self.shapes["class"]
                font_size = 14  # Reduced from 16
                group = "ontology_class"
            elif node_type == "datatype":
                color = DATATYPE_COLOR
                shape = self.shapes["datatype"]
                font_size = 12  # Reduced from 14
                group = "ontology_datatype"
            elif node_type == "property":
                color = PROPERTY_COLOR
                shape = self.shapes["property"]
                font_size = 12  # Reduced from 14
                group = "ontology_property"
            else:  # instance
                color = INSTANCE_COLOR
                shape = self.shapes["individual"]
                font_size = 12  # Reduced from 14
                group = group or "instance"

            # Set node size - reduced from original
            if size is None:
                if node_type == "class":
                    size = 18  # Reduced from 25
                elif node_type == "datatype":
                    size = 12  # Reduced from 15
                elif node_type == "property":
                    size = 14  # Reduced from 18
                else:
                    size = 8  # Reduced from 12

            # Create node with optimized styling
            node = {
                "id": get_node_id(name),
                "label": name,
                "color": color,
                "shape": shape,
                "size": size,
                "group": group,
                "font": {
                    "size": font_size,
                    "color": "#FFFFFF" if self.dark_mode else "#000000",
                },
                "title": f"<div style='max-width: 250px;'><h3>{name}</h3><p>Type: {node_type.capitalize()}</p></div>",
            }

            nodes.append(node)
            added_nodes.add(name)
            return get_node_id(name)

        # 1. Add ontology classes
        for cls in ontology.classes.values():
            add_node(cls.name, "class")
            if cls.superclass:
                add_node(cls.superclass, "class")

                edges.append(
                    {
                        "from": get_node_id(cls.superclass),
                        "to": get_node_id(cls.name),
                        "label": "isA",
                        "font": {
                            "size": 8,  # Reduced from 10
                            "align": "middle",
                            "background": "#2E2E2E" if self.dark_mode else "#FFFFFF",
                        },
                        "arrows": {"to": {"enabled": True, "type": "arrow"}},
                        "dashes": [5, 5],  # Dashed line for isA relationships
                        "color": {"color": "#AAAAAA", "opacity": 0.7},
                    }
                )

        # 3. Add object properties - with optimized edge generation
        for obj_prop in ontology.object_properties.values():
            # Add the property itself as a node
            prop_id = add_node(obj_prop.name, "property")

            for dom in obj_prop.domain:
                dom_id = add_node(dom, "class")

                # Connect domain to property
                edges.append(
                    {
                        "from": dom_id,
                        "to": prop_id,
                        "label": "hasDomain",
                        "font": {
                            "size": 7,  # Reduced from 8
                            "align": "middle",
                            "background": "#2E2E2E" if self.dark_mode else "#FFFFFF",
                        },
                        "arrows": {"to": {"enabled": True, "type": "arrow"}},
                        "dashes": [2, 2],
                        "color": {"color": "#BBBBBB", "opacity": 0.5},
                    }
                )

            for rng in obj_prop.range:
                rng_id = add_node(rng, "class")

                # Connect property to range
                edges.append(
                    {
                        "from": prop_id,
                        "to": rng_id,
                        "label": "hasRange",
                        "font": {
                            "size": 7,  # Reduced from 8
                            "align": "middle",
                            "background": "#2E2E2E" if self.dark_mode else "#FFFFFF",
                        },
                        "arrows": {"to": {"enabled": True, "type": "arrow"}},
                        "dashes": [2, 2],
                        "color": {"color": "#BBBBBB", "opacity": 0.5},
                    }
                )

        # 4. Add data properties - with optimized edge generation
        for data_prop in ontology.data_properties.values():
            # Add the property itself as a node
            prop_id = add_node(data_prop.name, "property")

            for dom in data_prop.domain:
                dom_id = add_node(dom, "class")

                # Connect domain to property
                edges.append(
                    {
                        "from": dom_id,
                        "to": prop_id,
                        "label": "hasDomain",
                        "font": {
                            "size": 7,  # Reduced from 8
                            "align": "middle",
                            "background": "#2E2E2E" if self.dark_mode else "#FFFFFF",
                        },
                        "arrows": {"to": {"enabled": True, "type": "arrow"}},
                        "dashes": [2, 2],
                        "color": {"color": "#BBBBBB", "opacity": 0.5},
                    }
                )

            # Add datatype node
            datatype_name = data_prop.range
            datatype_id = add_node(datatype_name, "datatype")

            # Connect property to datatype
            edges.append(
                {
                    "from": prop_id,
                    "to": datatype_id,
                    "label": "hasRange",
                    "font": {
                        "size": 7,  # Reduced from 8
                        "align": "middle",
                        "background": "#2E2E2E" if self.dark_mode else "#FFFFFF",
                    },
                    "arrows": {"to": {"enabled": True, "type": "arrow"}},
                    "dashes": [2, 2],
                    "color": {"color": "#BBBBBB", "opacity": 0.5},
                }
            )

        # 5. Add KG instances and their relationships - optimized
        entity_to_types = {}  # e.g., "Harold" -> {"Person", ...}

        # First identify all 'isA' relationships to map entities to types
        for triplet in kg.triplets or []:
            if triplet.predicate.lower() == "isa":
                subj_name = triplet.subject
                obj_name = triplet.object
                entity_to_types.setdefault(subj_name, set()).add(obj_name)

        # For efficient processing, group triplets by subject
        subject_to_triplets = {}
        for triplet in kg.triplets or []:
            # Skip isA relationships (already processed)
            if triplet.predicate.lower() == "isa":
                continue

            if triplet.subject not in subject_to_triplets:
                subject_to_triplets[triplet.subject] = []
            subject_to_triplets[triplet.subject].append(triplet)

        # Add instances as nodes
        for subject, triplets in subject_to_triplets.items():
            # Add subject node if it doesn't exist
            if subject not in added_nodes:
                # Check if it has a type
                types = entity_to_types.get(subject, set())
                group = list(types)[0] if types else None
                add_node(subject, "instance", group)

            # Process all triplets for this subject
            for triplet in triplets:
                # Add object node if it doesn't exist
                obj_name = triplet.object
                if obj_name not in added_nodes:
                    # Check if it has a type
                    types = entity_to_types.get(obj_name, set())
                    group = list(types)[0] if types else None
                    add_node(obj_name, "instance", group)

                # Add the relationship edge
                source_id = get_node_id(subject)
                target_id = get_node_id(obj_name)

                edges.append(
                    {
                        "from": source_id,
                        "to": target_id,
                        "label": triplet.predicate,
                        "font": {
                            "size": 8,  # Reduced from 10
                            "align": "middle",
                            "background": "#2E2E2E" if self.dark_mode else "#FFFFFF",
                        },
                        "arrows": {"to": {"enabled": True, "type": "arrow"}},
                        "color": {
                            "color": "#E91E63",  # Pink for instance relationships
                            "opacity": 0.8,
                        },
                    }
                )

        # 6. Connect instances to their classes - batch process for efficiency
        for entity, types in entity_to_types.items():
            if entity in added_nodes:
                source_id = get_node_id(entity)

                for type_name in types:
                    if type_name in added_nodes:
                        target_id = get_node_id(type_name)

                        edges.append(
                            {
                                "from": source_id,
                                "to": target_id,
                                "label": "isA",
                                "font": {
                                    "size": 7,  # Reduced from 8
                                    "align": "middle",
                                    "background": "#2E2E2E" if self.dark_mode else "#FFFFFF",
                                },
                                "arrows": {"to": {"enabled": True, "type": "arrow"}},
                                "dashes": [3, 3],
                                "color": {
                                    "color": "#9C27B0",  # Purple for isA instance to class
                                    "opacity": 0.6,
                                },
                            }
                        )

        # Generate graph data with performance optimizations
        graph_data = self.generate_graph_data(nodes, edges)

        # Adjust physics settings for combined visualization
        graph_data["options"]["physics"]["forceAtlas2Based"]["gravitationalConstant"] = -70
        graph_data["options"]["physics"]["forceAtlas2Based"]["springLength"] = 150

        # Increase simulation iterations for better layout
        graph_data["options"]["physics"]["stabilization"]["iterations"] = 200

        # Create the visualization
        self._create_visualization(
            graph_data,
            output_path,
            title=title or f"Combined Ontology & KG: {len(nodes)} nodes, {len(edges)} relationships",
        )


class KnowledgeGraphViz:
    """
    High-level wrapper for visualizing knowledge graphs and ontologies with
    additional features like automatic layout algorithms, filtering, and analytics.
    """

    def __init__(
        self,
        dark_mode: bool = True,
        auto_open: bool = True,
        output_dir: str = "visualizations",
    ):
        self.dark_mode = dark_mode
        self.auto_open = auto_open
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Initialize visualizer with performance optimizations
        self.visualizer = AdvancedGraphVisualizer(
            dark_mode=dark_mode,
            auto_open=auto_open,
            render_mode="webgl",  # Force WebGL for performance
            initial_stabilization=True,  # Enable initial stabilization
            max_nodes_full_render=500,  # Threshold for simplified rendering
        )

    def visualize_ontology(self, ontology: Ontology, filename: str = "ontology_viz.html"):
        """
        Visualize an ontology with an interactive graph.

        Parameters:
        -----------
        ontology : Ontology
            The ontology to visualize
        filename : str
            Filename for the output HTML

        Returns:
        --------
        str
            Path to the generated visualization
        """
        output_path = self.output_dir / filename
        return self.visualizer.visualize_ontology(ontology, output_path)

    def visualize_kg(
        self,
        kg: KG,
        filename: str = "knowledge_graph_viz.html",
        ontology: Ontology | None = None,
    ):
        """
        Visualize a knowledge graph with an interactive graph.

        Parameters:
        -----------
        kg : KG
            The knowledge graph to visualize
        filename : str
            Filename for the output HTML
        ontology : Ontology | None
            Optional ontology to use for enhanced visualization

        Returns:
        --------
        str
            Path to the generated visualization
        """
        output_path = self.output_dir / filename
        return self.visualizer.visualize_kg(kg, output_path, ontology)

    def visualize_combined(
        self,
        ontology: Ontology,
        kg: KG,
        filename: str = "combined_viz.html",
        title: str = "Combined Ontology & Knowledge Graph",
    ):
        """
        Visualize both ontology and knowledge graph in a single view,
        showing the connections between instances and their classes.

        Parameters:
        -----------
        ontology : Ontology
            The ontology to visualize
        kg : KG
            The knowledge graph to visualize
        filename : str
            Filename for the output HTML
        title : str
            Title for the visualization

        Returns:
        --------
        str
            Path to the generated visualization
        """
        output_path = self.output_dir / filename
        return self.visualizer.visualize_combined(ontology, kg, output_path, title)

    def filter_and_visualize(
        self,
        kg: KG,
        filter_criteria: dict[str, Any] = None,
        filename: str = "filtered_kg_viz.html",
    ):
        """
        Filter a knowledge graph based on criteria and visualize the result.

        Parameters:
        -----------
        kg : KG
            The knowledge graph to filter and visualize
        filter_criteria : dict[str, Any]
            Criteria for filtering the KG. Example:
            {
                "predicates": ["hasName", "worksFor"],  # Only include these predicates
                "entity_types": ["Person", "Organization"],  # Only include these entity types
                "exclude_entities": ["Entity1", "Entity2"],  # Exclude specific entities
            }
        filename : str
            Filename for the output HTML

        Returns:
        --------
        str
            Path to the generated visualization
        """
        if filter_criteria is None:
            return self.visualize_kg(kg, filename)

        # Create a filtered copy of the KG
        filtered_kg = KG(triplets=[])

        # Keep track of entities to include based on their types
        entities_to_include = set()
        entity_types = {}

        # First pass: collect entity types
        for triplet in kg.triplets or []:
            if triplet.predicate.lower() == "isa":
                entity_types.setdefault(triplet.subject, []).append(triplet.object)

                # Check if this entity should be included based on type
                if "entity_types" not in filter_criteria or triplet.object in filter_criteria["entity_types"]:
                    entities_to_include.add(triplet.subject)

        # Second pass: add filtered triplets
        for triplet in kg.triplets or []:
            # Check if triplet should be excluded based on predicates
            if "predicates" in filter_criteria and triplet.predicate not in filter_criteria["predicates"]:
                continue

            # Check if entities should be excluded
            if "exclude_entities" in filter_criteria and (
                triplet.subject in filter_criteria["exclude_entities"]
                or triplet.object in filter_criteria["exclude_entities"]
            ):
                continue

            # For non-isA relationships, check if entities are in the inclusion list
            if triplet.predicate.lower() != "isa":
                if "entity_types" in filter_criteria:
                    if triplet.subject not in entities_to_include and triplet.object not in entities_to_include:
                        continue

            # Add the triplet to the filtered KG
            filtered_kg.triplets.append(triplet)

        # Visualize the filtered KG
        output_path = self.output_dir / filename
        return self.visualizer.visualize_kg(filtered_kg, output_path)

    def analyze_graph_metrics(self, kg: KG) -> dict[str, Any]:
        """
        Calculate and return various metrics for the knowledge graph.

        Parameters:
        -----------
        kg : KG
            The knowledge graph to analyze

        Returns:
        --------
        dict[str, Any]
            Dictionary with metrics including:
            - node_count: Total number of entities
            - edge_count: Total number of relationships
            - density: Graph density
            - degree_distribution: Count of connections per entity
            - predicate_distribution: Count of each predicate type
            - centrality: Most central entities
        """
        # Extract entities and relationships
        entities = set()
        predicates = {}
        entity_connections = {}

        for triplet in kg.triplets or []:
            entities.add(triplet.subject)
            entities.add(triplet.object)

            # Count predicates
            predicates[triplet.predicate] = predicates.get(triplet.predicate, 0) + 1

            # Count connections per entity (degree)
            entity_connections[triplet.subject] = entity_connections.get(triplet.subject, 0) + 1
            entity_connections[triplet.object] = entity_connections.get(triplet.object, 0) + 1

        # Calculate metrics
        node_count = len(entities)
        edge_count = len(kg.triplets) if kg.triplets else 0

        # Graph density (ratio of actual edges to possible edges)
        # In a directed graph, max edges = n(n-1)
        density = 0
        if node_count > 1:
            density = edge_count / (node_count * (node_count - 1))

        # Degree distribution
        degree_distribution = {}
        for entity, degree in entity_connections.items():
            degree_distribution[degree] = degree_distribution.get(degree, 0) + 1

        # Find central entities (by degree centrality)
        centrality = sorted(
            [(entity, degree) for entity, degree in entity_connections.items()],
            key=lambda x: x[1],
            reverse=True,
        )[:10]  # Top 10

        return {
            "node_count": node_count,
            "edge_count": edge_count,
            "density": density,
            "degree_distribution": degree_distribution,
            "predicate_distribution": predicates,
            "centrality": centrality,
        }

    def generate_report(
        self,
        ontology: Ontology = None,
        kg: KG = None,
        filename: str = "graph_analysis_report.html",
    ):
        """
        Generate a comprehensive HTML report with visualizations and metrics.

        Parameters:
        -----------
        ontology : Ontology, optional
            Ontology to include in the report
        kg : KG, optional
            Knowledge graph to include in the report
        filename : str
            Filename for the output HTML report

        Returns:
        --------
        str
            Path to the generated report
        """
        # Create visualization filenames
        ontology_viz = "ontology_report_viz.html" if ontology else None
        kg_viz = "kg_report_viz.html" if kg else None
        combined_viz = "combined_report_viz.html" if ontology and kg else None

        # Generate visualizations
        if ontology:
            self.visualize_ontology(ontology, ontology_viz)

        if kg:
            self.visualize_kg(kg, kg_viz)

        if ontology and kg:
            self.visualize_combined(ontology, kg, combined_viz)

        # Calculate metrics
        kg_metrics = None
        if kg:
            kg_metrics = self.analyze_graph_metrics(kg)

        # Generate HTML report
        report_path = self.output_dir / filename

        # Libraries for charts - using specific versions for better compatibility
        chart_js = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js"

        # Generate metric charts JS - optimized for performance
        degree_chart_js = ""
        predicate_chart_js = ""

        if kg_metrics:
            # Degree distribution chart - optimized with better binning
            degrees = list(kg_metrics["degree_distribution"].keys())
            degree_counts = list(kg_metrics["degree_distribution"].values())

            # Apply binning for large degree ranges
            if len(degrees) > 20:
                binned_degrees = {}
                max_degree = max(degrees)
                bin_size = max(1, max_degree // 20)  # Aim for ~20 bins

                for degree, count in zip(degrees, degree_counts):
                    bin_index = degree // bin_size
                    bin_label = f"{bin_index * bin_size}-{(bin_index + 1) * bin_size - 1}"
                    binned_degrees[bin_label] = binned_degrees.get(bin_label, 0) + count

                degrees = list(binned_degrees.keys())
                degree_counts = list(binned_degrees.values())

            degree_chart_js = f"""
            // Degree distribution chart with optimized settings
            const degreeCtx = document.getElementById('degree-chart').getContext('2d');
            new Chart(degreeCtx, {{
                type: 'bar',
                data: {{
                    labels: {str(degrees)},
                    datasets: [{{
                        label: 'Entity Count',
                        data: {str(degree_counts)},
                        backgroundColor: '{("#7E57C2" if self.dark_mode else "#9C27B0")}',
                        borderColor: '{("#9575CD" if self.dark_mode else "#BA68C8")}',
                        borderWidth: 1
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: {{
                        duration: 500  // Reduced for performance
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true,
                            title: {{
                                display: true,
                                text: 'Number of Entities',
                                color: '{("#DDDDDD" if self.dark_mode else "#333333")}'
                            }}
                        }},
                        x: {{
                            title: {{
                                display: true,
                                text: 'Degree (Number of Connections)',
                                color: '{("#DDDDDD" if self.dark_mode else "#333333")}'
                            }}
                        }}
                    }},
                    plugins: {{
                        legend: {{
                            display: false
                        }},
                        title: {{
                            display: true,
                            text: 'Degree Distribution',
                            color: '{("#FFFFFF" if self.dark_mode else "#000000")}'
                        }}
                    }}
                }}
            }});
            """

            # Predicate distribution chart - optimized for performance
            if kg_metrics["predicate_distribution"]:
                # Limit to top 10 predicates for readability
                top_predicates = sorted(
                    kg_metrics["predicate_distribution"].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]

                pred_labels = [p[0] for p in top_predicates]
                pred_counts = [p[1] for p in top_predicates]

                # Add "Other" category if there are more predicates
                if len(kg_metrics["predicate_distribution"]) > 10:
                    other_count = sum(kg_metrics["predicate_distribution"].values()) - sum(pred_counts)
                    pred_labels.append("Other")
                    pred_counts.append(other_count)

                predicate_chart_js = f"""
                // Predicate distribution chart with optimized settings
                const predicateCtx = document.getElementById('predicate-chart').getContext('2d');
                new Chart(predicateCtx, {{
                    type: 'doughnut',
                    data: {{
                        labels: {str(pred_labels)},
                        datasets: [{{
                            data: {str(pred_counts)},
                            backgroundColor: [
                                '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF',
                                '#FF9F40', '#8C9EFF', '#A5D6A7', '#FFD54F', '#81D4FA'
                            ]
                        }}]
                    }},
                    options: {{
                        responsive: true,
                        maintainAspectRatio: false,
                        animation: {{
                            duration: 500  // Reduced for performance
                        }},
                        plugins: {{
                            legend: {{
                                position: 'right',
                                labels: {{
                                    color: '{("#DDDDDD" if self.dark_mode else "#333333")}',
                                    boxWidth: 15,  // Smaller legend boxes
                                    padding: 8     // Less padding between items
                                }}
                            }},
                            title: {{
                                display: true,
                                text: 'Predicate Distribution',
                                color: '{("#FFFFFF" if self.dark_mode else "#000000")}'
                            }}
                        }}
                    }}
                }});
                """

        # Create HTML content - optimized UI
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Knowledge Graph Analysis Report</title>
    <script src="{chart_js}"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: Inter, system-ui, Avenir, Helvetica, Arial, sans-serif;
        }}

        body {{
            background-color: {("#1E1E1E" if self.dark_mode else "#FFFFFF")};
            color: {("#FFFFFF" if self.dark_mode else "#000000")};
            padding: 16px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 16px;
        }}

        .header {{
            text-align: center;
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid {("#555555" if self.dark_mode else "#DDDDDD")};
        }}

        .title {{
            font-size: 24px;
            margin-bottom: 8px;
        }}

        .subtitle {{
            font-size: 14px;
            color: {("#AAAAAA" if self.dark_mode else "#666666")};
        }}

        .section {{
            margin-bottom: 32px;
            background-color: {("#2D2D2D" if self.dark_mode else "#F5F5F5")};
            border-radius: 5px;
            padding: 16px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}

        .section-title {{
            font-size: 18px;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid {("#555555" if self.dark_mode else "#DDDDDD")};
        }}

        .metrics-container {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 16px;
        }}

        .metric-card {{
            background-color: {("#3D3D3D" if self.dark_mode else "#FFFFFF")};
            border-radius: 5px;
            padding: 12px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }}

        .metric-title {{
            font-size: 14px;
            margin-bottom: 8px;
            color: {("#BBBBBB" if self.dark_mode else "#666666")};
        }}

        .metric-value {{
            font-size: 20px;
            font-weight: bold;
        }}

        .charts-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 16px;
        }}

        .chart-card {{
            background-color: {("#3D3D3D" if self.dark_mode else "#FFFFFF")};
            border-radius: 5px;
            padding: 12px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
            height: 280px;
        }}

        .table-container {{
            overflow-x: auto;
            margin-top: 16px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 16px;
            font-size: 14px;
        }}

        th, td {{
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid {("#555555" if self.dark_mode else "#DDDDDD")};
        }}

        th {{
            background-color: {("#444444" if self.dark_mode else "#EEEEEE")};
            font-weight: bold;
        }}

        tr:hover {{
            background-color: {("#4D4D4D" if self.dark_mode else "#F0F0F0")};
        }}

        .viz-container {{
            margin-top: 16px;
        }}

        .viz-iframe {{
            width: 100%;
            height: 450px;
            border: none;
            border-radius: 5px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}

        .tab-container {{
            margin-top: 16px;
        }}

        .tabs {{
            display: flex;
            margin-bottom: 8px;
            flex-wrap: wrap;
        }}

        .tab {{
            padding: 8px 16px;
            background-color: {("#3D3D3D" if self.dark_mode else "#EEEEEE")};
            border-radius: 5px 5px 0 0;
            margin-right: 4px;
            margin-bottom: 4px; /* For wrapping */
            cursor: pointer;
            transition: background-color 0.2s;
            font-size: 14px;
        }}

        .tab.active {{
            background-color: {("#7E57C2" if self.dark_mode else "#9C27B0")};
            color: white;
        }}

        .tab-content {{
            display: none;
        }}

        .tab-content.active {{
            display: block;
        }}

        /* Responsive adjustments */
        @media (max-width: 768px) {{
            .metrics-container {{
                grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            }}

            .charts-container {{
                grid-template-columns: 1fr;
            }}

            .viz-iframe {{
                height: 350px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">Knowledge Graph Analysis Report</h1>
            <p class="subtitle">Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="section">
            <h2 class="section-title">Summary</h2>
            <div class="metrics-container">
                {
            f'''
                <div class="metric-card">
                    <div class="metric-title">Total Entities</div>
                    <div class="metric-value">{kg_metrics["node_count"]}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Total Relationships</div>
                    <div class="metric-value">{kg_metrics["edge_count"]}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Graph Density</div>
                    <div class="metric-value">{kg_metrics["density"]:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Predicate Types</div>
                    <div class="metric-value">{len(kg_metrics["predicate_distribution"])}</div>
                </div>
                '''
            if kg_metrics
            else '''
                <div class="metric-card">
                    <div class="metric-title">No Knowledge Graph Data</div>
                    <div class="metric-value">-</div>
                </div>
                '''
        }

                {
            f'''
                <div class="metric-card">
                    <div class="metric-title">Ontology Classes</div>
                    <div class="metric-value">{len(ontology.classes)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Object Properties</div>
                    <div class="metric-value">{len(ontology.object_properties)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Data Properties</div>
                    <div class="metric-value">{len(ontology.data_properties)}</div>
                </div>
                '''
            if ontology
            else '''
                <div class="metric-card">
                    <div class="metric-title">No Ontology Data</div>
                    <div class="metric-value">-</div>
                </div>
                '''
        }
            </div>
        </div>

        {
            f'''
        <div class="section">
            <h2 class="section-title">Knowledge Graph Analysis</h2>
            <div class="charts-container">
                <div class="chart-card">
                    <canvas id="degree-chart"></canvas>
                </div>
                <div class="chart-card">
                    <canvas id="predicate-chart"></canvas>
                </div>
            </div>

            <div class="table-container">
                <h3 style="margin: 16px 0 8px 0; font-size: 16px;">Most Central Entities (by Degree)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Entity</th>
                            <th>Connections</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f"<tr><td>{entity}</td><td>{degree}</td></tr>" for entity, degree in kg_metrics["centrality"])}
                    </tbody>
                </table>
            </div>

            <div class="table-container">
                <h3 style="margin: 16px 0 8px 0; font-size: 16px;">Predicate Distribution</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Predicate</th>
                            <th>Count</th>
                            <th>Percentage</th>
                        </tr>
                    </thead>
                    <tbody>
                        {"".join(f"<tr><td>{pred}</td><td>{count}</td><td>{count / kg_metrics['edge_count'] * 100:.1f}%</td></tr>" for pred, count in sorted(kg_metrics["predicate_distribution"].items(), key=lambda x: x[1], reverse=True))}
                    </tbody>
                </table>
            </div>
        </div>
        '''
            if kg_metrics
            else ""
        }

        <div class="section">
            <h2 class="section-title">Visualizations</h2>
            <div class="tab-container">
                <div class="tabs">
                    {'<div class="tab active" data-tab="ontology-tab">Ontology</div>' if ontology else ""}
                    {
            f'<div class="tab{" active" if not ontology else ""}" data-tab="kg-tab">Knowledge Graph</div>' if kg else ""
        }
                    {'<div class="tab" data-tab="combined-tab">Combined View</div>' if ontology and kg else ""}
                </div>

                {
            f'''
                <div class="tab-content active" id="ontology-tab">
                    <div class="viz-container">
                        <iframe src="{ontology_viz}" class="viz-iframe" title="Ontology Visualization"></iframe>
                    </div>
                </div>
                '''
            if ontology
            else ""
        }

                {
            f'''
                <div class="tab-content{" active" if not ontology else ""}" id="kg-tab">
                    <div class="viz-container">
                        <iframe src="{kg_viz}" class="viz-iframe" title="Knowledge Graph Visualization"></iframe>
                    </div>
                </div>
                '''
            if kg
            else ""
        }

                {
            f'''
                <div class="tab-content" id="combined-tab">
                    <div class="viz-container">
                        <iframe src="{combined_viz}" class="viz-iframe" title="Combined Visualization"></iframe>
                    </div>
                </div>
                '''
            if ontology and kg
            else ""
        }
            </div>
        </div>
    </div>

    <script>
        // Tab functionality with optimized event handling
        document.addEventListener('DOMContentLoaded', function() {{
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => {{
                tab.addEventListener('click', () => {{
                    // Remove active class from all tabs and content
                    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

                    // Add active class to clicked tab
                    tab.classList.add('active');

                    // Show corresponding content
                    const tabId = tab.getAttribute('data-tab');
                    const tabContent = document.getElementById(tabId);
                    if (tabContent) {{
                        tabContent.classList.add('active');
                    }}
                }});
            }});

            // Charts initialization with optimized loading
            setTimeout(() => {{
                {degree_chart_js}
                {predicate_chart_js}
            }}, 100);
        }});
    </script>
</body>
</html>
        """

        # Save the HTML file
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        # Open the report in a browser
        if self.auto_open:
            webbrowser.open("file://" + str(report_path.absolute()))

        return str(report_path)
