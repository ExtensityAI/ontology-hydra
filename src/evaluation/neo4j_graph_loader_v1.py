from neo4j import GraphDatabase
import json
from collections import defaultdict
import re

# === CONFIGURATION ===
JSON_PATH = "./graph.json"

class GraphLoader:
    def __init__(self, uri="bolt://localhost:7687", auth=("neo4j", "ontology")):
        self.driver = GraphDatabase.driver(uri, auth=auth)
        self.load_data()
        self.process_data()

    def load_data(self):
        """Load and initialize data structures"""
        with open(JSON_PATH, "r") as f:
            self.data = json.load(f)

        # Handle both triplets and direct nodes/edges formats
        if "triplets" in self.data:
            # Convert triplets to nodes and edges format
            self.nodes = set()
            self.edges = []

            # Extract unique nodes and create edges from triplets
            for triplet in self.data["triplets"]:
                self.nodes.add(triplet["subject"])
                self.nodes.add(triplet["object"])
                self.edges.append({
                    "source": triplet["subject"],
                    "label": triplet["predicate"],
                    "target": triplet["object"]
                })

            # Convert nodes to list of dictionaries
            self.nodes = [{"id": node} for node in self.nodes]
        else:
            # Direct nodes/edges format
            self.nodes = self.data["nodes"]
            self.edges = self.data["links"]

        self.node_ids = {node["id"] for node in self.nodes}

        # Initialize indexed data
        self.node_labels = defaultdict(set)
        self.node_props = defaultdict(dict)
        self.trait_links = []
        self.devarc_links = []
        self.culture_links = []
        self.relationship_links = []
        self.alias_map = defaultdict(set)

        # Categorize special nodes
        self.trait_ids = {n for n in self.node_ids if "trait" in n.lower()
                         and not any(k in n.lower() for k in ["event", "setting"])}
        self.culture_ids = {n for n in self.node_ids
                           if any(k in n.lower() for k in ["cultural", "british"])}
        self.devarc_ids = {n for n in self.node_ids
                          if "arc" in n.lower() or "development" in n.lower()}

    def process_data(self):
        """Process edges and categorize relationships"""
        # Initialize trait subtypes
        self.trait_subtypes = defaultdict(str)

        for edge in self.edges:
            source, label, target = edge["source"], edge["label"], edge["target"]

            if label == "isA":
                # Handle hierarchical labels
                self.node_labels[source].add(target.replace(" ", "_"))
                # Also add any parent labels transitively
                parent_labels = self.node_labels.get(target, set())
                self.node_labels[source].update(parent_labels)
            elif label.startswith("hasCharacterTrait") or label.endswith("Trait"):
                # Extract trait subtype if present
                trait_type = None
                if "Psychological" in label:
                    trait_type = "PsychologicalTrait"
                elif "Physical" in label:
                    trait_type = "PhysicalTrait"
                elif "Character" in label:
                    trait_type = "CharacterTrait"

                self.trait_links.append((source, target, "HAS_TRAIT"))
                if trait_type:
                    self.trait_subtypes[target] = trait_type
            elif label == "hasDevelopmentArc":
                self.devarc_links.append((source, target))
            elif label in ["hasCulturalBackground", "hasCulturalAttribute"]:
                self.culture_links.append((source, target))
            elif label == "hasAlias" or label == "sameAs":
                self.alias_map[source].add(target)
                self.alias_map[target].add(source)
            else:
                self.relationship_links.append((source, label, target))

        # Set default labels
        for nid in self.node_ids:
            if not self.node_labels[nid]:
                self.node_labels[nid].add("Entity")

    @staticmethod
    def clean_label(label):
        return re.sub(r"[^a-zA-Z0-9_]", "_", label)

    @staticmethod
    def format_name(id_str):
        s = id_str.replace("_", " ").replace("-", " ")
        s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
        return s.strip().title()

    def create_nodes(self, tx):
        """Create all nodes with their labels and properties"""
        for node in self.nodes:
            node_id = node["id"]
            labels = ":".join(self.clean_label(lbl) for lbl in self.node_labels[node_id])
            props = {
                "id": node_id,
                "name": self.format_name(node_id)
            }
            props.update({k: v for k, v in node.items() if k != "id"})
            query = f"MERGE (n:{labels} {{id: $id}}) SET n += $props"
            tx.run(query, id=node_id, props=props)

    def create_relationships(self, tx):
        """Create general relationships between nodes"""
        seen = set()
        for source, label, target in self.relationship_links:
            key = (source, label, target)
            if key not in seen:
                seen.add(key)
                tx.run(f"""
                MATCH (a {{id: $source}}), (b {{id: $target}})
                MERGE (a)-[r:`{self.clean_label(label)}`]->(b)
                """, source=source, target=target)

    def create_special_nodes(self, tx):
        """Create and link trait, culture, and development arc nodes"""
        # Traits with subtypes
        for source, trait, rel_type in self.trait_links:
            # Get trait subtype if it exists
            trait_subtype = self.trait_subtypes.get(trait, "Trait")
            labels = f"Trait:{trait_subtype}" if trait_subtype != "Trait" else "Trait"

            tx.run(f"""
            MERGE (trait:{labels} {{id: $trait}})
            ON CREATE SET trait.name = $name
            WITH trait
            MATCH (n {{id: $source}})
            MERGE (n)-[:{rel_type}]->(trait)
            """, trait=trait, name=self.format_name(trait), source=source)

        # Development Arcs
        for source, arc in self.devarc_links:
            tx.run("""
            MERGE (arc:DevelopmentArc {id: $arc, name: $name})
            WITH arc
            MATCH (n {id: $source})
            MERGE (n)-[:HAS_DEVELOPMENT_ARC]->(arc)
            """, arc=arc, name=self.format_name(arc), source=source)

        # Culture
        for source, culture in self.culture_links:
            tx.run("""
            MERGE (c:Culture {id: $culture, name: $name})
            WITH c
            MATCH (n {id: $source})
            MERGE (n)-[:HAS_CULTURE]->(c)
            """, culture=culture, name=self.format_name(culture), source=source)

    def create_alias_relationships(self, tx):
        """Create bidirectional alias relationships with properties"""
        # Group aliases into connected components
        alias_groups = []
        processed = set()

        def get_all_aliases(node, aliases=None):
            if aliases is None:
                aliases = set()
            aliases.add(node)
            for alias in self.alias_map[node]:
                if alias not in aliases:
                    get_all_aliases(alias, aliases)
            return aliases

        # Build alias groups
        for node in self.alias_map:
            if node not in processed:
                group = get_all_aliases(node)
                alias_groups.append(group)
                processed.update(group)

        # Create relationships and store alias information
        for group in alias_groups:
            primary = min(group)  # Use shortest name as primary
            for alias in group:
                if alias != primary:
                    tx.run("""
                    MATCH (a {id: $source}), (b {id: $alias})
                    MERGE (a)-[r:ALIAS_OF]-(b)
                    SET a.aliases = $all_aliases,
                        b.aliases = $all_aliases,
                        a.is_primary = $is_primary_a,
                        b.is_primary = $is_primary_b
                    """,
                    source=primary,
                    alias=alias,
                    all_aliases=list(group - {primary}),
                    is_primary_a=True,
                    is_primary_b=False)

    def load_graph(self):
        """Main method to load the entire graph"""
        with self.driver.session() as session:
            # Create base structure
            session.execute_write(self.create_nodes)
            session.execute_write(self.create_relationships)
            session.execute_write(self.create_special_nodes)
            session.execute_write(self.create_alias_relationships)

            # Additional verification steps
            session.execute_write(self.verify_special_nodes)

        print("Graph loaded and verified successfully!")
        self.driver.close()

    def verify_special_nodes(self, tx):
        """Verify and patch any missing special nodes and labels"""
        # Verify traits with subtypes
        for trait in self.trait_ids:
            trait_subtype = self.trait_subtypes.get(trait, "Trait")
            labels = f"Trait:{trait_subtype}" if trait_subtype != "Trait" else "Trait"

            tx.run(f"""
            MATCH (t {{id: $id}})
            SET t:{labels}
            SET t.name = $name
            """, id=trait, name=self.format_name(trait))

        # Verify culture
        for culture in self.culture_ids:
            tx.run("""
            MATCH (c {id: $id})
            SET c:Culture
            SET c.name = $name
            """, id=culture, name=self.format_name(culture))

        # Verify development arcs
        for arc in self.devarc_ids:
            tx.run("""
            MATCH (a {id: $id})
            SET a:DevelopmentArc
            SET a.name = $name
            """, id=arc, name=self.format_name(arc))

        # Patch isA labels - ensure all labels are properly set
        for node_id, labels in self.node_labels.items():
            for label in labels:
                clean_label = self.clean_label(label)
                tx.run(f"""
                MATCH (n {{id: $id}})
                WHERE NOT n:{clean_label}
                SET n:{clean_label}
                """, id=node_id)

if __name__ == "__main__":
    loader = GraphLoader()
    loader.load_graph()