from neo4j import GraphDatabase
from typing import Dict, List
import json
import csv
import os

class Neo4jKGLoader:
    def __init__(self, uri: str, user: str, password: str):
        """Initialize the Neo4j connection."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def clear_database(self):
        """Clear all nodes and relationships in the database."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_constraints(self):
        """Create constraints for better performance."""
        with self.driver.session() as session:
            # Create base constraint for Entity
            try:
                session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE")
            except:
                # Handle case for older Neo4j versions
                pass

    def convert_json_to_csv(self, kg_file: str):
        """Convert JSON knowledge graph to CSV files for efficient Neo4j loading."""
        # Read the JSON file
        with open(kg_file, 'r') as f:
            data = json.load(f)

        triplets = data["triplets"]

        # Collect unique node names
        nodes = set()
        for t in triplets:
            nodes.add(t["subject"])
            nodes.add(t["object"])

        # Extract identifier from the kg_file path
        # Find the part after "eval/runs/" and before "/kg.json"
        path_parts = kg_file.split("/")
        try:
            runs_index = path_parts.index("runs")
            identifier_parts = path_parts[runs_index + 1:-1]  # Skip "runs" and "kg.json"
            # Filter out "topics" and replace spaces with underscores
            filtered_parts = []
            for part in identifier_parts:
                if part != "topics":
                    # Replace spaces with underscores
                    filtered_parts.append(part.replace(" ", "_"))
            identifier = "_".join(filtered_parts)  # Replace slashes with underscores
        except (ValueError, IndexError):
            # Fallback if path structure is different
            identifier = "kg_data"

        # Neo4j import folder path
        neo4j_import_dir = "/Users/ryang/Library/Application Support/Neo4j Desktop/Application/relate-data/dbmss/dbms-8ff6e63e-4586-411a-a8ba-f42cb734d84b/import"

        # Write nodes.csv with identifier
        nodes_file = os.path.join(neo4j_import_dir, f"nodes_{identifier}.csv")
        with open(nodes_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name"])  # header
            for node in sorted(nodes):
                writer.writerow([node])

        # Write relationships.csv with identifier
        relationships_file = os.path.join(neo4j_import_dir, f"relationships_{identifier}.csv")
        with open(relationships_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["subject", "predicate", "object"])  # header
            for t in triplets:
                writer.writerow([t["subject"], t["predicate"], t["object"]])

        print(f"CSV files created: {nodes_file}, {relationships_file}")
        return nodes_file, relationships_file

    def load_nodes_from_csv(self, csv_file: str):
        """Load nodes from CSV file using Neo4j's LOAD CSV."""
        with self.driver.session() as session:
            # Get just the filename since it's in the Neo4j import folder
            filename = os.path.basename(csv_file)
            file_url = f"file:///{filename}"

            # Load nodes
            result = session.run(f"""
            LOAD CSV WITH HEADERS FROM '{file_url}' AS row
            MERGE (n:Entity {{name: row.name}})
            RETURN count(*) as nodes_created
            """)
            count = result.single()["nodes_created"]
            print(f"Loaded {count} nodes from CSV")

    def load_relationships_from_csv(self, csv_file: str):
        """Load relationships from CSV file using Neo4j's LOAD CSV and APOC."""
        with self.driver.session() as session:
            # Get just the filename since it's in the Neo4j import folder
            filename = os.path.basename(csv_file)
            file_url = f"file:///{filename}"

            # Load relationships using APOC
            result = session.run(f"""
            LOAD CSV WITH HEADERS FROM '{file_url}' AS row
            MATCH (a:Entity {{name: row.subject}}), (b:Entity {{name: row.object}})
            CALL apoc.create.relationship(a, row.predicate, {{}}, b) YIELD rel
            RETURN count(*) as relationships_created
            """)
            count = result.single()["relationships_created"]
            print(f"Loaded {count} relationships from CSV")

    def load_knowledge_graph(self, kg_file: str):
        """Load the entire knowledge graph from a JSON file using CSV conversion."""
        # Clear existing data and create constraints
        self.clear_database()
        self.create_constraints()

        # Convert JSON to CSV - always use the Neo4j import directory
        nodes_file, relationships_file = self.convert_json_to_csv(kg_file)

        # Load nodes from CSV
        self.load_nodes_from_csv(nodes_file)

        # Load relationships from CSV
        self.load_relationships_from_csv(relationships_file)

        print("Knowledge graph successfully loaded into Neo4j using CSV method!")

def main():
    # Neo4j connection parameters
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "ontology"

    # File path
    KG_FILE = "/Users/ryang/Work/ExtensityAI/research-ontology/eval/runs/20250711_FM1pXA/kg_only/topics/Biomedical Engineer/kg.json"

    try:
        # Initialize loader
        loader = Neo4jKGLoader(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        # Load the knowledge graph
        loader.load_knowledge_graph(KG_FILE)

        # Close the connection
        loader.close()

        print("Knowledge graph successfully loaded into Neo4j!")

    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()