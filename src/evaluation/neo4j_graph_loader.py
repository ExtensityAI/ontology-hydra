from neo4j import GraphDatabase
from typing import Dict, List
import json

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

    def load_triplet(self, subject: str, predicate: str, object: str):
        """Create a single triplet in the graph with proper labels and relationship types.

        For 'isA' relationships, the object becomes an additional label for the subject node.
        For all other relationships, creates a relationship of type predicate between nodes.
        """
        # Clean up predicate and object for use as Neo4j identifiers
        rel_type = predicate.upper().replace(' ', '_')
        label = object.replace(' ', '_')

        if predicate.lower() == "isa":
            # For isA relationships, add the object as a label to the subject node
            query = f"""
            MERGE (s:Entity {{name: $subject}})
            SET s:{label}
            MERGE (o:Entity {{name: $object}})
            MERGE (s)-[r:`{rel_type}`]->(o)
            """
        else:
            query = f"""
            MERGE (s:Entity {{name: $subject}})
            MERGE (o:Entity {{name: $object}})
            MERGE (s)-[r:`{rel_type}`]->(o)
            """

        with self.driver.session() as session:
            session.run(query, subject=subject, object=object)

    def load_knowledge_graph(self, kg_file: str):
        """Load the entire knowledge graph from a JSON file."""
        # Read the JSON file
        with open(kg_file, 'r') as f:
            kg_data = json.load(f)

        # Clear existing data and create constraints
        self.clear_database()
        self.create_constraints()

        # Process all triplets
        total_triplets = len(kg_data['triplets'])
        for i, triplet in enumerate(kg_data['triplets'], 1):
            self.load_triplet(
                triplet['subject'],
                triplet['predicate'],
                triplet['object']
            )
            if i % 100 == 0:
                print(f"Processed {i}/{total_triplets} triplets")

        print(f"Successfully loaded {total_triplets} triplets into Neo4j")

def main():
    # Neo4j connection parameters
    NEO4J_URI = "neo4j://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "ontology"

    # File path
    KG_FILE = "../../eval/runs/20250702_D7Cxag/biomed/topics/Biomedical Engineering/kg.json"

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