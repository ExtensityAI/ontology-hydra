import pandas as pd
from neo4j import GraphDatabase
from typing import Optional
from pydantic import BaseModel, Field
from symai import Expression
from symai.strategy import contract
from symai.models import LLMDataModel
import json
import re

# === CONFIGURATION ===
JSON_PATH = "./generated_questions.json"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "ontology"

class Neo4jQueryInput(LLMDataModel):
    """Input for Neo4j query generation"""
    question: str = Field(description="The natural language question to convert to Cypher")
    schema: str = Field(description="The schema of the Neo4j database")

class Neo4jQueryOutput(LLMDataModel):
    """Output containing the generated Cypher query"""
    query: str = Field(description="The generated Cypher query")

@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=False,
    remedy_retry_params=dict(
        tries=3,
        delay=0.5,
        max_delay=10,
        jitter=0.1,
        backoff=2,
        graceful=False
    )
)
class QuestionToCypherConverter(Expression):
    def __init__(
        self,
        driver: GraphDatabase.driver,
        seed: Optional[int] = 42,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.data_model = Neo4jQueryOutput
        self.driver = driver

    def get_schema(self) -> str:
        """Query Neo4j for complete schema information.

        Returns:
            str: Formatted schema string with comprehensive schema details including:
                - Node labels and their counts
                - Relationship types and their counts
                - Properties and their usage across node types
                - Common relationship patterns
        """
        with self.driver.session() as session:
            # Get node labels and their counts
            labels_result = session.run("""
                MATCH (n)
                WITH labels(n) as labels
                UNWIND labels as label
                WITH label, count(*) as count
                RETURN label, count
                ORDER BY label
            """)
            node_labels = [(record["label"], record["count"]) for record in labels_result]

            # Get relationship types and their counts
            rels_result = session.run("""
                MATCH ()-[r]->()
                WITH type(r) as relType, count(*) as count
                RETURN relType, count
                ORDER BY relType
            """)
            relationship_types = [(record["relType"], record["count"]) for record in rels_result]

            # Get property keys and their usage across node types
            props_result = session.run("""
                MATCH (n)
                WITH labels(n) as nodeLabels, keys(n) as props
                UNWIND nodeLabels as label
                UNWIND props as prop
                WITH label, prop, count(*) as count
                RETURN label, prop, count
                ORDER BY label, prop
            """)
            property_usage = [(record["label"], record["prop"], record["count"]) for record in props_result]

            # Get common relationship patterns
            patterns_result = session.run("""
                MATCH (a)-[r]->(b)
                WITH labels(a)[0] as fromType, type(r) as relType, labels(b)[0] as toType,
                     count(*) as count
                RETURN fromType, relType, toType, count
                ORDER BY count DESC
            """)
            patterns = [(record["fromType"], record["relType"], record["toType"], record["count"])
                       for record in patterns_result]

            # Format the schema string
            schema = "Complete Neo4j Schema:\n\n"

            schema += "1. Node Labels and Counts:\n"
            for label, count in node_labels:
                schema += f"   - {label}: {count} nodes\n"

            schema += "\n2. Relationship Types and Counts:\n"
            for rel_type, count in relationship_types:
                schema += f"   - :{rel_type}: {count} relationships\n"

            schema += "\n3. Property Keys by Node Label:\n"
            current_label = None
            for label, prop, count in property_usage:
                if label != current_label:
                    schema += f"\n   {label}:\n"
                    current_label = label
                schema += f"   - {prop} (used in {count} nodes)\n"

            schema += "\n4. Common Relationship Patterns:\n"
            for from_type, rel_type, to_type, count in patterns:
                schema += f"   - ({from_type})-[:{rel_type}]->({to_type}): {count} occurrences\n"

            return schema

    def forward(self, input: Neo4jQueryInput, **kwargs) -> Neo4jQueryOutput:
        if self.contract_result is None:
            return Neo4jQueryOutput(query="")
        return self.contract_result

    def pre(self, input: Neo4jQueryInput) -> bool:
        if not isinstance(input, Neo4jQueryInput):
            raise ValueError("Input must be a Neo4jQueryInput instance!")
        if not input.question.strip():
            raise ValueError("Question cannot be empty!")
        return True

    def post(self, output: Neo4jQueryOutput) -> bool:
        # Basic validation checks
        if not output.query.strip():
            raise ValueError("Generated Cypher query cannot be empty")
        if not output.query.upper().startswith("MATCH"):
            raise ValueError("Generated Cypher query must start with MATCH")

        # Extract schema elements from the query
        q_labels, q_rels, q_props = self._extract_cypher_elements(output.query)

        # Get actual schema elements from Neo4j
        with self.driver.session() as session:
            labels, rels, props = session.read_transaction(self._get_schema)

        # Validate against schema
        invalid_labels = q_labels - labels
        invalid_rels = q_rels - rels
        invalid_props = q_props - props

        if invalid_labels:
            raise ValueError(f"Query uses non-existent labels: {invalid_labels}")
        if invalid_rels:
            raise ValueError(f"Query uses non-existent relationships: {invalid_rels}")
        if invalid_props:
            raise ValueError(f"Query uses non-existent properties: {invalid_props}")

        return True

    def _get_schema(self, tx):
        """Get schema elements from Neo4j."""
        labels = {record["label"] for record in tx.run("CALL db.labels()")}
        rels = {record["relationshipType"] for record in tx.run("CALL db.relationshipTypes()")}
        props = {record["propertyKey"] for record in tx.run("CALL db.propertyKeys()")}
        return labels, rels, props

    def _extract_cypher_elements(self, query):
        """Extract Cypher elements using regex."""
        labels = set(re.findall(r":([A-Z][a-zA-Z0-9_]*)", query))
        relationships = set(re.findall(r":([A-Z_]+)", query))
        properties = set(re.findall(r"\.\s*([a-zA-Z0-9_]+)", query))
        return labels, relationships, properties

    @property
    def prompt(self) -> str:
        return """[[Neo4j Cypher Query Generation]]
Convert the given natural language question into a Neo4j Cypher query using ONLY the schema elements provided. No other elements exist or can be used.

IMPORTANT: No other schema elements exist in the database beyond those listed above. You cannot use:
- Any node labels not listed
- Any relationship types not listed
- Any properties not listed
- Any additional attributes or metadata

Query Guidelines:
1. Start with MATCH clause
2. Use node labels and relationships exactly as shown in the schema
3. Use only the properties listed in the schema
4. Connect patterns using relationships to avoid cartesian products
5. Include RETURN clause with specific fields
6. If the question cannot be answered with the available schema, explain why

Example valid query format:
MATCH (node:Label1)-[:RELATIONSHIP_TYPE]->(other:Label2)
RETURN node.property1, other.property2

Example invalid query format:
MATCH (node:NonExistentLabel {nonExistentProperty: 'value'})  // WRONG - uses elements not in schema
RETURN node.nonExistentProperty

Remember: Only use the exact node labels, relationship types, and properties listed in the schema above. Any other elements will fail as they do not exist in the database.

If the question cannot be answered using only the provided schema elements, explain why the query cannot be constructed."""

# === NEO4J EXECUTION ===
def run_neo4j_query(driver, query):
    with driver.session() as session:
        result = session.run(query)
        return [record.data() for record in result]
# === MAIN PIPELINE ===
def main():
    # Load JSON data
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # Initialize Neo4j connection
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    successful_queries = 0
    queries_with_results = 0
    total_queries = 0
    converter = QuestionToCypherConverter(driver=driver)
    schema = converter.get_schema()
    print(schema)

    # Process each question from the JSON structure
    for article in data['data']:
        for paragraph in article['paragraphs']:
            total_queries += len(paragraph['qas'])
            for qa in paragraph['qas']:
                question = qa['question']
                try:
                    # Convert question to query using contract
                    input_data = Neo4jQueryInput(
                        question=question,
                        schema=schema
                    )
                    result = converter(input=input_data)
                    query = result.query

                    if not query.strip():
                        print(f"Skipping: '{question}' (No Cypher query generated)")
                        continue

                    print(f"Executing query for: '{question}'")
                    print(f"Generated Cypher: {query}")
                    results = run_neo4j_query(driver, query)
                    print(f"Results: {results}\n")
                    successful_queries += 1
                    if len(results) > 0:  # Check if results list is not empty
                        queries_with_results += 1

                except Exception as e:
                    print(f"Error processing question '{question}': {e}\n")

    print(f"Successfully processed {successful_queries} out of {total_queries} queries")
    print(f"Queries that returned results: {queries_with_results}")
    driver.close()

if __name__ == "__main__":
    main()