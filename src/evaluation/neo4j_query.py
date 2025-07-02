import pandas as pd
from neo4j import GraphDatabase
from typing import Optional, List
from pydantic import BaseModel, Field
from symai import Expression
from symai.strategy import contract
from symai.models import LLMDataModel
import json
import re
from math import ceil
from pathlib import Path
from collections import defaultdict

# === CONFIGURATION ===
JSON_PATH = Path(__file__).parent.parent.parent / "MedExQA" / "test" / "biomedical_engineer_test.json"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "ontology"
BATCH_SIZE = 5  # Number of questions to process in each batch
NUM_ITERATIONS = 10  # Number of times to run the evaluation

class Neo4jQueryInput(LLMDataModel):
    """Input for Neo4j query generation"""
    questions: List[str] = Field(description="List of natural language questions to convert to Cypher")
    options: List[List[str]] = Field(description="List of possible answers for each question")
    answers: List[str] = Field(description="List of expected answers for each question")
    schema: str = Field(description="The schema of the Neo4j database")

class Neo4jQueryOutput(LLMDataModel):
    """Output containing the generated Cypher queries"""
    queries: List[str] = Field(description="List of generated Cypher queries")

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
class BatchQuestionToCypherConverter(Expression):
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
            return Neo4jQueryOutput(queries=[""] * len(input.questions))
        return self.contract_result

    def pre(self, input: Neo4jQueryInput) -> bool:
        if not isinstance(input, Neo4jQueryInput):
            raise ValueError("Input must be a Neo4jQueryInput instance!")
        if not input.questions or len(input.questions) == 0:
            raise ValueError("Questions list cannot be empty!")
        if len(input.questions) != len(input.options) or len(input.questions) != len(input.answers):
            raise ValueError("Number of questions, options, and answers must match!")
        return True

    def post(self, output: Neo4jQueryOutput) -> bool:
        if not output.queries or len(output.queries) == 0:
            raise ValueError("Generated Cypher queries cannot be empty")

        # Validate that number of output queries matches number of input questions
        if len(output.queries) != len(self._input.questions):
            raise ValueError(f"Number of output queries ({len(output.queries)}) does not match number of input questions ({len(self._input.questions)})")

        try:
            with self.driver.session() as session:
                labels, rels, props = session.read_transaction(self._get_schema)
        except Exception as e:
            raise ValueError(f"Failed to get schema from Neo4j: {e}")

        for i, query in enumerate(output.queries):
            if not query.strip():
                raise ValueError(f"Query {i+1} cannot be empty")
            if not query.upper().startswith("MATCH"):
                raise ValueError(f"Query {i+1} must start with MATCH: {query[:50]}...")

            # Extract and validate schema elements
            q_labels, q_rels, q_props = self._extract_cypher_elements(query)

            invalid_labels = q_labels - labels
            invalid_rels = q_rels - rels
            invalid_props = q_props - props

            if invalid_labels:
                raise ValueError(f"Query {i+1} uses non-existent labels: {invalid_labels}")
            if invalid_rels:
                raise ValueError(f"Query {i+1} uses non-existent relationships: {invalid_rels}")
            if invalid_props:
                raise ValueError(f"Query {i+1} uses non-existent properties: {invalid_props}")

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
        return """[[Neo4j Cypher Query Generation - Batch Processing]]
Convert the given list of natural language questions into Neo4j Cypher queries using ONLY the schema elements provided. No other elements exist or can be used.

For each question in the batch, generate a corresponding Cypher query following these guidelines:

IMPORTANT GUIDELINES:
1. For "what is used in X" type questions:
   - Look for tools/devices/materials CONNECTED to X through relationships
   - Search for techniques/procedures that are part of X
   - Consider relationships between medical devices and relevant biological tissues
   - Use OPTIONAL MATCH for different types of relationships to get comprehensive results

2. For questions about procedures/techniques:
   - Look for relationships between procedures and their components
   - Consider both direct and indirect relationships (through intermediate nodes)
   - Include relevant properties that describe usage or application

3. Query Structure Best Practices:
   - Start with main concept MATCH clause
   - Use multiple OPTIONAL MATCH clauses to gather related information
   - Connect patterns using relationships to avoid cartesian products
   - Use WHERE clauses to filter relevant results
   - Return meaningful properties and relationships, not just node IDs
   - Use aggregation (COLLECT, COUNT) when appropriate to group related items

Example for "What is used in dermatological surgery?":
MATCH (p:MedicalProcedure {name: 'DermatologicalSurgery'})
OPTIONAL MATCH (p)-[r1]-(d:MedicalDevice)
OPTIONAL MATCH (p)-[r2]-(t:Technique)
OPTIONAL MATCH (p)-[r3]-(m:Material)
RETURN
    COLLECT(DISTINCT d.name) as devices,
    COLLECT(DISTINCT t.name) as techniques,
    COLLECT(DISTINCT m.name) as materials

Remember:
- Generate one query for each question in the input batch
- Only use the exact node labels, relationship types, and properties listed in the schema
- Any other elements will fail as they do not exist in the database
- Maintain consistent query structure across the batch for similar question types"""

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

    converter = BatchQuestionToCypherConverter(driver=driver)
    kg_file = "/Users/ryang/Work/ExtensityAI/research-ontology/eval/runs/20250530_vL04rg/biomed/topics/Biomedical Engineering/kg.json"
    with open(kg_file, "r") as f:
        kg_data = json.load(f)
    schema = json.dumps(kg_data)

    print(f"Loaded schema from: {kg_file}")
    print(f"Schema size: {len(schema)} characters")
    print(f"Schema preview: {schema[:200]}...")

    # Collect all questions, options, and answers
    all_questions = []
    all_options = []
    all_answers = []

    for article in data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                all_questions.append(qa['question'])
                all_options.append([ans['text'] for ans in qa['all_answers']])
                all_answers.append(qa['answers'][0]['text'] if qa['answers'] else "")

    total_queries = len(all_questions)

    # Initialize results storage
    results_data = []
    query_stats = defaultdict(lambda: {
        'successful': False,
        'returned_results': False,
        'correct': False,
        'iterations': []
    })

    print(f"Running evaluation {NUM_ITERATIONS} times for {total_queries} queries...")

    # Run evaluation N times
    for iteration in range(NUM_ITERATIONS):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{NUM_ITERATIONS}")
        print(f"{'='*60}")

        # Process questions in batches
        num_batches = ceil(len(all_questions) / BATCH_SIZE)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min((batch_idx + 1) * BATCH_SIZE, len(all_questions))

            print(f"\nProcessing batch {batch_idx + 1}/{num_batches}")

            batch_questions = all_questions[start_idx:end_idx]
            batch_options = all_options[start_idx:end_idx]
            batch_answers = all_answers[start_idx:end_idx]

            try:
                # Convert questions to queries using contract
                input_data = Neo4jQueryInput(
                    questions=batch_questions,
                    options=batch_options,
                    answers=batch_answers,
                    schema=schema
                )

                print(f"  Converting {len(batch_questions)} questions to Cypher queries...")
                result = converter(input=input_data)
                queries = result.queries
                print(f"  Successfully generated {len(queries)} queries")

                # Process each query in the batch
                for i, (question, options, answer, query) in enumerate(zip(batch_questions, batch_options, batch_answers, queries)):
                    query_idx = start_idx + i

                    # Initialize iteration result
                    iteration_result = {
                        'iteration': iteration + 1,
                        'query_idx': query_idx + 1,
                        'question': question,
                        'options': options,
                        'expected_answer': answer,
                        'generated_query': query,
                        'successful': False,
                        'returned_results': False,
                        'correct': False,
                        'results': [],
                        'error': None
                    }

                    if not query.strip():
                        iteration_result['error'] = "No Cypher query generated"
                        print(f"  Skipping: '{question}' (No Cypher query generated)")
                    else:
                        try:
                            results = run_neo4j_query(driver, query)
                            iteration_result['results'] = results
                            iteration_result['successful'] = True

                            if len(results) > 0:
                                iteration_result['returned_results'] = True
                                result_value = list(results[0].values())[0] if results[0] else None
                                if result_value is not None:
                                    result_str = str(result_value).lower().replace(' ', '')
                                    answer_str = str(answer).lower().replace(' ', '')
                                    if result_str == answer_str:
                                        iteration_result['correct'] = True

                            print(f"  Question {query_idx + 1}: '{question}'")
                            print(f"  Generated Cypher: {query}")
                            print(f"  Results: {results}")
                            print(f"  Successful: {iteration_result['successful']}, Returned Results: {iteration_result['returned_results']}, Correct: {iteration_result['correct']}")

                        except Exception as e:
                            iteration_result['error'] = str(e)
                            print(f"  Error executing query: {e}")

                    # Store iteration result
                    results_data.append(iteration_result)

                    # Update query stats
                    query_stats[query_idx]['iterations'].append(iteration_result)
                    if iteration_result['successful']:
                        query_stats[query_idx]['successful'] = True
                    if iteration_result['returned_results']:
                        query_stats[query_idx]['returned_results'] = True
                    if iteration_result['correct']:
                        query_stats[query_idx]['correct'] = True

            except Exception as e:
                print(f"  ERROR processing batch {batch_idx + 1}: {e}")
                print(f"  Error type: {type(e).__name__}")
                print(f"  Error details: {str(e)}")

                # Try to get more specific error information
                if hasattr(e, '__cause__') and e.__cause__:
                    print(f"  Caused by: {e.__cause__}")

                # Mark all queries in this batch as failed
                for i in range(start_idx, end_idx):
                    iteration_result = {
                        'iteration': iteration + 1,
                        'query_idx': i + 1,
                        'question': all_questions[i],
                        'options': all_options[i],
                        'expected_answer': all_answers[i],
                        'generated_query': "",
                        'successful': False,
                        'returned_results': False,
                        'correct': False,
                        'results': [],
                        'error': f"Batch processing error: {type(e).__name__}: {str(e)}"
                    }
                    results_data.append(iteration_result)
                    query_stats[i]['iterations'].append(iteration_result)

    # Generate detailed results table
    print(f"\n{'='*80}")
    print("DETAILED RESULTS TABLE")
    print(f"{'='*80}")

    # Create DataFrame for detailed results
    df_results = pd.DataFrame(results_data)

    # Display detailed table
    print("\nDetailed Results by Query and Iteration:")
    print("-" * 80)

    for query_idx in range(total_queries):
        query_data = df_results[df_results['query_idx'] == query_idx + 1]
        question = query_data.iloc[0]['question']

        print(f"\nQuery {query_idx + 1}: {question}")
        print(f"Expected Answer: {query_data.iloc[0]['expected_answer']}")
        print(f"{'Iter':<4} {'Success':<8} {'Results':<8} {'Correct':<8} {'Query':<20} {'Error':<20}")
        print("-" * 80)

        for _, row in query_data.iterrows():
            query_preview = row['generated_query'][:17] + "..." if len(row['generated_query']) > 20 else row['generated_query']
            error_preview = str(row['error'])[:17] + "..." if row['error'] and len(str(row['error'])) > 20 else str(row['error']) or ""

            print(f"{row['iteration']:<4} {str(row['successful']):<8} {str(row['returned_results']):<8} {str(row['correct']):<8} {query_preview:<20} {error_preview:<20}")

    # Calculate aggregated statistics
    print(f"\n{'='*80}")
    print("AGGREGATED STATISTICS")
    print(f"{'='*80}")

    successful_queries = sum(1 for stats in query_stats.values() if stats['successful'])
    queries_with_results = sum(1 for stats in query_stats.values() if stats['returned_results'])
    correct_queries = sum(1 for stats in query_stats.values() if stats['correct'])

    print(f"\nFinal Results (Any Success Across {NUM_ITERATIONS} Iterations):")
    print(f"Successfully processed: {successful_queries} / {total_queries} queries")
    print(f"Queries that returned results: {queries_with_results} / {total_queries}")
    print(f"Correct results: {correct_queries} / {total_queries}")

    # Calculate per-iteration statistics
    print(f"\nPer-Iteration Statistics:")
    print(f"{'Iteration':<10} {'Successful':<12} {'With Results':<12} {'Correct':<10}")
    print("-" * 50)

    for iteration in range(NUM_ITERATIONS):
        iter_data = df_results[df_results['iteration'] == iteration + 1]
        successful_count = iter_data['successful'].sum()
        results_count = iter_data['returned_results'].sum()
        correct_count = iter_data['correct'].sum()

        print(f"{iteration + 1:<10} {successful_count:<12} {results_count:<12} {correct_count:<10}")

    # Save detailed results to CSV
    output_file = "neo4j_evaluation_results.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nDetailed results saved to: {output_file}")

    # Save aggregated statistics
    stats_data = []
    for query_idx, stats in query_stats.items():
        stats_data.append({
            'query_idx': query_idx + 1,
            'question': stats['iterations'][0]['question'],
            'expected_answer': stats['iterations'][0]['expected_answer'],
            'successful_any_iteration': stats['successful'],
            'returned_results_any_iteration': stats['returned_results'],
            'correct_any_iteration': stats['correct'],
            'successful_iterations': sum(1 for iter_result in stats['iterations'] if iter_result['successful']),
            'results_iterations': sum(1 for iter_result in stats['iterations'] if iter_result['returned_results']),
            'correct_iterations': sum(1 for iter_result in stats['iterations'] if iter_result['correct'])
        })

    df_stats = pd.DataFrame(stats_data)
    stats_file = "neo4j_evaluation_stats.csv"
    df_stats.to_csv(stats_file, index=False)
    print(f"Aggregated statistics saved to: {stats_file}")

    driver.close()

if __name__ == "__main__":
    main()