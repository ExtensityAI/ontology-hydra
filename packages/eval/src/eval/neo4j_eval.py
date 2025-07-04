import json
import time
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import List, Optional

import pandas as pd
from .logging_setup import logger
from neo4j import GraphDatabase
from pydantic import BaseModel, Field
from symai import Expression
from symai.strategy import contract
from symai.models import LLMDataModel
from symai.utils import RuntimeInfo
from symai.components import MetadataTracker

from eval.squad_v2.data import SquadQAPair
from ontopipe.models import KG


# Pricing configuration for cost estimation
PRICING = {
    ('GPTXChatEngine', 'gpt-4.1'): {
        'input': 2. / 1e6,
        'cached_input': 0.5 / 1e6,
        'output': 8. / 1e6
    },
    ('GPTXReasoningEngine', 'o4-mini'): {
        'input': 1.1 / 1e6,
        'cached_input': 0.275 / 1e6,
        'output': 4.4 / 1e6
    },
    ('GPTXReasoningEngine', 'o3'): {
        'input': 2. / 1e6,
        'cached_input': 0.5 / 1e6,
        'output': 8. / 1e6
    },
    ('GPTXSearchEngine', 'gpt-4.1-mini'): {
        'input': 0.40 / 1e6,
        'cached_input': 0.10 / 1e6,
        'output': 1.6 / 1e6,
        'calls': 30. / 1e3
    },
    ('GPTXSearchEngine', 'gpt-4.1'): {
        'input': 2. / 1e6,
        'cached_input': 0.5 / 1e6,
        'output': 8. / 1e6,
        'calls': 50 / 1e3
    },
    ('GPTXChatEngine', 'gpt-4o'): {
        'input': 1.2 / 1e6,
        'cached_input': 0.3 / 1e6,
        'output': 4.8 / 1e6
    },
    ('GPTXChatEngine', 'gpt-4o-mini'): {
        'input': 1.1 / 1e6,
        'cached_input': 0.275 / 1e6,
        'output': 4.4 / 1e6
    }
}


def estimate_cost(info: RuntimeInfo, pricing: dict) -> float:
    """Estimate cost based on token usage and pricing."""
    input_cost = (info.prompt_tokens - info.cached_tokens) * pricing.get('input', 0)
    cached_input_cost = info.cached_tokens * pricing.get('cached_input', 0)
    output_cost = info.completion_tokens * pricing.get('output', 0)
    return input_cost + cached_input_cost + output_cost


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

    def get_schema(self, database_name: str = "neo4j") -> str:
        """Query Neo4j for complete schema information."""
        with self.driver.session(database=database_name) as session:
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
            schema = f"Complete Neo4j Schema (Database: {database_name}):\n\n"

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

        # Basic validation of query format
        for i, query in enumerate(output.queries):
            if not query.strip():
                raise ValueError(f"Query {i+1} cannot be empty")
            if not query.upper().startswith("MATCH"):
                raise ValueError(f"Query {i+1} must start with MATCH: {query[:50]}...")

        return True

    def _extract_cypher_elements(self, query):
        """Extract Cypher elements using regex."""
        import re
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


class Neo4jConfig(BaseModel):
    """Configuration for Neo4j evaluation"""
    enabled: bool = False
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = "ontology"
    batch_size: int = 5
    num_iterations: int = 1
    use_run_specific_databases: bool = True  # New field to control database strategy
    default_database: str = "neo4j"  # Fallback database name
    auto_cleanup: bool = False  # New field to control automatic cleanup

    def __init__(self, **data):
        # Allow environment variables to override defaults
        import os

        # Override with environment variables if they exist
        if 'NEO4J_URI' in os.environ:
            data['uri'] = os.environ['NEO4J_URI']
        if 'NEO4J_USER' in os.environ:
            data['user'] = os.environ['NEO4J_USER']
        if 'NEO4J_PASSWORD' in os.environ:
            data['password'] = os.environ['NEO4J_PASSWORD']
        if 'NEO4J_USE_RUN_SPECIFIC_DATABASES' in os.environ:
            data['use_run_specific_databases'] = os.environ['NEO4J_USE_RUN_SPECIFIC_DATABASES'].lower() == 'true'
        if 'NEO4J_DEFAULT_DATABASE' in os.environ:
            data['default_database'] = os.environ['NEO4J_DEFAULT_DATABASE']
        if 'NEO4J_AUTO_CLEANUP' in os.environ:
            data['auto_cleanup'] = os.environ['NEO4J_AUTO_CLEANUP'].lower() == 'true'

        super().__init__(**data)


def run_neo4j_query(driver, query, database_name: str = "neo4j"):
    """Execute a Neo4j query and return results"""
    with driver.session(database=database_name) as session:
        result = session.run(query)
        return [record.data() for record in result]


def _extract_run_id_from_path(path: Path) -> str:
    """Extract the run ID from the evaluation path."""
    # The path structure is: eval/runs/YYYYMMDD_XXXXX/scenario_id/topics/topic_name
    # We need to extract the YYYYMMDD_XXXXX part
    parts = path.parts
    for i, part in enumerate(parts):
        if part == "runs" and i + 1 < len(parts):
            return parts[i + 1]

    # Fallback: use the last part of the path
    return path.name


def _create_run_specific_database(driver: GraphDatabase.driver, run_id: str, config: Neo4jConfig) -> str:
    """Create a run-specific database and return the database name."""
    if not config.use_run_specific_databases:
        logger.info(f"Run-specific databases disabled, using default database: {config.default_database}")
        return config.default_database

    # Convert run_id to use only lowercase letters and numbers
    # Neo4j database names will look like: r20250704pdiq9q
    import re
    # Remove all non-alphanumeric characters and convert to lowercase
    safe_run_id = re.sub(r'[^a-zA-Z0-9]', '', run_id.lower())
    # Add "r" prefix to ensure it starts with a letter
    database_name = f"r{safe_run_id}"

    try:
        with driver.session() as session:
            # First, try to check if we can list databases (requires admin privileges)
            try:
                result = session.run("SHOW DATABASES")
                existing_databases = [record["name"] for record in result]

                if database_name not in existing_databases:
                    # Try to create the database with proper quoting
                    try:
                        session.run(f'CREATE DATABASE `{database_name}`')
                        logger.info(f"Created Neo4j database: {database_name}")
                    except Exception as create_error:
                        logger.warning(f"Could not create database '{database_name}': {create_error}")
                        logger.warning("This might be due to insufficient privileges or Neo4j version limitations")
                        logger.warning(f"Falling back to default database: {config.default_database}")
                        return config.default_database
                else:
                    logger.info(f"Using existing Neo4j database: {database_name}")

            except Exception as list_error:
                logger.warning(f"Could not list databases: {list_error}")
                logger.warning("This might be due to insufficient privileges")
                logger.warning(f"Falling back to default database: {config.default_database}")
                return config.default_database

    except Exception as e:
        logger.warning(f"Could not access Neo4j: {e}")
        logger.warning(f"Falling back to default database: {config.default_database}")
        return config.default_database

    return database_name


def _load_kg_to_neo4j(kg: KG, driver: GraphDatabase.driver, database_name: str = "neo4j"):
    """Load knowledge graph into Neo4j database"""
    logger.info(f"Loading knowledge graph into Neo4j database: {database_name}")

    # Clear existing data in the specified database
    with driver.session(database=database_name) as session:
        session.run("MATCH (n) DETACH DELETE n")

        # Create constraints
        try:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE")
        except:
            # Handle case for older Neo4j versions
            pass

    # Load triplets
    total_triplets = len(kg.triplets)
    for i, triplet in enumerate(kg.triplets, 1):
        _load_triplet(driver, triplet.subject, triplet.predicate, triplet.object, database_name)
        if i % 100 == 0:
            logger.debug(f"Processed {i}/{total_triplets} triplets")

    logger.info(f"Successfully loaded {total_triplets} triplets into Neo4j database: {database_name}")


def _load_triplet(driver: GraphDatabase.driver, subject: str, predicate: str, object: str, database_name: str = "neo4j"):
    """Create a single triplet in the graph with proper labels and relationship types."""
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

    with driver.session(database=database_name) as session:
        session.run(query, subject=subject, object=object)


def _cleanup_old_databases(driver: GraphDatabase.driver, config: Neo4jConfig):
    """Clean up existing run-specific databases based on configuration."""
    if not config.auto_cleanup:
        logger.debug("Auto-cleanup disabled, skipping database cleanup")
        return

    try:
        with driver.session() as session:
            # Get list of all databases
            result = session.run("SHOW DATABASES")
            databases = [record["name"] for record in result]

            # Filter for run-specific databases (databases that start with "r" followed by lowercase letters and numbers)
            import re
            run_databases = [db for db in databases if re.match(r'^r[a-z0-9]+$', db) and db != 'neo4j' and db != 'system']

            if not run_databases:
                logger.debug("No run-specific databases found for cleanup")
                return

            logger.info(f"Found {len(run_databases)} run-specific databases for cleanup")

            cleaned_count = 0
            for db_name in run_databases:
                try:
                    logger.info(f"Cleaning up database: {db_name}")
                    session.run(f'DROP DATABASE `{db_name}`')
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"Failed to clean up database {db_name}: {e}")
                    continue

            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} run-specific databases")
            else:
                logger.debug("No databases were cleaned up")

    except Exception as e:
        logger.warning(f"Failed to perform database cleanup: {e}")


def _get_database_schema(driver: GraphDatabase.driver, database_name: str) -> str:
    """Get the complete schema information from a Neo4j database."""
    with driver.session(database=database_name) as session:
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
        schema = f"Complete Neo4j Schema (Database: {database_name}):\n\n"

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


def _eval_neo4j_qa(kg: KG, qas: List[SquadQAPair], config: Neo4jConfig, output_path: Path):
    """Evaluate QA performance using Neo4j queries"""
    if not config.enabled:
        logger.info("Neo4j evaluation disabled, skipping...")
        return

    logger.info("Starting Neo4j evaluation...")

    # Extract run ID from the output path
    run_id = _extract_run_id_from_path(output_path)
    logger.info(f"Extracted run ID: {run_id}")

    # Initialize Neo4j connection
    driver = GraphDatabase.driver(config.uri, auth=(config.user, config.password))

    try:
        # Clean up old databases at the start of evaluation
        _cleanup_old_databases(driver, config)

        # Create run-specific database
        database_name = _create_run_specific_database(driver, run_id, config)
        logger.info(f"Using Neo4j database: {database_name}")

        # Load knowledge graph into the run-specific database
        _load_kg_to_neo4j(kg, driver, database_name)

        # Initialize converter with the driver (it will use the database_name parameter in queries)
        converter = BatchQuestionToCypherConverter(driver=driver)

        # Get schema from the run-specific database
        schema = _get_database_schema(driver, database_name)

        # Collect all questions, options, and answers
        all_questions = []
        all_options = []
        all_answers = []

        for qa in qas:
            all_questions.append(qa.question)
            # For SQuAD format, we need to extract options from the answers
            options = [ans.text for ans in qa.answers] if qa.answers else []
            all_options.append(options)
            all_answers.append(qa.answers[0].text if qa.answers else "")

        total_queries = len(all_questions)
        logger.info(f"Processing {total_queries} questions with Neo4j")

        # Initialize results storage
        results_data = []
        query_stats = defaultdict(lambda: {
            'successful': False,
            'returned_results': False,
            'correct': False,
            'iterations': []
        })

        # Initialize runtime tracking
        total_runtime_info = RuntimeInfo(0, 0, 0, 0, 0, 0, 0, 0)

        # Run evaluation N times
        for iteration in range(config.num_iterations):
            logger.info(f"Neo4j iteration {iteration + 1}/{config.num_iterations}")

            # Process questions in batches
            num_batches = ceil(len(all_questions) / config.batch_size)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = min((batch_idx + 1) * config.batch_size, len(all_questions))

                logger.debug(f"Processing Neo4j batch {batch_idx + 1}/{num_batches}")

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

                    # Track runtime information for this batch
                    batch_runtime_info = RuntimeInfo(0, 0, 0, 0, 0, 0, 0, 0)
                    with MetadataTracker() as tracker:
                        start_time = time.perf_counter()
                        try:
                            result = converter(input=input_data)
                        except Exception as e:
                            raise e
                        finally:
                            time.sleep(0.05)
                            end_time = time.perf_counter()

                    # Process runtime information
                    usage_per_engine = RuntimeInfo.from_tracker(tracker, end_time - start_time)

                    for (engine_name, model_name), data in usage_per_engine.items():
                        if (engine_name, model_name) in PRICING:
                            batch_runtime_info += RuntimeInfo.estimate_cost(data, estimate_cost, pricing=PRICING[(engine_name, model_name)])
                        else:
                            batch_runtime_info += data

                    # Add total elapsed time
                    batch_runtime_info.total_elapsed_time = end_time - start_time
                    total_runtime_info += batch_runtime_info

                    queries = result.queries
                    logger.debug(f"Generated {len(queries)} Neo4j queries")

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
                            logger.debug(f"Skipping: '{question}' (No Cypher query generated)")
                        else:
                            try:
                                # Execute query in the run-specific database
                                with driver.session(database=database_name) as session:
                                    result = session.run(query)
                                    results = [record.data() for record in result]

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

                                logger.debug(f"Neo4j Question {query_idx + 1}: '{question}' - Success: {iteration_result['successful']}, Results: {iteration_result['returned_results']}, Correct: {iteration_result['correct']}")

                            except Exception as e:
                                iteration_result['error'] = str(e)
                                logger.debug(f"Error executing Neo4j query: {e}")

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
                    logger.error(f"Error processing Neo4j batch {batch_idx + 1}: {e}")

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

        # Calculate aggregated statistics
        successful_queries = sum(1 for stats in query_stats.values() if stats['successful'])
        queries_with_results = sum(1 for stats in query_stats.values() if stats['returned_results'])
        correct_queries = sum(1 for stats in query_stats.values() if stats['correct'])

        logger.info("Neo4j Evaluation Results:")
        logger.info(f"Database used: {database_name}")
        logger.info(f"Successfully processed: {successful_queries} / {total_queries} queries")
        logger.info(f"Queries that returned results: {queries_with_results} / {total_queries}")
        logger.info(f"Correct results: {correct_queries} / {total_queries}")
        logger.info(f"Total elapsed time: {total_runtime_info.total_elapsed_time:.2f} seconds")
        logger.info(f"Estimated cost: ${total_runtime_info.cost_estimate:.4f}")

        # Create DataFrames for CSV output
        df_results = pd.DataFrame(results_data)

        # Debug: Check if DataFrame is empty or missing columns
        if df_results.empty:
            logger.warning("No results data to process, creating empty DataFrames")
            df_results = pd.DataFrame(columns=['iteration', 'query_idx', 'question', 'options', 'expected_answer',
                                              'generated_query', 'successful', 'returned_results', 'correct', 'results', 'error'])

        # Aggregated statistics DataFrame
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

        # Per-iteration statistics DataFrame
        iteration_stats_data = []
        for iteration in range(config.num_iterations):
            try:
                # Check if 'iteration' column exists
                if 'iteration' not in df_results.columns:
                    logger.warning("'iteration' column not found in results DataFrame")
                    # Create empty iteration data
                    iteration_stats_data.append({
                        'iteration': iteration + 1,
                        'successful_count': 0,
                        'results_count': 0,
                        'correct_count': 0,
                        'total_queries': 0
                    })
                    continue

                iter_data = df_results[df_results['iteration'] == iteration + 1]
                successful_count = iter_data['successful'].sum() if 'successful' in iter_data.columns else 0
                results_count = iter_data['returned_results'].sum() if 'returned_results' in iter_data.columns else 0
                correct_count = iter_data['correct'].sum() if 'correct' in iter_data.columns else 0

                iteration_stats_data.append({
                    'iteration': iteration + 1,
                    'successful_count': successful_count,
                    'results_count': results_count,
                    'correct_count': correct_count,
                    'total_queries': len(iter_data)
                })
            except Exception as e:
                logger.warning(f"Error processing iteration {iteration + 1}: {e}")
                iteration_stats_data.append({
                    'iteration': iteration + 1,
                    'successful_count': 0,
                    'results_count': 0,
                    'correct_count': 0,
                    'total_queries': 0
                })

        # Add total row
        total_successful = sum(row['successful_count'] for row in iteration_stats_data)
        total_results = sum(row['results_count'] for row in iteration_stats_data)
        total_correct = sum(row['correct_count'] for row in iteration_stats_data)
        total_queries = sum(row['total_queries'] for row in iteration_stats_data)

        iteration_stats_data.append({
            'iteration': 'TOTAL',
            'successful_count': total_successful,
            'results_count': total_results,
            'correct_count': total_correct,
            'total_queries': total_queries
        })

        df_iteration_stats = pd.DataFrame(iteration_stats_data)

        # Runtime statistics DataFrame
        runtime_stats_data = [{
            'metric': 'total_elapsed_time_seconds',
            'value': total_runtime_info.total_elapsed_time,
            'formatted_value': f"{total_runtime_info.total_elapsed_time:.2f}"
        }, {
            'metric': 'total_prompt_tokens',
            'value': total_runtime_info.prompt_tokens,
            'formatted_value': f"{total_runtime_info.prompt_tokens:,}"
        }, {
            'metric': 'total_completion_tokens',
            'value': total_runtime_info.completion_tokens,
            'formatted_value': f"{total_runtime_info.completion_tokens:,}"
        }, {
            'metric': 'total_reasoning_tokens',
            'value': total_runtime_info.reasoning_tokens,
            'formatted_value': f"{total_runtime_info.reasoning_tokens:,}"
        }, {
            'metric': 'total_cached_tokens',
            'value': total_runtime_info.cached_tokens,
            'formatted_value': f"{total_runtime_info.cached_tokens:,}"
        }, {
            'metric': 'total_tokens',
            'value': total_runtime_info.total_tokens,
            'formatted_value': f"{total_runtime_info.total_tokens:,}"
        }, {
            'metric': 'total_calls',
            'value': total_runtime_info.total_calls,
            'formatted_value': str(total_runtime_info.total_calls)
        }, {
            'metric': 'estimated_cost_usd',
            'value': total_runtime_info.cost_estimate,
            'formatted_value': f"${total_runtime_info.cost_estimate:.4f}"
        }]

        df_runtime_stats = pd.DataFrame(runtime_stats_data)

        # Save all DataFrames to CSV files
        neo4j_output_path = output_path / "neo4j_eval"
        neo4j_output_path.mkdir(exist_ok=True)

        detailed_results_file = neo4j_output_path / "neo4j_evaluation_results.csv"
        df_results.to_csv(detailed_results_file, index=False)
        logger.info(f"Neo4j detailed results saved to: {detailed_results_file}")

        aggregated_stats_file = neo4j_output_path / "neo4j_evaluation_stats.csv"
        df_stats.to_csv(aggregated_stats_file, index=False)
        logger.info(f"Neo4j aggregated statistics saved to: {aggregated_stats_file}")

        iteration_stats_file = neo4j_output_path / "neo4j_iteration_stats.csv"
        df_iteration_stats.to_csv(iteration_stats_file, index=False)
        logger.info(f"Neo4j per-iteration statistics saved to: {iteration_stats_file}")

        runtime_stats_file = neo4j_output_path / "neo4j_runtime_stats.csv"
        df_runtime_stats.to_csv(runtime_stats_file, index=False)
        logger.info(f"Neo4j runtime statistics saved to: {runtime_stats_file}")

        # Save summary metrics
        neo4j_metrics = {
            'database_name': database_name,
            'run_id': run_id,
            'successful_queries': successful_queries,
            'total_queries': total_queries,
            'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
            'queries_with_results': queries_with_results,
            'results_rate': queries_with_results / total_queries if total_queries > 0 else 0,
            'correct_queries': correct_queries,
            'accuracy': correct_queries / total_queries if total_queries > 0 else 0,
            'total_elapsed_time_seconds': total_runtime_info.total_elapsed_time,
            'estimated_cost_usd': total_runtime_info.cost_estimate,
            'auto_cleanup_enabled': config.auto_cleanup
        }

        metrics_file = neo4j_output_path / "neo4j_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(neo4j_metrics, f, indent=2)
        logger.info(f"Neo4j metrics saved to: {metrics_file}")

    finally:
        driver.close()