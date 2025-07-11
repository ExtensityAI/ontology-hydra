import json
import re
import time
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    ('GPTXChatEngine', 'gpt-4.1-mini'): {
        'input': 0.40 / 1e6,
        'cached_input': 0.10 / 1e6,
        'output': 1.6 / 1e6
    },
    ('GPTXChatEngine', 'gpt-4.1'): {
        'input': 2. / 1e6,
        'cached_input': 0.5 / 1e6,
        'output': 8. / 1e6
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
    },
    ('GPTXChatEngine', 'gpt-4.1-nano'): {
        'input': 0.10 / 1e6,
        'cached_input': 0.025 / 1e6,
        'output': 0.40 / 1e6
    },
    ('GeminiEngine', 'gemini-2.5-flash'): {
        'input': 0.15 / 1e6,
        'cached_input': 0.0375 / 1e6,
        'output': 0.60 / 1e6
    },
    ('GeminiEngine', 'gemini-2.5-pro'): {
        'input': 1.25 / 1e6,
        'cached_input': 0.3125 / 1e6,
        'output': 10.00 / 1e6
    }
}


def estimate_cost(info: RuntimeInfo, pricing: dict) -> float:
    """Estimate cost based on token usage and pricing."""
    input_cost = (info.prompt_tokens - info.cached_tokens) * pricing.get('input', 0)
    cached_input_cost = info.cached_tokens * pricing.get('cached_input', 0)
    output_cost = info.completion_tokens * pricing.get('output', 0)
    return input_cost + cached_input_cost + output_cost



class BatchAnswerInput(LLMDataModel):
    """Input for batch answer matching"""
    answer_pairs: List[dict] = Field(description="List of answer pairs to match, each containing 'predicted_answer' and 'expected_answer'")

class BatchAnswerOutput(LLMDataModel):
    """Output for batch answer matching"""
    match_scores: List[float] = Field(description="List of similarity scores between predicted and expected answers")

@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=True,
    remedy_retry_params=dict(
        tries=25,
        delay=0.5,
        max_delay=10,
        jitter=0.1,
        backoff=2,
        graceful=False
    )
)
class BatchFuzzyAnswerMatcher(Expression):
    def __init__(
        self,
        threshold: float = 0.8,
        seed: Optional[int] = 42,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.seed = seed

    def forward(self, input: BatchAnswerInput) -> BatchAnswerOutput:
        """Match predicted answers with expected answers using fuzzy matching for a batch."""
        if self.contract_result is None:
            print(self.contract_result)
            print("Failed")
            return BatchAnswerOutput(match_scores=[0.0] * len(input.answer_pairs))
        return self.contract_result

    def pre(self, input: BatchAnswerInput) -> bool:
        """Validate input data contains required fields."""
        if not isinstance(input, BatchAnswerInput):
            raise ValueError("Input must be a BatchAnswerInput instance!")
        if not input.answer_pairs or len(input.answer_pairs) == 0:
            raise ValueError("Answer pairs list cannot be empty!")
        for pair in input.answer_pairs:
            if not isinstance(pair, dict) or 'predicted_answer' not in pair or 'expected_answer' not in pair:
                raise ValueError("Each answer pair must be a dict with 'predicted_answer' and 'expected_answer' keys!")
        return True

    def post(self, output: BatchAnswerOutput) -> bool:
        """Validate output contains match scores."""
        if not isinstance(output, BatchAnswerOutput):
            raise ValueError("Output must be a BatchAnswerOutput instance!")
        if len(output.match_scores) != self.num_a:
            raise ValueError(f"Number of match scores ({len(output.match_scores)}) does not match number of input pairs ({self.num_a})")
        if not isinstance(output.match_scores, list):
            raise ValueError("match_scores must be a list!")
        for score in output.match_scores:
            if not isinstance(score, (int, float)):
                raise ValueError("All match scores must be numbers!")
            if score < 0 or score > 1:
                raise ValueError("All match scores must be between 0 and 1!")
        return True

    def act(self, input: BatchAnswerInput, **kwargs) -> BatchAnswerInput:
        self.num_a = len(input.answer_pairs)
        return input

    @property
    def prompt(self) -> str:
        """Return prompt template for batch fuzzy answer matching."""
        return f"""[[Batch Fuzzy Answer Matching]]
Compare multiple pairs of predicted answers with their expected answers and return similarity scores between 0 and 1 for each pair.

For each answer pair, analyze the similarity considering:
1. Exact match
2. Case-insensitive match
3. Semantic similarity
4. Word overlap and containment
5. Character-level similarity

Be lenient with formatting differences (case, spaces, underscores) but strict with semantic meaning.
Focus on whether the answers convey the same concept.

Return a list of similarity scores in the same order as the input pairs."""



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
    verbose=True,
    remedy_retry_params=dict(
        tries=25,
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
        database_name: str = "neo4j",
        seed: Optional[int] = 42,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.data_model = Neo4jQueryOutput
        self.driver = driver
        self.database_name = database_name

    def get_schema(self, database_name: str = "neo4j") -> str:
        """Get schema using Neo4j built-in procedures."""
        return _get_neo4j_schema(self.driver, database_name)

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
        if len(output.queries) != self.num_qs:
            raise ValueError(f"Number of output queries ({len(output.queries)}) does not match number of input questions ({self.num_qs})")

        try:
            with self.driver.session(database=self.database_name) as session:
                # Get schema elements using Neo4j built-in procedures
                labels_result = session.run("CALL db.labels() YIELD label RETURN label")
                labels = {record["label"] for record in labels_result}

                rels_result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
                rels = {record["relationshipType"] for record in rels_result}

                props_result = session.run("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey")
                props = {record["propertyKey"] for record in props_result}
        except Exception as e:
            raise ValueError(f"Failed to get schema from Neo4j database '{self.database_name}': {e}")

        # Debug: Log available schema elements
        logger.debug(f"Available labels in database '{self.database_name}': {sorted(labels)}")
        logger.debug(f"Available relationships in database '{self.database_name}': {sorted(rels)}")
        logger.debug(f"Available properties in database '{self.database_name}': {sorted(props)}")

        for i, query in enumerate(output.queries):
            if not query.strip():
                raise ValueError(f"Query {i+1} cannot be empty")
            if not query.upper().startswith("MATCH"):
                raise ValueError(f"Query {i+1} must start with MATCH: {query[:50]}...")

            # Extract and validate schema elements
            q_labels, q_rels, q_props = self._extract_cypher_elements(query)

            # Debug: Log extracted elements
            logger.debug(f"Query {i+1} extracted labels: {q_labels}")
            logger.debug(f"Query {i+1} extracted relationships: {q_rels}")
            logger.debug(f"Query {i+1} extracted properties: {q_props}")

            invalid_labels = q_labels - labels
            invalid_rels = q_rels - rels
            invalid_props = q_props - props

            if invalid_labels:
                raise ValueError(f"Query {i+1} uses non-existent labels: {invalid_labels}. Available labels: {sorted(labels)}")
            if invalid_rels:
                raise ValueError(f"Query {i+1} uses non-existent relationships: {invalid_rels}. Available relationships: {sorted(rels)}")
            if invalid_props:
                raise ValueError(f"Query {i+1} uses non-existent properties: {invalid_props}. Available properties: {sorted(props)}")

            # Extract name values from property patterns and WHERE clauses
            pattern1 = r'(\{[^}]*name\s*:\s*[\'"`])([^\'"`]+)([\'"`][^}]*\})'
            pattern2 = r'(WHERE\s+[a-zA-Z_][a-zA-Z0-9_]*\.name\s*=\s*[\'"`])([^\'"`]+)([\'"`])'

            # Check name values for proper formatting
            for pattern in [pattern1, pattern2]:
                matches = re.finditer(pattern, query)
                for match in matches:
                    name_value = match.group(2)
                    if any(c.isupper() for c in name_value) or ' ' in name_value:
                        raise ValueError(f"Name value '{name_value}' must be lowercase and have underscores instead of spaces")

        return True



    def _extract_cypher_elements(self, query):
        """Extract Cypher elements using regex."""
        # Extract node labels - look for :Label patterns in node definitions
        # Node labels appear in patterns like (n:Label) or (n:Label1:Label2)
        # This regex looks for :Label inside parentheses, which indicates node labels
        labels = set(re.findall(r"\([^)]*:([A-Z][a-zA-Z0-9_]*|[a-z][a-zA-Z0-9_]*|[A-Z][a-z]+[A-Z][a-zA-Z0-9_]*)[^)]*\)", query))

        # Extract relationship types - look for :RELATIONSHIP patterns in relationship definitions
        # Relationship types appear in patterns like [r:RELATIONSHIP] or -[:RELATIONSHIP]->
        # This regex looks for :RELATIONSHIP inside square brackets, which indicates relationship types
        relationships = set()
        # Pattern 1: [r:RELATIONSHIP] or [r:RELATIONSHIP]-> or [r:RELATIONSHIP]-
        rel_pattern1 = re.findall(r"\[[^\]]*:([A-Z_]+)[^\]]*\]", query)
        relationships.update(rel_pattern1)

        # Pattern 2: -[:RELATIONSHIP]-> or -[:RELATIONSHIP]-
        rel_pattern2 = re.findall(r"-\s*\[:([A-Z_]+)\]\s*[->-]", query)
        relationships.update(rel_pattern2)

        # Extract property names - look for .property patterns
        properties = set(re.findall(r"\.\s*([a-zA-Z0-9_]+)", query))
        return labels, relationships, properties

    def act(self, input: Neo4jQueryInput, **kwargs) -> Neo4jQueryInput:
        self.num_qs = len(input.questions)
        return input


    @property
    def prompt(self) -> str:
        return """[[Neo4j Cypher Query Generation - Batch Processing]]
Convert the given list of natural language questions into Neo4j Cypher queries using ONLY the schema elements provided. No other elements exist or can be used.

IMPORTANT NAME FORMATTING RULES:
- ALL names in property values must be lowercase with underscores instead of spaces
- Examples:
  * "Magnetic Resonance Imaging" becomes "magnetic_resonance_imaging"
  * "Robotic Surgical System" becomes "robotic_surgical_system"
  * "Ultrasound Guided Biopsy" becomes "ultrasound_guided_biopsy"
  * "Dermatological Surgery" becomes "dermatological_surgery"
- This applies to all {name: 'value'} patterns in the queries and (name = 'value') patterns
- Node labels and relationship types remain as provided in the schema
- Always use the exact names from the question/options, but convert them to this format

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
MATCH (p:MedicalProcedure {name: 'dermatological_surgery'})
OPTIONAL MATCH (p)-[r1]-(d:MedicalDevice)
OPTIONAL MATCH (p)-[r2]-(t:Technique)
OPTIONAL MATCH (p)-[r3]-(m:Material)
RETURN
    COLLECT(DISTINCT d.name) as devices,
    COLLECT(DISTINCT t.name) as techniques,
    COLLECT(DISTINCT m.name) as materials

Example for "What type of imaging is used to detect bone fractures?":
MATCH (em:ImagingModality {name: 'x_ray'}) RETURN em.name as imaging_type

Example for "What power source is used in a continuous glucose monitor?":
MATCH (d:MedicalDevice {name: 'continuous_glucose_monitor'})
OPTIONAL MATCH (d)-[r:POWERED_BY]-(p:PowerSource)
WHERE p.name = 'lithium_battery'
RETURN p.name AS power_source

Remember:
- Generate one query for each question in the input batch
- Only use the exact node labels, relationship types, and properties listed in the schema
- ALL names in property values must be lowercase with underscores instead of spaces
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
    fuzzy_threshold: float = 0.8  # Threshold for fuzzy matching
    max_workers: int = 4  # New field for parallel processing

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
        if 'NEO4J_FUZZY_THRESHOLD' in os.environ:
            data['fuzzy_threshold'] = float(os.environ['NEO4J_FUZZY_THRESHOLD'])
        if 'NEO4J_MAX_WORKERS' in os.environ:
            data['max_workers'] = int(os.environ['NEO4J_MAX_WORKERS'])

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
    import time
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

                        # Wait for database to be available and verify it exists
                        max_retries = 10
                        retry_delay = 0.5
                        for attempt in range(max_retries):
                            try:
                                time.sleep(retry_delay)
                                # Try to verify the database exists by listing databases again
                                result = session.run("SHOW DATABASES")
                                updated_databases = [record["name"] for record in result]
                                if database_name in updated_databases:
                                    logger.info(f"Database {database_name} verified and ready")
                                    break
                                else:
                                    logger.debug(f"Database {database_name} not yet available, attempt {attempt + 1}/{max_retries}")
                            except Exception as verify_error:
                                logger.debug(f"Error verifying database {database_name}, attempt {attempt + 1}/{max_retries}: {verify_error}")

                        # Final verification attempt
                        try:
                            with driver.session(database=database_name) as test_session:
                                test_session.run("RETURN 1")
                                logger.info(f"Database {database_name} is accessible")
                        except Exception as final_test_error:
                            logger.warning(f"Database {database_name} created but not accessible: {final_test_error}")
                            logger.warning(f"Falling back to default database: {config.default_database}")
                            return config.default_database

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
    """Load knowledge graph into Neo4j database using CSV method"""
    logger.info(f"Loading knowledge graph into Neo4j database: {database_name}")

    import time
    import json
    import csv
    import os

    # Retry mechanism for database connection
    max_retries = 5
    retry_delay = 1.0

    for attempt in range(max_retries):
        try:
            # Clear existing data in the specified database
            with driver.session(database=database_name) as session:
                session.run("MATCH (n) DETACH DELETE n")

                # Create constraints
                try:
                    session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.name IS UNIQUE")
                except:
                    # Handle case for older Neo4j versions
                    pass

            # If we get here, the database is accessible
            break

        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(f"Database {database_name} not accessible on attempt {attempt + 1}/{max_retries}: {e}")
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Failed to access database {database_name} after {max_retries} attempts: {e}")
                raise

    # Neo4j import folder path
    neo4j_import_dir = "/Users/ryang/Library/Application Support/Neo4j Desktop/Application/relate-data/dbmss/dbms-8ff6e63e-4586-411a-a8ba-f42cb734d84b/import"

    # Convert KG to CSV format
    triplets = kg.triplets

    # Collect unique node names
    nodes = set()
    for t in triplets:
        nodes.add(t.subject)
        nodes.add(t.object)

    # Generate identifier for CSV files
    identifier = f"kg_{database_name}_{int(time.time())}"

    # Write nodes.csv
    nodes_file = os.path.join(neo4j_import_dir, f"nodes_{identifier}.csv")
    with open(nodes_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])  # header
        for node in sorted(nodes):
            writer.writerow([node])

    # Write relationships.csv
    relationships_file = os.path.join(neo4j_import_dir, f"relationships_{identifier}.csv")
    with open(relationships_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["subject", "predicate", "object"])  # header
        for t in triplets:
            writer.writerow([t.subject, t.predicate, t.object])

    logger.info(f"CSV files created: {nodes_file}, {relationships_file}")

    # Load nodes from CSV
    with driver.session(database=database_name) as session:
        filename = os.path.basename(nodes_file)
        file_url = f"file:///{filename}"

        result = session.run(f"""
        LOAD CSV WITH HEADERS FROM '{file_url}' AS row
        MERGE (n:Entity {{name: row.name}})
        RETURN count(*) as nodes_created
        """)
        count = result.single()["nodes_created"]
        logger.info(f"Loaded {count} nodes from CSV")

    # Load relationships from CSV
    with driver.session(database=database_name) as session:
        filename = os.path.basename(relationships_file)
        file_url = f"file:///{filename}"

        result = session.run(f"""
        LOAD CSV WITH HEADERS FROM '{file_url}' AS row
        MATCH (a:Entity {{name: row.subject}}), (b:Entity {{name: row.object}})
        CALL apoc.create.relationship(a, row.predicate, {{}}, b) YIELD rel
        RETURN count(*) as relationships_created
        """)
        count = result.single()["relationships_created"]
        logger.info(f"Loaded {count} relationships from CSV")

    logger.info(f"Successfully loaded knowledge graph into Neo4j database: {database_name} using CSV method")


def _get_neo4j_schema(driver: GraphDatabase.driver, database_name: str) -> str:
    """Get schema using Neo4j built-in procedures and functions."""
    with driver.session(database=database_name) as session:
        schema_parts = []
        schema_parts.append(f"Complete Neo4j Schema (Database: {database_name}):\n")

        # 1. Get node labels using CALL db.labels()
        try:
            labels_result = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
            labels = [record["label"] for record in labels_result]
            schema_parts.append("\n1. Node Labels:")
            for label in labels:
                # Get count for each label
                count_result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                count = count_result.single()["count"]
                schema_parts.append(f"   - {label}: {count} nodes")
        except Exception as e:
            logger.warning(f"Could not get node labels: {e}")
            schema_parts.append("\n1. Node Labels: Could not retrieve")

        # 2. Get relationship types using CALL db.relationshipTypes()
        try:
            rels_result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType")
            rels = [record["relationshipType"] for record in rels_result]
            schema_parts.append("\n2. Relationship Types:")
            for rel in rels:
                # Get count for each relationship type
                count_result = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) as count")
                count = count_result.single()["count"]
                schema_parts.append(f"   - :{rel}: {count} relationships")
        except Exception as e:
            logger.warning(f"Could not get relationship types: {e}")
            schema_parts.append("\n2. Relationship Types: Could not retrieve")

        # 3. Get property keys using CALL db.propertyKeys()
        try:
            props_result = session.run("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey ORDER BY propertyKey")
            props = [record["propertyKey"] for record in props_result]
            schema_parts.append("\n3. Property Keys:")
            for prop in props:
                schema_parts.append(f"   - {prop}")
        except Exception as e:
            logger.warning(f"Could not get property keys: {e}")
            schema_parts.append("\n3. Property Keys: Could not retrieve")

        # 4. Get constraints using CALL db.constraints()
        try:
            constraints_result = session.run("CALL db.constraints() YIELD name, description RETURN name, description ORDER BY name")
            constraints = [(record["name"], record["description"]) for record in constraints_result]
            if constraints:
                schema_parts.append("\n4. Constraints:")
                for name, description in constraints:
                    schema_parts.append(f"   - {name}: {description}")
            else:
                schema_parts.append("\n4. Constraints: None")
        except Exception as e:
            logger.warning(f"Could not get constraints: {e}")
            schema_parts.append("\n4. Constraints: Could not retrieve")

        # 5. Get indexes using CALL db.indexes()
        try:
            indexes_result = session.run("CALL db.indexes() YIELD name, type, labelsOrTypes, properties RETURN name, type, labelsOrTypes, properties ORDER BY name")
            indexes = [(record["name"], record["type"], record["labelsOrTypes"], record["properties"]) for record in indexes_result]
            if indexes:
                schema_parts.append("\n5. Indexes:")
                for name, idx_type, labels, props in indexes:
                    schema_parts.append(f"   - {name}: {idx_type} on {labels} ({props})")
            else:
                schema_parts.append("\n5. Indexes: None")
        except Exception as e:
            logger.warning(f"Could not get indexes: {e}")
            schema_parts.append("\n5. Indexes: Could not retrieve")

        # 6. Get database info using CALL db.info()
        try:
            info_result = session.run("CALL db.info() YIELD name, version, edition RETURN name, version, edition")
            info = info_result.single()
            schema_parts.append(f"\n6. Database Info:")
            schema_parts.append(f"   - Name: {info['name']}")
            schema_parts.append(f"   - Version: {info['version']}")
            schema_parts.append(f"   - Edition: {info['edition']}")
        except Exception as e:
            logger.warning(f"Could not get database info: {e}")
            schema_parts.append("\n6. Database Info: Could not retrieve")

        return "\n".join(schema_parts)


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





def _process_batch_parallel(batch_data, converter, batch_fuzzy_matcher, driver, database_name, config, iteration, start_idx, schema):
    """Process a single batch of questions in parallel.

    Args:
        batch_data: Tuple of (batch_questions, batch_options, batch_answers)
        converter: BatchQuestionToCypherConverter instance
        batch_fuzzy_matcher: BatchFuzzyAnswerMatcher instance
        driver: Neo4j driver
        database_name: Database name to use
        config: Neo4jConfig instance
        iteration: Current iteration number
        start_idx: Starting index for this batch
        schema: Database schema string

    Returns:
        Tuple of (batch_results, batch_runtime_info)
    """
    batch_questions, batch_options, batch_answers = batch_data

    # Initialize batch runtime tracking
    batch_runtime_info = RuntimeInfo(0, 0, 0, 0, 0, 0, 0, 0)
    batch_iteration_results = []

    try:
        # Convert questions to queries using contract
        input_data = Neo4jQueryInput(
            questions=batch_questions,
            options=batch_options,
            answers=batch_answers,
            schema=schema
        )

        # Track runtime information for this batch
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
                logger.debug(f"Using pricing model: {engine_name} - {model_name}")
                batch_runtime_info += RuntimeInfo.estimate_cost(data, estimate_cost, pricing=PRICING[(engine_name, model_name)])
            else:
                logger.warning(f"No pricing model found for: {engine_name} - {model_name}, using raw data")
                batch_runtime_info += data

        # Add total elapsed time
        batch_runtime_info.total_elapsed_time = end_time - start_time

        queries = result.queries
        logger.debug(f"Generated {len(queries)} Neo4j queries")

        # Initialize batch results
        batch_answer_pairs = []

        # First pass: Execute all queries and collect results
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
                            # Add to batch for fuzzy matching
                            batch_answer_pairs.append({
                                'predicted_answer': str(result_value),
                                'expected_answer': str(answer)
                            })
                        else:
                            # No result value, add empty pair
                            batch_answer_pairs.append({
                                'predicted_answer': '',
                                'expected_answer': str(answer)
                            })
                    else:
                        # No results, add empty pair
                        batch_answer_pairs.append({
                            'predicted_answer': '',
                            'expected_answer': str(answer)
                        })

                    logger.debug(f"Neo4j Question {query_idx + 1}: '{question}' - Success: {iteration_result['successful']}, Results: {iteration_result['returned_results']}")

                except Exception as e:
                    iteration_result['error'] = str(e)
                    logger.debug(f"Error executing Neo4j query: {e}")
                    # Add empty pair for failed query
                    batch_answer_pairs.append({
                        'predicted_answer': '',
                        'expected_answer': str(answer)
                    })

            batch_iteration_results.append(iteration_result)

        # Second pass: Perform batch fuzzy matching if we have answer pairs
        if batch_answer_pairs:
            try:
                # Prepare batch input for fuzzy matching
                batch_match_input = BatchAnswerInput(answer_pairs=batch_answer_pairs)

                # Use batch fuzzy matcher to determine similarity scores
                batch_match_output = batch_fuzzy_matcher(input=batch_match_input)
                batch_match_scores = batch_match_output.match_scores

                # Apply match scores to results
                for i, (iteration_result, match_score) in enumerate(zip(batch_iteration_results, batch_match_scores)):
                    if iteration_result['returned_results'] and iteration_result['successful']:
                        # Determine if answer is correct based on threshold
                        is_correct = match_score >= config.fuzzy_threshold
                        iteration_result['correct'] = is_correct
                        iteration_result['match_score'] = match_score
                        logger.debug(f"Batch fuzzy match {i+1} - Score: {match_score:.3f}, Threshold: {config.fuzzy_threshold}, Correct: {is_correct}")
                    else:
                        # No results or failed query, mark as incorrect
                        iteration_result['correct'] = False
                        iteration_result['match_score'] = 0.0

            except Exception as match_error:
                logger.warning(f"Batch fuzzy matching failed: {match_error}")
                # Mark all as incorrect if batch fuzzy matching fails
                for iteration_result in batch_iteration_results:
                    iteration_result['correct'] = False
                    iteration_result['match_score'] = 0.0

    except Exception as e:
        logger.error(f"Error processing batch starting at index {start_idx}: {e}")

        # Mark all queries in this batch as failed
        for i in range(len(batch_questions)):
            iteration_result = {
                'iteration': iteration + 1,
                'query_idx': start_idx + i + 1,
                'question': batch_questions[i],
                'options': batch_options[i],
                'expected_answer': batch_answers[i],
                'generated_query': "",
                'successful': False,
                'returned_results': False,
                'correct': False,
                'results': [],
                'error': f"Batch processing error: {type(e).__name__}: {str(e)}"
            }
            batch_iteration_results.append(iteration_result)

    return batch_iteration_results, batch_runtime_info


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
        converter = BatchQuestionToCypherConverter(driver=driver, database_name=database_name)

        # Initialize batch fuzzy answer matcher
        batch_fuzzy_matcher = BatchFuzzyAnswerMatcher(threshold=config.fuzzy_threshold)
        logger.info(f"Batch fuzzy matching enabled with threshold: {config.fuzzy_threshold}")

        # Get schema using Neo4j built-in functions
        schema = _get_neo4j_schema(driver, database_name)

        # Collect all questions, options, and answers
        all_questions = []
        all_options = []
        all_answers = []

        for qa in qas:
            all_questions.append(qa.question)
            # Extract multiple choice options from all_answers
            options = [ans.text for ans in qa.all_answers] if qa.all_answers else []
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

            # Process questions in batches with parallel execution
            num_batches = ceil(len(all_questions) / config.batch_size)

            # Prepare batch data for parallel processing
            batch_tasks = []
            for batch_idx in range(num_batches):
                start_idx = batch_idx * config.batch_size
                end_idx = min((batch_idx + 1) * config.batch_size, len(all_questions))

                batch_questions = all_questions[start_idx:end_idx]
                batch_options = all_options[start_idx:end_idx]
                batch_answers = all_answers[start_idx:end_idx]

                batch_tasks.append((batch_questions, batch_options, batch_answers, start_idx))

            logger.info(f"Processing {num_batches} batches with up to {config.max_workers} parallel workers")

            # Process batches in parallel
            with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                # Submit all batch tasks
                future_to_batch = {
                    executor.submit(
                        _process_batch_parallel,
                        batch_data,
                        converter,
                        batch_fuzzy_matcher,
                        driver,
                        database_name,
                        config,
                        iteration,
                        start_idx,
                        schema
                    ): (batch_data, start_idx) for batch_data, start_idx in [(task[0:3], task[3]) for task in batch_tasks]
                }

                # Collect results as they complete
                for future in as_completed(future_to_batch):
                    batch_data, start_idx = future_to_batch[future]
                    batch_idx = start_idx // config.batch_size

                    try:
                        batch_iteration_results, batch_runtime_info = future.result()

                        # Add batch runtime to total
                        total_runtime_info += batch_runtime_info

                        # Store all iteration results
                        for iteration_result in batch_iteration_results:
                            results_data.append(iteration_result)

                            # Update query stats
                            query_idx = iteration_result['query_idx'] - 1  # Convert back to 0-based index
                            query_stats[query_idx]['iterations'].append(iteration_result)
                            if iteration_result['successful']:
                                query_stats[query_idx]['successful'] = True
                            if iteration_result['returned_results']:
                                query_stats[query_idx]['returned_results'] = True
                            if iteration_result['correct']:
                                query_stats[query_idx]['correct'] = True

                        logger.debug(f"Completed Neo4j batch {batch_idx + 1}/{num_batches}")

                    except Exception as e:
                        logger.error(f"Error processing Neo4j batch {batch_idx + 1}: {e}")

                        # Mark all queries in this batch as failed
                        batch_questions, batch_options, batch_answers = batch_data
                        for i in range(len(batch_questions)):
                            iteration_result = {
                                'iteration': iteration + 1,
                                'query_idx': start_idx + i + 1,
                                'question': batch_questions[i],
                                'options': batch_options[i],
                                'expected_answer': batch_answers[i],
                                'generated_query': "",
                                'successful': False,
                                'returned_results': False,
                                'correct': False,
                                'results': [],
                                'error': f"Batch processing error: {type(e).__name__}: {str(e)}"
                            }
                            results_data.append(iteration_result)
                            query_stats[start_idx + i]['iterations'].append(iteration_result)

        # Calculate aggregated statistics
        successful_queries = sum(1 for stats in query_stats.values() if stats['successful'])
        queries_with_results = sum(1 for stats in query_stats.values() if stats['returned_results'])
        correct_queries = sum(1 for stats in query_stats.values() if stats['correct'])

        # Calculate average match score across all queries
        all_match_scores = []
        for stats in query_stats.values():
            for iter_result in stats['iterations']:
                if iter_result.get('match_score') is not None:
                    all_match_scores.append(iter_result['match_score'])

        avg_overall_match_score = sum(all_match_scores) / len(all_match_scores) if all_match_scores else 0.0

        logger.info("Neo4j Evaluation Results:")
        logger.info(f"Database used: {database_name}")
        logger.info(f"Matching method: Batch Fuzzy")
        logger.info(f"Fuzzy threshold: {config.fuzzy_threshold}")
        logger.info(f"Parallel processing: {config.max_workers} workers")
        logger.info(f"Successfully processed: {successful_queries} / {total_queries} queries")
        logger.info(f"Queries that returned results: {queries_with_results} / {total_queries}")
        logger.info(f"Correct results: {correct_queries} / {total_queries}")
        logger.info(f"Average match score: {avg_overall_match_score:.3f}")
        logger.info(f"Total elapsed time: {total_runtime_info.total_elapsed_time:.2f} seconds")
        logger.info(f"Estimated cost: ${total_runtime_info.cost_estimate:.4f}")

        # Create DataFrames for CSV output
        df_results = pd.DataFrame(results_data)

        # Debug: Check if DataFrame is empty or missing columns
        if df_results.empty:
            logger.warning("No results data to process, creating empty DataFrames")
            df_results = pd.DataFrame(columns=['iteration', 'query_idx', 'question', 'options', 'expected_answer',
                                              'generated_query', 'successful', 'returned_results', 'correct', 'match_score', 'results', 'error'])

        # Aggregated statistics DataFrame
        stats_data = []
        for query_idx, stats in query_stats.items():
            # Calculate average match score for this query across all iterations
            match_scores = [iter_result.get('match_score', 0.0) for iter_result in stats['iterations'] if iter_result.get('match_score') is not None]
            avg_match_score = sum(match_scores) / len(match_scores) if match_scores else 0.0

            stats_data.append({
                'query_idx': query_idx + 1,
                'question': stats['iterations'][0]['question'],
                'expected_answer': stats['iterations'][0]['expected_answer'],
                'successful_any_iteration': stats['successful'],
                'returned_results_any_iteration': stats['returned_results'],
                'correct_any_iteration': stats['correct'],
                'successful_iterations': sum(1 for iter_result in stats['iterations'] if iter_result['successful']),
                'results_iterations': sum(1 for iter_result in stats['iterations'] if iter_result['returned_results']),
                'correct_iterations': sum(1 for iter_result in stats['iterations'] if iter_result['correct']),
                'avg_match_score': avg_match_score,
                'max_match_score': max(match_scores) if match_scores else 0.0
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
            'matching_method': 'fuzzy',
            'fuzzy_threshold': config.fuzzy_threshold,
            'parallel_workers': config.max_workers,
            'successful_queries': successful_queries,
            'total_queries': total_queries,
            'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
            'queries_with_results': queries_with_results,
            'results_rate': queries_with_results / total_queries if total_queries > 0 else 0,
            'correct_queries': correct_queries,
            'accuracy': correct_queries / total_queries if total_queries > 0 else 0,
            'average_match_score': avg_overall_match_score,
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