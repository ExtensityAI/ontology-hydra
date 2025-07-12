import json
import re
import time
from collections import defaultdict
from math import ceil
from pathlib import Path
from typing import List, Optional, Dict, Tuple
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


class SemanticMappingInput(LLMDataModel):
    """Input for semantic option mapping"""
    options: List[str] = Field(description="List of answer options to map to KG entities")
    graph_data: str = Field(description="The complete graph data from the Neo4j database")
    question_context: str = Field(description="The question context to help with mapping")


class SemanticMappingOutput(LLMDataModel):
    """Output for semantic option mapping"""
    mappings: List[Dict[str, str]] = Field(description="List of mappings from option text to KG entity names")


@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=False,
    remedy_retry_params=dict(
        tries=25,
        delay=0.5,
        max_delay=10,
        jitter=0.1,
        backoff=2,
        graceful=False
    )
)
class SemanticOptionMapper(Expression):
    """Maps answer options to relevant KG entities/concepts using semantic understanding."""

    def __init__(
        self,
        driver: GraphDatabase.driver = None,
        database_name: str = "neo4j",
        seed: Optional[int] = 42,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.data_model = SemanticMappingOutput
        self.driver = driver
        self.database_name = database_name

    def forward(self, input: SemanticMappingInput) -> SemanticMappingOutput:
        if self.contract_result is None:
            # Return empty mappings if contract fails
            return SemanticMappingOutput(mappings=[{"original": opt, "mapped": ""} for opt in input.options])
        return self.contract_result

    def pre(self, input: SemanticMappingInput) -> bool:
        if not isinstance(input, SemanticMappingInput):
            raise ValueError("Input must be a SemanticMappingInput instance!")
        if not input.options or len(input.options) == 0:
            raise ValueError("Options list cannot be empty!")
        if not input.graph_data:
            raise ValueError("Graph data cannot be empty!")
        return True

    def post(self, output: SemanticMappingOutput) -> bool:
        if not isinstance(output, SemanticMappingOutput):
            raise ValueError("Output must be a SemanticMappingOutput instance!")
        if len(output.mappings) != self.num_options:
            raise ValueError(f"Number of mappings ({len(output.mappings)}) does not match number of input options ({self.num_options})")

        # Validate mappings against actual graph data if driver is available
        if self.driver is not None:
            try:
                # Get actual node names from the graph
                with self.driver.session(database=self.database_name) as session:
                    nodes_result = session.run("MATCH (n) RETURN DISTINCT n.name as name")
                    node_names = {record["name"] for record in nodes_result}

                # Validate that mapped entities exist in the graph
                for i, mapping in enumerate(output.mappings):
                    if mapping.get("mapped") and mapping["mapped"].strip():
                        mapped_entity = mapping["mapped"].strip()
                        if mapped_entity not in node_names:
                            logger.warning(f"Semantic mapping {i+1}: Mapped entity '{mapped_entity}' does not exist in the graph. Available entities: {sorted(list(node_names)[:10])}...")
                            # Don't fail validation, just warn - the model might have made a reasonable semantic guess

            except Exception as e:
                logger.warning(f"Could not validate semantic mappings against graph data: {e}")
                # Don't fail validation if we can't access the database

        return True

    def act(self, input: SemanticMappingInput, **kwargs) -> SemanticMappingInput:
        self.num_options = len(input.options)
        return input

    @property
    def prompt(self) -> str:
        return """[[Semantic Option Mapping]]
Map each answer option to the most relevant entity/concept in the knowledge graph data. Do NOT treat the full option text as a node name. Instead, identify the core concept or entity that the option describes.

MAPPING GUIDELINES:
1. Extract the core concept from each option text
2. Find the most semantically similar entity in the graph data
3. Use the exact entity name from the graph data (with proper formatting)
4. If no exact match exists, choose the closest semantic match
5. Consider the question context when making mappings

FORMATTING RULES:
- All entity names must be lowercase with underscores instead of spaces
- Use exact names from the graph data
- If an option describes multiple concepts, map to the primary one

EXAMPLES:
Option: "A protein that helps carry oxygen in red blood cells and gives them their red color"
Mapped to: "hemoglobin" (not the full option text)

Option: "A type of white blood cell that produces antibodies to fight infection"
Mapped to: "b_lymphocyte"

Option: "A hormone produced by the pancreas that regulates blood sugar levels"
Mapped to: "insulin"

Return a list of mappings where each mapping contains:
- "original": the original option text
- "mapped": the mapped entity name from the graph data

If no suitable mapping is found, use an empty string for "mapped"."""


class BatchAnswerInput(LLMDataModel):
    """Input for batch answer matching"""
    answer_pairs: List[dict] = Field(description="List of answer pairs to match, each containing 'predicted_answer' and 'expected_answer'")

class BatchAnswerOutput(LLMDataModel):
    """Output for batch answer matching"""
    match_scores: List[float] = Field(description="List of similarity scores between predicted and expected answers")

@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=False,
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
    graph_data: str = Field(description="The complete graph data from the Neo4j database")
    semantic_mappings: Optional[List[List[Dict[str, str]]]] = Field(
        description="List of semantic mappings for each question's options",
        default=None
    )


class Neo4jQueryOutput(LLMDataModel):
    """Output containing the generated Cypher queries"""
    queries: List[str] = Field(description="List of generated Cypher queries")

@contract(
    pre_remedy=False,
    post_remedy=True,
    verbose=False,
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

    def get_graph_data(self, database_name: str = "neo4j") -> str:
        """Get complete graph data using Neo4j built-in procedures."""
        return _get_neo4j_graph_data(self.driver, database_name)

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
            # Get complete graph data for validation
            graph_data = _get_neo4j_graph_data(self.driver, self.database_name)

            # Extract actual node labels, relationship types, and node names from the graph data
            with self.driver.session(database=self.database_name) as session:
                # Get all node labels and their counts
                labels_result = session.run("CALL db.labels() YIELD label RETURN label")
                labels = {record["label"] for record in labels_result}

                # Get all relationship types
                rels_result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
                rels = {record["relationshipType"] for record in rels_result}

                # Get all node names from the actual graph
                nodes_result = session.run("MATCH (n) RETURN DISTINCT n.name as name, labels(n) as labels")
                node_names = set()
                node_labels_map = {}
                for record in nodes_result:
                    name = record["name"]
                    node_labels = record["labels"]
                    node_names.add(name)
                    node_labels_map[name] = node_labels

                # Get all property keys
                props_result = session.run("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey")
                props = {record["propertyKey"] for record in props_result}

        except Exception as e:
            raise ValueError(f"Failed to get graph data from Neo4j database '{self.database_name}': {e}")

        # Debug: Log available graph elements
        logger.debug(f"Available labels in database '{self.database_name}': {sorted(labels)}")
        logger.debug(f"Available relationships in database '{self.database_name}': {sorted(rels)}")
        logger.debug(f"Available node names in database '{self.database_name}': {sorted(node_names)}")
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

            # Extract name values from property patterns and WHERE clauses and validate against actual node names
            pattern1 = r'(\{[^}]*name\s*:\s*[\'"`])([^\'"`]+)([\'"`][^}]*\})'
            pattern2 = r'(WHERE\s+[a-zA-Z_][a-zA-Z0-9_]*\.name\s*=\s*[\'"`])([^\'"`]+)([\'"`])'

            # Check name values for proper formatting and existence
            for pattern in [pattern1, pattern2]:
                matches = re.finditer(pattern, query)
                for match in matches:
                    name_value = match.group(2)
                    if any(c.isupper() for c in name_value) or ' ' in name_value:
                        raise ValueError(f"Name value '{name_value}' must be lowercase and have underscores instead of spaces")

                    # Check if the node name actually exists in the graph
                    if name_value not in node_names:
                        raise ValueError(f"Node name '{name_value}' does not exist in the graph. Available node names: {sorted(list(node_names)[:20])}...")

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
        return """[[Neo4j Cypher Query Generation - Batch Processing with Semantic Mapping]]
Convert the given list of natural language questions into Neo4j Cypher queries using ONLY the graph data elements provided. No other elements exist or can be used.

CRITICAL SEMANTIC MAPPING INSTRUCTIONS:
When generating Cypher queries, do NOT treat full answer option text as node names. Instead, use the provided semantic mappings to map options to relevant KG entities/concepts.

SEMANTIC MAPPING APPROACH:
- Each option has been pre-mapped to the most relevant entity in the knowledge graph
- Use the mapped entity names (not the original option text) in your queries
- The mappings are provided in the semantic_mappings field
- If no mapping is available, use semantic understanding to find the best match

IMPORTANT NAME FORMATTING RULES:
- ALL names in property values must be lowercase with underscores instead of spaces
- Examples:
  * "Magnetic Resonance Imaging" becomes "magnetic_resonance_imaging"
  * "Robotic Surgical System" becomes "robotic_surgical_system"
  * "Ultrasound Guided Biopsy" becomes "ultrasound_guided_biopsy"
  * "Dermatological Surgery" becomes "dermatological_surgery"
- This applies to all {name: 'value'} patterns in the queries and (name = 'value') patterns
- Node labels and relationship types remain as provided in the graph data

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

SEMANTIC MAPPING EXAMPLES:
Question: "What device is used to measure temperature differences?"
Options: ["A device made up of the junction of two different metals...", "A device that deforms when voltage is applied"]
Mappings: [{"original": "A device made up of the junction...", "mapped": "thermocouple"}, {"original": "A device that deforms...", "mapped": "piezoelectric_device"}]

Instead of using the full option text, use:
MATCH (d:Entity {name: 'thermocouple'}) RETURN d.name as device_name

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
- Use semantic mappings when available, not full option text
- Only use the exact node labels, relationship types, and properties listed in the graph data
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
    enable_semantic_mapping: bool = True  # New field to enable/disable semantic mapping
    semantic_mapping_batch_size: int = 10  # Batch size for semantic mapping operations
    neo4j_import_dir: str  # Neo4j import directory path for CSV file loading

    def __init__(self, **data):
        # Allow environment variables to override defaults
        import os

        # Available environment variables:
        # NEO4J_URI - Neo4j connection URI
        # NEO4J_USER - Neo4j username
        # NEO4J_PASSWORD - Neo4j password
        # NEO4J_USE_RUN_SPECIFIC_DATABASES - Enable/disable run-specific databases
        # NEO4J_DEFAULT_DATABASE - Default database name
        # NEO4J_AUTO_CLEANUP - Enable/disable automatic database cleanup
        # NEO4J_FUZZY_THRESHOLD - Threshold for fuzzy matching
        # NEO4J_MAX_WORKERS - Number of parallel workers
        # NEO4J_ENABLE_SEMANTIC_MAPPING - Enable/disable semantic mapping
        # NEO4J_SEMANTIC_MAPPING_BATCH_SIZE - Batch size for semantic mapping
        # NEO4J_IMPORT_DIR - Neo4j import directory path for CSV file loading

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
        if 'NEO4J_ENABLE_SEMANTIC_MAPPING' in os.environ:
            data['enable_semantic_mapping'] = os.environ['NEO4J_ENABLE_SEMANTIC_MAPPING'].lower() == 'true'
        if 'NEO4J_SEMANTIC_MAPPING_BATCH_SIZE' in os.environ:
            data['semantic_mapping_batch_size'] = int(os.environ['NEO4J_SEMANTIC_MAPPING_BATCH_SIZE'])
        if 'NEO4J_IMPORT_DIR' in os.environ:
            data['neo4j_import_dir'] = os.environ['NEO4J_IMPORT_DIR']

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


def _load_kg_to_neo4j(kg: KG, driver: GraphDatabase.driver, config: Neo4jConfig, database_name: str = "neo4j"):
    """Load knowledge graph into Neo4j database using improved loader"""
    logger.info(f"Loading knowledge graph into Neo4j database: {database_name}")

    import time
    import json
    import csv
    import os
    import sys
    from pathlib import Path

    # Add the src/evaluation directory to the path to import the improved loader
    src_eval_path = Path(__file__).parent.parent.parent.parent / "src" / "evaluation"
    if src_eval_path.exists():
        sys.path.insert(0, str(src_eval_path))
        try:
            from neo4j_graph_loader_v2 import Neo4jKGLoader
            logger.info("Using improved Neo4jKGLoader with node type preservation")

            # Create a temporary JSON file from the KG object
            temp_kg_file = f"/tmp/kg_{database_name}_{int(time.time())}.json"

            # Convert KG object to JSON format
            kg_data = {
                "name": "KnowledgeGraph",
                "triplets": [
                    {
                        "subject": t.subject,
                        "predicate": t.predicate,
                        "object": t.object
                    }
                    for t in kg.triplets if t is not None
                ]
            }

            with open(temp_kg_file, 'w') as f:
                json.dump(kg_data, f, indent=2)

            try:
                # Use the improved loader
                loader = Neo4jKGLoader(
                    uri=config.uri,  # Use the config URI instead of trying to construct it
                    user=config.user,
                    password=config.password,
                    neo4j_import_dir=config.neo4j_import_dir  # Use config import directory
                )

                # Load the knowledge graph
                loader.load_knowledge_graph(temp_kg_file)
                logger.info(f"Successfully loaded knowledge graph using improved loader")

            finally:
                # Clean up temporary file
                if os.path.exists(temp_kg_file):
                    os.remove(temp_kg_file)

        except ImportError as e:
            logger.warning(f"Could not import improved loader: {e}")
            logger.info("Falling back to basic CSV method")
            _load_kg_to_neo4j_basic(kg, driver, config, database_name)
        except Exception as e:
            logger.warning(f"Error using improved loader: {e}")
            logger.info("Falling back to basic CSV method")
            _load_kg_to_neo4j_basic(kg, driver, config, database_name)
    else:
        logger.warning(f"Could not find src/evaluation directory: {src_eval_path}")
        logger.info("Falling back to basic CSV method")
        _load_kg_to_neo4j_basic(kg, driver, config, database_name)


def _load_kg_to_neo4j_basic(kg: KG, driver: GraphDatabase.driver, config: Neo4jConfig, database_name: str = "neo4j"):
    """Load knowledge graph into Neo4j database using basic CSV method (fallback)"""
    logger.info(f"Loading knowledge graph into Neo4j database using basic method: {database_name}")

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

        # Neo4j import folder path - must be provided in config
    neo4j_import_dir = config.neo4j_import_dir
    logger.info(f"Using Neo4j import directory: {neo4j_import_dir}")

    # Validate that the import directory exists and is writable
    if not os.path.exists(neo4j_import_dir):
        raise FileNotFoundError(f"Neo4j import directory does not exist: {neo4j_import_dir}")

    if not os.access(neo4j_import_dir, os.W_OK):
        raise PermissionError(f"Cannot write to Neo4j import directory: {neo4j_import_dir}")

    # Convert KG to CSV format
    triplets = kg.triplets

    # Collect unique node names
    nodes = set()
    for t in triplets:
        if t is not None:
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
            if t is not None:
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

    logger.info(f"Successfully loaded knowledge graph into Neo4j database: {database_name} using basic CSV method")


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


def _get_neo4j_graph_data(driver: GraphDatabase.driver, database_name: str) -> str:
    """Get actual graph data (nodes and relationships) from Neo4j database."""
    with driver.session(database=database_name) as session:
        try:
            # Get all nodes and relationships
            result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m")

            # Collect all graph data
            graph_data = []
            node_labels = set()
            relationship_types = set()
            node_properties = set()

            for record in result:
                source_node = record["n"]
                relationship = record["r"]
                target_node = record["m"]

                # Extract node information
                source_labels = list(source_node.labels)
                target_labels = list(target_node.labels)
                node_labels.update(source_labels)
                node_labels.update(target_labels)

                # Extract relationship information
                rel_type = relationship.type
                relationship_types.add(rel_type)

                # Extract properties
                source_props = dict(source_node)
                target_props = dict(target_node)
                rel_props = dict(relationship)

                node_properties.update(source_props.keys())
                node_properties.update(target_props.keys())

                # Store the relationship data
                graph_data.append({
                    "source_node": {
                        "labels": source_labels,
                        "properties": source_props
                    },
                    "relationship": {
                        "type": rel_type,
                        "properties": rel_props
                    },
                    "target_node": {
                        "labels": target_labels,
                        "properties": target_props
                    }
                })

            # Format the graph data as a comprehensive description
            graph_description = f"Complete Neo4j Graph Data (Database: {database_name}):\n\n"

            # Summary statistics
            graph_description += f"Graph Statistics:\n"
            graph_description += f"- Total relationships: {len(graph_data)}\n"
            graph_description += f"- Node labels: {sorted(node_labels)}\n"
            graph_description += f"- Relationship types: {sorted(relationship_types)}\n"
            graph_description += f"- Node properties: {sorted(node_properties)}\n\n"

            # Complete graph structure - ALL relationships
            graph_description += "Complete Graph Structure:\n"
            graph_description += "The knowledge graph contains the following relationships:\n\n"

            # Group relationships by type for better organization
            rel_groups = {}
            for item in graph_data:
                rel_type = item["relationship"]["type"]
                if rel_type not in rel_groups:
                    rel_groups[rel_type] = []
                rel_groups[rel_type].append(item)

            for rel_type in sorted(rel_groups.keys()):
                graph_description += f"Relationship Type: {rel_type}\n"
                graph_description += f"Count: {len(rel_groups[rel_type])}\n"
                graph_description += "All relationships:\n"

                # Show ALL relationships of this type, not just examples
                for i, item in enumerate(rel_groups[rel_type]):
                    source_name = item["source_node"]["properties"].get("name", "unnamed")
                    target_name = item["target_node"]["properties"].get("name", "unnamed")
                    source_labels = ":".join(item["source_node"]["labels"])
                    target_labels = ":".join(item["target_node"]["labels"])

                    graph_description += f"  {i+1}. ({source_labels} {{name: '{source_name}'}})-[:{rel_type}]->({target_labels} {{name: '{target_name}'}})\n"

                graph_description += "\n"

            # Complete node information
            graph_description += "Complete Node Information:\n"
            for label in sorted(node_labels):
                # Count nodes with this label
                count_result = session.run(f"MATCH (n:{label}) RETURN count(n) as count")
                count = count_result.single()["count"]
                graph_description += f"- {label}: {count} nodes\n"

                # Show ALL nodes with this label, not just examples
                examples_result = session.run(f"MATCH (n:{label}) RETURN n.name as name ORDER BY n.name")
                examples = [record["name"] for record in examples_result if record["name"]]
                if examples:
                    graph_description += f"  All nodes: {', '.join(examples)}\n"
                graph_description += "\n"

            return graph_description

        except Exception as e:
            logger.warning(f"Could not get graph data: {e}")
            # Fallback to basic schema if graph data retrieval fails
            return _get_neo4j_schema_fallback(driver, database_name)


def _get_neo4j_schema_fallback(driver: GraphDatabase.driver, database_name: str) -> str:
    """Fallback method to get basic schema information if graph data retrieval fails."""
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


def _perform_semantic_mapping(questions: List[str], options: List[List[str]], graph_data: str, config: Neo4jConfig, driver: GraphDatabase.driver = None, database_name: str = "neo4j") -> List[List[Dict[str, str]]]:
    """Perform semantic mapping for all questions and their options.

    Args:
        questions: List of questions
        options: List of option lists for each question
        graph_data: Complete graph data from Neo4j database
        config: Neo4jConfig instance
        driver: Neo4j driver for validation (optional)
        database_name: Database name for validation (optional)

    Returns:
        List of semantic mappings for each question's options
    """
    if not config.enable_semantic_mapping:
        logger.info("Semantic mapping disabled, using empty mappings")
        return [[{"original": opt, "mapped": ""} for opt in question_options] for question_options in options]

    logger.info("Performing semantic mapping for answer options...")

    semantic_mapper = SemanticOptionMapper(driver=driver, database_name=database_name)
    all_mappings = []

    # Process in batches to avoid overwhelming the LLM
    batch_size = config.semantic_mapping_batch_size
    num_batches = ceil(len(questions) / batch_size)

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(questions))

        batch_questions = questions[start_idx:end_idx]
        batch_options = options[start_idx:end_idx]

        logger.debug(f"Processing semantic mapping batch {batch_idx + 1}/{num_batches} (questions {start_idx + 1}-{end_idx})")

        for i, (question, question_options) in enumerate(zip(batch_questions, batch_options)):
            try:
                # Create input for semantic mapping
                mapping_input = SemanticMappingInput(
                    options=question_options,
                    graph_data=graph_data,
                    question_context=question
                )

                # Perform semantic mapping
                mapping_output = semantic_mapper(input=mapping_input)

                # Store mappings for this question
                all_mappings.append(mapping_output.mappings)

                logger.debug(f"Semantic mapping for question {start_idx + i + 1}: {len(question_options)} options mapped")

            except Exception as e:
                logger.warning(f"Failed to perform semantic mapping for question {start_idx + i + 1}: {e}")
                # Create empty mappings as fallback
                fallback_mappings = [{"original": opt, "mapped": ""} for opt in question_options]
                all_mappings.append(fallback_mappings)

    logger.info(f"Completed semantic mapping for {len(questions)} questions")
    return all_mappings


def _process_iteration_parallel(iteration_data, converter, batch_fuzzy_matcher, driver, database_name, config, graph_data):
    """Process a single iteration of all questions sequentially.

    Args:
        iteration_data: Tuple of (iteration, all_questions, all_options, all_answers)
        converter: BatchQuestionToCypherConverter instance
        batch_fuzzy_matcher: BatchFuzzyAnswerMatcher instance
        driver: Neo4j driver
        database_name: Database name to use
        config: Neo4jConfig instance
        graph_data: Complete graph data from Neo4j database

    Returns:
        Tuple of (iteration_results, iteration_runtime_info)
    """
    iteration, all_questions, all_options, all_answers = iteration_data

    # Initialize iteration runtime tracking
    iteration_runtime_info = RuntimeInfo(0, 0, 0, 0, 0, 0, 0, 0)
    iteration_results = []

    logger.info(f"Processing Neo4j iteration {iteration + 1}/{config.num_iterations}")

    try:
        # Process questions in batches sequentially within this iteration
        num_batches = ceil(len(all_questions) / config.batch_size)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * config.batch_size
            end_idx = min((batch_idx + 1) * config.batch_size, len(all_questions))

            batch_questions = all_questions[start_idx:end_idx]
            batch_options = all_options[start_idx:end_idx]
            batch_answers = all_answers[start_idx:end_idx]

            logger.debug(f"Processing batch {batch_idx + 1}/{num_batches} in iteration {iteration + 1}")

            # Process this batch
            batch_results, batch_runtime_info = _process_batch_parallel(
                (batch_questions, batch_options, batch_answers),
                converter,
                batch_fuzzy_matcher,
                driver,
                database_name,
                config,
                iteration,
                start_idx,
                graph_data
            )

            # Add batch results and runtime to iteration totals
            iteration_results.extend(batch_results)
            iteration_runtime_info += batch_runtime_info

            logger.debug(f"Completed batch {batch_idx + 1}/{num_batches} in iteration {iteration + 1}")

    except Exception as e:
        logger.error(f"Error processing iteration {iteration + 1}: {e}")

        # Mark all queries in this iteration as failed
        for i in range(len(all_questions)):
            iteration_result = {
                'iteration': iteration + 1,
                'query_idx': i + 1,
                'question': all_questions[i],
                'options': all_options[i],
                'expected_answer': all_answers[i],
                'generated_query': "",
                'semantic_mappings': [],
                'successful': False,
                'returned_results': False,
                'correct': False,
                'results': [],
                'error': f"Iteration processing error: {type(e).__name__}: {str(e)}"
            }
            iteration_results.append(iteration_result)

    return iteration_results, iteration_runtime_info


def _process_batch_parallel(batch_data, converter, batch_fuzzy_matcher, driver, database_name, config, iteration, start_idx, graph_data):
    """Process a single batch of questions sequentially within an iteration.

    Args:
        batch_data: Tuple of (batch_questions, batch_options, batch_answers)
        converter: BatchQuestionToCypherConverter instance
        batch_fuzzy_matcher: BatchFuzzyAnswerMatcher instance
        driver: Neo4j driver
        database_name: Database name to use
        config: Neo4jConfig instance
        iteration: Current iteration number
        start_idx: Starting index for this batch
        graph_data: Complete graph data from Neo4j database

    Returns:
        Tuple of (batch_results, batch_runtime_info)
    """
    batch_questions, batch_options, batch_answers = batch_data

    # Initialize batch runtime tracking
    batch_runtime_info = RuntimeInfo(0, 0, 0, 0, 0, 0, 0, 0)
    batch_iteration_results = []

    try:
        # Perform semantic mapping for this batch (if enabled)
        batch_semantic_mappings = _perform_semantic_mapping(batch_questions, batch_options, graph_data, config, driver, database_name)
        # Convert questions to queries using contract with semantic mappings
        input_data = Neo4jQueryInput(
            questions=batch_questions,
            options=batch_options,
            answers=batch_answers,
            graph_data=graph_data,
            semantic_mappings=batch_semantic_mappings
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
                'semantic_mappings': batch_semantic_mappings[i] if i < len(batch_semantic_mappings) else [],
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
                'semantic_mappings': [],
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
        _load_kg_to_neo4j(kg, driver, config, database_name)

        # Initialize converter with the driver (it will use the database_name parameter in queries)
        converter = BatchQuestionToCypherConverter(driver=driver, database_name=database_name)

        # Initialize batch fuzzy answer matcher
        batch_fuzzy_matcher = BatchFuzzyAnswerMatcher(threshold=config.fuzzy_threshold)
        logger.info(f"Batch fuzzy matching enabled with threshold: {config.fuzzy_threshold}")

        # Get complete graph data using Neo4j built-in functions
        graph_data = _get_neo4j_graph_data(driver, database_name)

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

        # Process iterations in parallel instead of batches within iterations
        logger.info(f"Processing {config.num_iterations} iterations with up to {config.max_workers} parallel workers")

        # Prepare iteration data for parallel processing
        iteration_tasks = []
        for iteration in range(config.num_iterations):
            iteration_tasks.append((iteration, all_questions, all_options, all_answers))

        # Process iterations in parallel
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            # Submit all iteration tasks
            future_to_iteration = {
                executor.submit(
                    _process_iteration_parallel,
                    iteration_data,
                    converter,
                    batch_fuzzy_matcher,
                    driver,
                    database_name,
                    config,
                    graph_data
                ): iteration_data for iteration_data in iteration_tasks
            }

            # Collect results as they complete
            for future in as_completed(future_to_iteration):
                iteration_data = future_to_iteration[future]
                iteration = iteration_data[0]

                try:
                    iteration_results, iteration_runtime_info = future.result()

                    # Add iteration runtime to total
                    total_runtime_info += iteration_runtime_info

                    # Store all iteration results
                    for iteration_result in iteration_results:
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

                    logger.debug(f"Completed Neo4j iteration {iteration + 1}/{config.num_iterations}")

                except Exception as e:
                    logger.error(f"Error processing Neo4j iteration {iteration + 1}: {e}")

                    # Mark all queries in this iteration as failed
                    iteration_questions, iteration_options, iteration_answers = iteration_data[1:]
                    for i in range(len(iteration_questions)):
                        iteration_result = {
                            'iteration': iteration + 1,
                            'query_idx': i + 1,
                            'question': iteration_questions[i],
                            'options': iteration_options[i],
                            'expected_answer': iteration_answers[i],
                            'generated_query': "",
                            'semantic_mappings': [],
                            'successful': False,
                            'returned_results': False,
                            'correct': False,
                            'results': [],
                            'error': f"Iteration processing error: {type(e).__name__}: {str(e)}"
                        }
                        results_data.append(iteration_result)
                        query_stats[i]['iterations'].append(iteration_result)

        # Calculate aggregated statistics based on unique queries
        unique_queries_count = len(query_stats)  # Number of unique queries
        successful_queries = sum(1 for stats in query_stats.values() if stats['successful'])
        queries_with_results = sum(1 for stats in query_stats.values() if stats['returned_results'])
        correct_queries = sum(1 for stats in query_stats.values() if stats['correct'])

        # Calculate average match score across all query-iteration combinations
        all_match_scores = []
        for stats in query_stats.values():
            for iter_result in stats['iterations']:
                if iter_result.get('match_score') is not None:
                    all_match_scores.append(iter_result['match_score'])

        avg_overall_match_score = sum(all_match_scores) / len(all_match_scores) if all_match_scores else 0.0

        logger.info("Neo4j Evaluation Results:")
        logger.info(f"Database used: {database_name}")
        logger.info(f"Semantic mapping: {'Enabled' if config.enable_semantic_mapping else 'Disabled'}")
        logger.info(f"Matching method: Batch Fuzzy")
        logger.info(f"Fuzzy threshold: {config.fuzzy_threshold}")
        logger.info(f"Parallel processing: {config.max_workers} workers")
        logger.info(f"Unique queries processed: {unique_queries_count}")
        logger.info(f"Total iterations: {config.num_iterations}")
        logger.info(f"Total query-iteration combinations: {total_queries}")
        logger.info(f"Queries successfully executed (any iteration): {successful_queries} / {unique_queries_count}")
        logger.info(f"Queries that returned results (any iteration): {queries_with_results} / {unique_queries_count}")
        logger.info(f"Queries with correct answers (any iteration): {correct_queries} / {unique_queries_count}")
        logger.info(f"Average match score: {avg_overall_match_score:.3f}")
        logger.info(f"Total elapsed time: {total_runtime_info.total_elapsed_time:.2f} seconds")
        logger.info(f"Estimated cost: ${total_runtime_info.cost_estimate:.4f}")

        # Create DataFrames for CSV output
        df_results = pd.DataFrame(results_data)

        # Debug: Check if DataFrame is empty or missing columns
        if df_results.empty:
            logger.warning("No results data to process, creating empty DataFrames")
            df_results = pd.DataFrame(columns=['iteration', 'query_idx', 'question', 'options', 'expected_answer',
                                              'generated_query', 'semantic_mappings', 'successful', 'returned_results', 'correct', 'match_score', 'results', 'error'])

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

        # Add total row - show unique query counts (same as final metrics)
        total_queries = len(query_stats)  # Number of unique queries

        iteration_stats_data.append({
            'iteration': 'TOTAL',
            'successful_count': successful_queries,  # Unique queries that succeeded in any iteration
            'results_count': queries_with_results,   # Unique queries that returned results in any iteration
            'correct_count': correct_queries,        # Unique queries that were correct in any iteration
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

        # Save the Neo4j graph data for accountability
        graph_data_file = neo4j_output_path / "neo4j_graph_data.txt"
        with open(graph_data_file, 'w') as f:
            f.write(graph_data)
        logger.info(f"Neo4j graph data saved to: {graph_data_file}")

        # Save summary metrics with clear documentation
        neo4j_metrics = {
            'database_name': database_name,
            'run_id': run_id,
            'semantic_mapping_enabled': config.enable_semantic_mapping,
            'semantic_mapping_batch_size': config.semantic_mapping_batch_size,
            'matching_method': 'fuzzy',
            'fuzzy_threshold': config.fuzzy_threshold,
            'parallel_workers': config.max_workers,
            'total_queries': unique_queries_count,  # Number of unique queries
            'successful_queries': successful_queries,  # Queries that executed successfully in at least one iteration
            'success_rate': successful_queries / unique_queries_count if unique_queries_count > 0 else 0,
            'queries_with_results': queries_with_results,  # Queries that returned results in at least one iteration
            'results_rate': queries_with_results / unique_queries_count if unique_queries_count > 0 else 0,
            'correct_queries': correct_queries,  # Queries with correct answers in at least one iteration
            'accuracy': correct_queries / unique_queries_count if unique_queries_count > 0 else 0,
            'average_match_score': avg_overall_match_score,
            'total_elapsed_time_seconds': total_runtime_info.total_elapsed_time,
            'estimated_cost_usd': total_runtime_info.cost_estimate,
            'auto_cleanup_enabled': config.auto_cleanup,
            'metrics_explanation': {
                'total_queries': 'Number of unique questions/queries processed',
                'successful_queries': 'Number of unique queries that executed successfully in at least one iteration',
                'queries_with_results': 'Number of unique queries that returned results in at least one iteration',
                'correct_queries': 'Number of unique queries with correct answers in at least one iteration',
                'success_rate': 'Percentage of unique queries that executed successfully at least once',
                'results_rate': 'Percentage of unique queries that returned results at least once',
                'accuracy': 'Percentage of unique queries with correct answers at least once',
                'average_match_score': 'Average fuzzy match score across all query-iteration combinations'
            }
        }

        metrics_file = neo4j_output_path / "neo4j_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(neo4j_metrics, f, indent=2)
        logger.info(f"Neo4j metrics saved to: {metrics_file}")

    finally:
        driver.close()