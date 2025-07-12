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

class SemanticMappingInput(LLMDataModel):
    """Input for semantic option mapping"""
    options: List[str] = Field(description="List of answer options to map to KG entities")
    schema: str = Field(description="The schema of the Neo4j database")
    question_context: str = Field(description="The question context to help with mapping")


class SemanticMappingOutput(LLMDataModel):
    """Output for semantic option mapping"""
    mappings: List[Dict[str, str]] = Field(description="List of mappings from option text to KG entity names")


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
class SemanticOptionMapper(Expression):
    """Maps answer options to relevant KG entities/concepts using semantic understanding."""

    def __init__(
        self,
        seed: Optional[int] = 42,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.data_model = SemanticMappingOutput

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
        if not input.schema:
            raise ValueError("Schema cannot be empty!")
        return True

    def post(self, output: SemanticMappingOutput) -> bool:
        if not isinstance(output, SemanticMappingOutput):
            raise ValueError("Output must be a SemanticMappingOutput instance!")
        if len(output.mappings) != self.num_options:
            raise ValueError(f"Number of mappings ({len(output.mappings)}) does not match number of input options ({self.num_options})")
        return True

    def act(self, input: SemanticMappingInput, **kwargs) -> SemanticMappingInput:
        self.num_options = len(input.options)
        return input

    @property
    def prompt(self) -> str:
        return """[[Semantic Option Mapping]]
Map each answer option to the most relevant entity/concept in the knowledge graph schema. Do NOT treat the full option text as a node name. Instead, identify the core concept or entity that the option describes.

MAPPING GUIDELINES:
1. Extract the core concept from each option text
2. Find the most semantically similar entity in the schema
3. Use the exact entity name from the schema (with proper formatting)
4. If no exact match exists, choose the closest semantic match
5. Consider the question context when making mappings

FORMATTING RULES:
- All entity names must be lowercase with underscores instead of spaces
- Use exact names from the schema
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
- "mapped": the mapped entity name from the schema

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
        return """[[Neo4j Cypher Query Generation - Batch Processing with Semantic Mapping]]
Convert the given list of natural language questions into Neo4j Cypher queries using ONLY the schema elements provided. No other elements exist or can be used.

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
- Node labels and relationship types remain as provided in the schema

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
- Only use the exact node labels, relationship types, and properties listed in the schema
- ALL names in property values must be lowercase with underscores instead of spaces
- Any other elements will fail as they do not exist in the database
- Maintain consistent query structure across the batch for similar question types"""
