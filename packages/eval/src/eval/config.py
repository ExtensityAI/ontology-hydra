from pydantic import BaseModel, ConfigDict
from eval.neo4j_eval import Neo4jConfig


class ModelConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    engine: str
    """The model engine to use (e.g., 'gpt-4.1')"""

    api_key: str
    """The API key for the model service"""


class EvalScenario(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str

    domain: str | None = None
    """The domain used for ontology creation. If None, no ontology will be created."""

    squad_titles: tuple[str, ...]
    """Titles of topics in the SQuAD dataset to use for evaluation (title field)"""

    neo4j: Neo4jConfig = Neo4jConfig()  # Add Neo4j config, default disabled

    dataset_mode: str = "test"  # "dev" or "test" - which dataset to use for evaluation

    skip_qa: bool = False  # Skip question answering evaluation (only do KG generation and Neo4j)


class EvalConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model: ModelConfig
    """Model configuration including engine and API key"""

    neo4j: Neo4jConfig = Neo4jConfig()
    """Global Neo4j configuration that can be overridden by scenario-specific settings"""

    scenarios: tuple[EvalScenario, ...]