import openai
from pydantic import BaseModel

from ontology.utils import MODEL

parse_concepts_prompt = """You are an ontology engineer tasked with extracting a comprehensive conceptual structure from a scope document about <domain>{domain}</domain>.

Your task:
Analyze the provided scope document and identify both key concepts and their instances:

For concepts (classes or categories):
1. Provide a clear, concise name
2. Write a detailed description that captures its meaning and relevance in the context of the domain
3. Identify its hierarchical relationship to other concepts (parent-child relationships)

For instances (specific examples or individuals of a concept):
1. Provide a distinctive name
2. Write a detailed description of this specific instance
3. Identify which concept this is an instance of

Guidelines for distinguishing concepts from instances:
- Concepts represent general categories, types, or classes of things
- Instances represent specific examples, individuals, or concrete manifestations of concepts
- The distinction depends on the domain context (e.g., "Bird" is a concept while "Bald Eagle" might be a concept or an instance depending on the domain)
- When ambiguous, use your best judgment based on the domain and scope document context

Organize the results in a hierarchical tree structure where:
- More general concepts are positioned as parents
- More specific concepts are positioned as their children
- Instances are attached to their respective concepts
- Related elements are grouped logically together

The output should be a comprehensive conceptual model that captures both the knowledge structure and specific examples from the <domain>{domain}</domain> domain as presented in the scope document."""


class Instance(BaseModel):
    """An instance of a concept."""

    name: str
    description: str


class Concept(BaseModel):
    """A concept in the ontology."""

    name: str
    description: str

    instances: list[Instance]

    children: list["Concept"]


def parse_concepts(domain: str, scope_document: str) -> Concept:
    """Parse concepts from the scope document."""
    response = openai.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": parse_concepts_prompt.format(domain=domain),
            },
            {
                "role": "user",
                "content": scope_document,
            },
        ],
        response_format=Concept,
    )

    obj = response.choices[0].message.parsed

    if obj is None:
        raise ValueError("Failed to parse concepts")

    return obj
