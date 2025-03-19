import openai

from ontology.groups import Group
from ontology.personas import Persona
from ontology.utils import MODEL

generate_questions_prompt = """
You are <persona>{persona}</persona>, a recognized expert specializing in {group}. You have been presented with a scope document for an ontology in the domain of <domain>{domain}</domain>.

## Your Task
Based on your expertise and the provided scope document, generate a list of questions that you personally would want answered by a comprehensive ontology in this domain.

## Guidelines for Question Generation
1. Focus on questions that are important to you as an expert in this field
2. Consider questions about:
   * Key concepts and their definitions
   * Important distinctions between related terms
   * Essential classifications or categorizations
   * Critical attributes or properties
   * Domain-specific constraints or rules

## Output Format
1. Organize your questions into numbered sections by topic area
2. Use bullet points for individual questions
3. Ensure each question is specific and concrete
4. Phrase questions to elicit detailed, precise answers

Generate only questions that you, as this specific expert persona, would consider relevant and important for understanding the domain. Do not include questions outside your area of expertise.

<scope_document>{scope_document}</scope_document>
"""


def generate_questions(
    domain: str, persona: Persona, group: Group, scope_document: str
):
    """Generate questions for a given domain and persona. The generated questions are based on the provided scope document."""
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": generate_questions_prompt.format(
                    persona=persona.description,
                    domain=domain,
                    group=f"{group.name} ({group.description})",
                    scope_document=scope_document,
                ),
            }
        ],
    )

    if response.choices[0].message.content is None:
        raise ValueError("Failed to generate questions")

    return response.choices[0].message.content.strip()
