import openai

from ontology.personas import Persona
from ontology.utils import MODEL

expert_scope_document_system_prompt = """You are <persona>{persona}</persona>. Your task is to author a rigorous, standalone document meticulously detailing the scope of inquiry for <domain>{domain}</domain> from your own perspective. This document will serve as a reference for subsequently defining key concepts, terminologies, and relationships within this domain.

Your writing must adhere to the following criteria:

- Clearly and explicitly delineate the scope by specifying the exact areas, phenomena, or entities encompassed by the domain.
- Define precise boundaries by articulating both inclusions and explicit exclusions, emphasizing distinctions to prevent ambiguity.
- Address all critical dimensions relevant to the domain, ensuring comprehensiveness without redundancy.
- Maintain an objective, scholarly tone characterized by clarity, conciseness, and specificity.
- Structure the document systematically to enhance readability and facilitate future referencing.

Exclude the following from your document:

- Any introductory or concluding remarks.
"""


def generate_scope_document(domain: str, persona: Persona):
    """Generate a scope document for a given domain and persona. The generated document delineates the scope and boundaries of the ontology."""
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": expert_scope_document_system_prompt.format(
                    persona=persona.description,
                    domain=domain,
                ),
            }
        ],
    )

    if response.choices[0].message.content is None:
        raise ValueError("Failed to generate scope document")

    return response.choices[0].message.content
