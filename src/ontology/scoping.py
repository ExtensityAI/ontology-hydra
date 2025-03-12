import openai
from pydantic import BaseModel

from ontology.personas import Persona
from ontology.utils import MODEL

expert_scope_document_system_prompt = """You are <persona>{persona}</persona>, a recognized expert specializing in <domain>{domain}</domain>. Your task is to author a rigorous, standalone document meticulously detailing the scope of inquiry for <domain>{domain}</domain> from your personal perspective.

Your writing must adhere to the following criteria:

- Clearly delineate the scope by specifying the exact areas, phenomena, or entities encompassed by the domain.
- Define precise boundaries by explicitly noting inclusions and only non-obvious exclusions, clearly distinguishing areas that might otherwise cause ambiguity.
- Address all critical dimensions relevant to the domain, ensuring comprehensiveness without redundancy.
- Maintain an objective, scholarly tone characterized by clarity, conciseness, and specificity.
- Structure the document systematically to enhance readability and facilitate future referencing.
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


class ScopeDocument(BaseModel):
    author: Persona
    content: str


merge_documents_prompt = """You are an expert in <domain>{domain}</domain>. Your task is to merge the provided scope documents into a single, coherent, clear, and exhaustive document. Each original document includes a persona description of its author; you must assume all provided personas when merging the documents. The merged document must:

- Retain all essential information from each original document without adding any external information.
- Reflect the perspectives and expertise of all provided personas.
- Eliminate redundancies and maintain clarity and logical flow.
- Ensure coherence and readability throughout.

Do not include any information that is not explicitly present in the original scope documents."""

# TODO
# TODO merge documents in chunks - merging all at once is too much for the model!
# TODO


def merge_scope_documents(domain: str, documents: list[ScopeDocument]):
    # sample into groups of 4 (or more), merge, then resample and merge until one remains
    # use approach leo said: tell model to behave as all personas at the same time

    document_messages: list = [
        {
            "role": "user",
            "content": f"<persona>{doc.author.description}</persona>\n\n{doc.content}",
        }
        for doc in documents
    ]

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": merge_documents_prompt.format(
                    domain=domain,
                ),
            },
            *document_messages,
        ],
    )

    if response.choices[0].message.content is None:
        raise ValueError("Failed to merge scope documents")

    return response.choices[0].message.content
