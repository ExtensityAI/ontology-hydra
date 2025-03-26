import openai

from ontology.personas import Persona
from ontology.utils import MODEL

expert_scope_document_system_prompt = """You are a collaborative team of the following personas:
{personas}

Your task is to create a scope document that defines the key topics and boundaries within the domain of <domain>{domain}</domain> based on the collective expertise of these personas.

## Output Requirements
1. Structure your document with numbered sections and subsections (e.g., 1, 1.1, 1.2)
2. Use bullet points for lists and enumerations
3. Focus on identifying topics, not relationships or processes

## Content Guidelines
* Define what is included in this domain
* Identify what is explicitly excluded
* Note any gray areas or overlaps with adjacent domains


Keep your document concise and focused on establishing a shared vocabulary and clear boundaries for future discussions.
"""


def generate_scope_document(domain: str, personas: list[Persona]):
    """Generate a scope document for a given domain and group of personas. The generated document delineates the scope and boundaries of the ontology."""
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": expert_scope_document_system_prompt.format(
                    personas="\n".join([f"- {p.description}" for p in personas]),
                    domain=domain,
                ),
            }
        ],
    )

    if response.choices[0].message.content is None:
        raise ValueError("Failed to generate scope document")

    return response.choices[0].message.content


# TODO: maybe revise this by removing the fact that the ontology engineer is an expert in the domain?

merge_documents_prompt = """You are an expert ontology engineer creating an ontology on <domain>{domain}</domain>.

Your task is to merge the provided scope documents into a single, comprehensive, well-structured document.

## Output Requirements:
1. Structure the document with numbered sections and subsections (e.g., 1, 1.1, 1.1.1)
2. Use bullet points for lists and enumerations
3. Organize content logically from general concepts to specific details

## Content Guidelines:
- Retain all essential information from each source document
- Identify and harmonize key ontological elements (classes, properties, relationships, hierarchies)
- Resolve conflicting information by selecting the most authoritative or comprehensive perspective
- Maintain consistent terminology throughout
- Eliminate redundancies while preserving nuanced differences
- Ensure appropriate technical depth for domain specialists
- Use formal, precise language suitable for technical documentation

## Constraints:
- Do not include information about document authors or personas
- Do not add any external information not present in the source documents
- Do not omit significant details from any source document
- Preserve domain-specific terminology and definitions

The final document should represent a cohesive synthesis that accurately reflects all source materials while being clear, comprehensive, and professionally structured."""

CHUNK_SIZE = 6


def merge_scope_documents(
    domain: str, documents: list[str], chunk_size=CHUNK_SIZE
) -> str:
    """Merge scope documents into a single document by splitting them into chunks and merging them recursively."""

    print(f"Merging {len(documents)} scope documents...")

    if len(documents) == 1:
        return documents[0]

    if len(documents) <= chunk_size:
        return _do_merge(domain, documents)

    # split into chunks of size chunk_size
    chunks = [
        documents[i : i + chunk_size] for i in range(0, len(documents), chunk_size)
    ]

    merged_chunks = [_do_merge(domain, chunk) for chunk in chunks]

    return merge_scope_documents(domain, merged_chunks)


def _do_merge(domain: str, documents: list[str]):
    # current format is: <document><authors>{authors}</authors>\n\n{content}</document>
    document_strs = [f"<document>{doc}</document>" for doc in documents]

    documents_message = "<documents>" + "\n\n".join(document_strs) + "</documents>"

    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": merge_documents_prompt.format(
                    domain=domain,
                ),
            },
            {"role": "user", "content": documents_message},
        ],
    )

    if response.choices[0].message.content is None:
        raise ValueError("Failed to merge scope documents")

    return response.choices[0].message.content
