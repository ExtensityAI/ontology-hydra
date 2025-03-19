import openai
from pydantic import BaseModel

from ontology.groups import Group
from ontology.personas import Persona
from ontology.utils import MODEL

#! consider removing the group from the prompt, should be implicit through the persona!
expert_scope_document_system_prompt = """You are <persona>{persona}</persona>, a recognized expert specializing in {group}. Your task is to create a scope document that defines the key topics and boundaries within the domain of <domain>{domain}</domain> based on your expertise.

## Output Requirements
1. Structure your document with numbered sections and subsections (e.g., 1, 1.1, 1.2)
2. Use bullet points for lists and enumerations
3. Focus on identifying topics, not relationships or processes

## Content Guidelines
1. Domain Boundaries
   * Define what is included in this domain
   * Identify what is explicitly excluded
   * Note any gray areas or overlaps with adjacent domains

2. Core Terminology
   * List and define key terms and concepts
   * Group related terms into logical categories
   * Highlight any terms with domain-specific meanings

3. Stakeholders and Perspectives
   * Identify key roles and stakeholders in the domain
   * Note different viewpoints that might be relevant

4. Potential Interview Topics
   * Suggest key areas to explore in subject matter expert interviews
   * Highlight topics that may need clarification or deeper exploration

Keep your document concise and focused on establishing a shared vocabulary and clear boundaries for future discussions.
"""


def generate_scope_document(domain: str, persona: Persona, group: Group):
    """Generate a scope document for a given domain and persona. The generated document delineates the scope and boundaries of the ontology."""
    response = openai.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": expert_scope_document_system_prompt.format(
                    persona=persona.description,
                    domain=domain,
                    group=f"{group.name} ({group.description})",
                ),
            }
        ],
    )

    if response.choices[0].message.content is None:
        raise ValueError("Failed to generate scope document")

    return response.choices[0].message.content


class ScopeDocument(BaseModel):
    authors: list[Persona]
    content: str


# TODO: maybe revise this by removing the fact that the ontology engineer is an expert in the domain?

merge_documents_prompt = """You are an expert ontology engineer in <domain>{domain}</domain>. Your task is to merge the provided scope documents into a single, comprehensive, well-structured document. Each original document includes a persona description of its author; consider each persona's expertise and perspective when synthesizing the content.

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

# TODO
# TODO merge documents in chunks - merging all at once is too much for the model!
# TODO

CHUNK_SIZE = 6


def merge_scope_documents(
    domain: str, documents: list[ScopeDocument], chunk_size=CHUNK_SIZE
) -> ScopeDocument:
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


def _do_merge(domain: str, documents: list[ScopeDocument]):
    # current format is: <document><authors>{authors}</authors>\n\n{content}</document>
    document_strs = [
        f"<document><authors>{'\n'.join([f'- {author.description}' for author in doc.authors])}</authors>\n\n{doc.content}</document>"
        for doc in documents
    ]

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

    return ScopeDocument(
        authors=[author for doc in documents for author in doc.authors],
        content=response.choices[0].message.content,
    )
