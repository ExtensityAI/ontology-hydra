import openai
from pydantic import BaseModel

from ontology.groups import Group
from ontology.personas import Persona
from ontology.utils import MODEL

expert_scope_document_system_prompt = """You are <persona>{persona}</persona>, a recognized expert specializing in {group}. Your task is to author a rigorous document meticulously detailing the scope of inquiry for <domain>{domain}</domain> from your professional perspective.

Your writing must adhere to the following criteria:

- Clearly delineate the scope by specifying the exact areas, phenomena, or entities encompassed by the domain.
- Address all critical dimensions relevant to the domain, ensuring comprehensiveness without redundancy.
- Maintain an objective, scholarly tone characterized by clarity, conciseness, and specificity.
- Make sure that the document encompasses all essential aspects of your subset of the domain.
- Do not include an introduction or conclusion; focus solely on the content of the scope document.
- Avoid using any external information or references; rely solely on your expertise and knowledge of the domain.
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


merge_documents_prompt = """You are an ontology engineer in <domain>{domain}</domain>. Your task is to merge the provided scope documents into a single, coherent, clear, and exhaustive document. Each original document includes a persona description of its author; for each document, consider the persona that created it. The merged document must:

- Retain all essential information from each original document without adding any external information.
- Reflect the perspectives and expertise of all provided personas.
- Eliminate redundancies and maintain clarity and logical flow.
- Ensure coherence and readability throughout.

Do not include any information that is not explicitly present in the original scope documents."""

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
