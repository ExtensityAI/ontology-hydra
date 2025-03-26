import os
from pathlib import Path

import openai

import ontology.utils  # noqa: F401
from ontology.comittee import Comittee
from ontology.concepts import parse_concepts

domain = "The Fundamentals of Different Programming Languages"


comittee_path = Path("artifacts/langs/sample-comittee.json")
scopes_path = Path("artifacts/langs/scopes")
merged_scope_path = Path("artifacts/langs/merged-scope.txt")
concept_path = Path("artifacts/langs/concept.json")


def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    """comittee = generate_comittee_for_domain(domain)

    comittee_path.write_text(comittee.model_dump_json(indent=2), encoding="utf-8")"""

    comittee = Comittee.model_validate_json(comittee_path.read_text(encoding="utf-8"))

    groups = comittee.divide_into_groups(4)

    """
    # generate scope documents delineating the domain
    for i, group in enumerate(groups):
        scope = generate_scope_document(domain, [m.persona for m in group])
        (scopes_path / f"scope_group_{i}.txt").write_text(scope, encoding="utf-8")"""

    # next steps: merge scope documents into single document

    scope_documents = [
        p.read_text(encoding="utf-8") for p in scopes_path.glob("scope_group_*.txt")
    ]

    """merged_scope = merge_scope_documents(domain, scope_documents)
    merged_scope_path.write_text(merged_scope, encoding="utf-8")"""

    merged_scope = Path.read_text(merged_scope_path, encoding="utf-8")

    concept = parse_concepts(domain, merged_scope)
    concept_path.write_text(concept.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
