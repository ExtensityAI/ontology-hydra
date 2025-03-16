import os
from pathlib import Path

import openai

import ontology.utils  # noqa: F401
from ontology.comittee import Comittee
from ontology.scoping import (
    ScopeDocument,
    merge_scope_documents,
)

domain = "Antarctica"  # taken from SQuAD benchmark, has 525 questions


comittee_path = Path("artifacts/antarctica/sample-comittee.json")
scopes_path = Path("artifacts/antarctica/scopes")
merged_scope_path = Path("artifacts/antarctica/merged-scope.txt")


def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    """comittee = generate_comittee_for_domain(domain)

    sample_comittee_path.write_text(
        comittee.model_dump_json(indent=2), encoding="utf-8"
    )"""

    comittee = Comittee.model_validate_json(comittee_path.read_text(encoding="utf-8"))

    # generate scope documents delineating the domain
    """for i, member in enumerate(comittee.members):
        scope = generate_scope_document(domain, member.persona, member.group)
        (scopes_path / f"scope_member_{i}.txt").write_text(scope, encoding="utf-8")"""

    # next steps: merge scope documents into single document

    scope_documents = [
        ScopeDocument(
            authors=[member.persona],
            content=(scopes_path / f"scope_member_{i}.txt").read_text("utf-8"),
        )
        for i, member in enumerate(comittee.members)
    ]

    merged_scope = merge_scope_documents(domain, scope_documents)
    merged_scope_path.write_text(merged_scope.content, encoding="utf-8")


if __name__ == "__main__":
    main()
