import os
from pathlib import Path

import openai

import ontology.utils  # noqa: F401
from ontology.comittee import Comittee
from ontology.scoping import ScopeDocument, merge_scope_documents

domain = "Antarctica"  # taken from SQuAD benchmark, has 525 questions


sample_comittee_path = Path("artifacts/antarctica/sample-comittee.json")
scopes_path = Path("artifacts/antarctica/scopes")


def main():
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # comittee = generate_comittee_for_domain(domain)

    # sample_comittee_path.write_text(
    #     comittee.model_dump_json(indent=2), encoding="utf-8"
    # )

    comittee = Comittee.model_validate_json(
        sample_comittee_path.read_text(encoding="utf-8")
    )

    # generate scope documents delineating the domain
    # for i, member in enumerate(comittee.members):
    #     scope = generate_scope_document(domain, member.persona)
    #     (scopes_path / f"scope_member_{i}.txt").write_text(scope, encoding="utf-8")

    # next steps: merge scope documents into single document

    scope_documents = [
        ScopeDocument(
            author=member.persona,
            content=(scopes_path / f"scope_member_{i}.txt").read_text("utf-8"),
        )
        for i, member in enumerate(comittee.members)
    ]

    merged_scope = merge_scope_documents(domain, scope_documents)
    print(merged_scope)


if __name__ == "__main__":
    main()
