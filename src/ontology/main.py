import os
from pathlib import Path

import openai
from dotenv import load_dotenv

from ontology.comittee import Comittee
from ontology.scoping import generate_scope_document

domain = "Antarctica"  # taken from SQuAD benchmark, has 525 questions


sample_comittee_path = Path("artifacts/antarctica/sample-comittee.json")
scopes_path = Path("artifacts/antarctica/scopes")


def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # comittee = generate_comittee_for_domain(domain)

    # sample_comittee_path.write_text(
    #     comittee.model_dump_json(indent=2), encoding="utf-8"
    # )

    comittee = Comittee.model_validate_json(
        sample_comittee_path.read_text(encoding="utf-8")
    )

    # generate scope documents delineating the domain
    for i, member in enumerate(comittee.members):
        scope = generate_scope_document(domain, member.persona)
        (scopes_path / f"scope_member_{i}.txt").write_text(scope, encoding="utf-8")


if __name__ == "__main__":
    main()
