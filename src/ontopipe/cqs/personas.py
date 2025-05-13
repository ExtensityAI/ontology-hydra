import openai
from pydantic import BaseModel

from ontopipe.cqs.groups import Group
from ontopipe.cqs.utils import MODEL

# note: if personas seem low-dimensional, or if they "overfit" to our goal, maybe prompt to modify them without stating our goal, just to make them more realistic

# idea: if we want even more diverse personas, prompt the generated personas from the groups to "find colleagues" who know more? or to "find people who have different perspectives"?

generate_personas_prompt = """You are an ontology engineer tasked with creating a comprehensive ontology on <domain>{domain}</domain>. You have decided to conduct interviews with individuals from <group>{group_name} ({group_description})</group> to deeply understand their varied perspectives and insights into this domain.

Your task:
Identify exactly {n} individuals belonging to this group whom you will interview. Ensure the selected individuals collectively represent a broad range of experiences, backgrounds, and characteristics. Provide a natural language description of each individual, being mindful to cover relevant demographic and experiential details. Include information such as their interests, personal traits, age, geographic location, education, work experience, relevant lived experiences, technological proficiency, and especially any additional distinctive attributes or contexts that provide valuable insights into the domain.

Your descriptions should be comprehensive yet flexible enough to allow the inclusion of extra information where beneficial, ensuring an authentic and nuanced portrayal of each individual without adhering strictly to predefined categories or examples."""

#! idea: add a "quirk it up" prompt to make personas more interesting, re: https://arxiv.org/pdf/1801.07243

# ? maybe prompt the model with the generated personas to "find colleagues" who know more? or to "find people who have different perspectives"?
# ? prompt the model with a "expand on the persona" prompt to make them more specific
# trade off: diversity vs realism, focus more on realism


class Persona(BaseModel):
    name: str
    description: str


class Personas(BaseModel):
    items: list[Persona]


PROPORTIONAL_TO_PRIORITY = -1
ZERO_OR_ONE = 0

# consider changing these values
_priority_to_n = {
    "high": 3,
    "medium": 2,
    "low": 1,
}


def generate_personas_for_group(
    domain: str, group_def: Group, n: int = PROPORTIONAL_TO_PRIORITY
):
    if n == PROPORTIONAL_TO_PRIORITY:
        # set the number of personas to generate based on the priority of the group
        n = _priority_to_n[group_def.priority.value]

    response = openai.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": generate_personas_prompt.format(
                    domain=domain,
                    group_name=group_def.name,
                    group_description=group_def.description,
                    n=n,
                ),
            },
        ],
        response_format=Personas,
    )

    obj = response.choices[0].message.parsed

    if obj is None:
        raise ValueError("Failed to generate personas")

    return obj
