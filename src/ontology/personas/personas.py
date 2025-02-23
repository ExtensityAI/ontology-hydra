from typing import Literal

import openai
from pydantic import BaseModel, Field

from ontology.personas.groups import GroupDef, Priority
from ontology.utils import MODEL

# note: if personas seem low-dimensional, or if they "overfit" to our goal, maybe prompt to modify them without stating our goal, just to make them more realistic

# idea: if we want even more diverse personas, prompt the generated personas from the groups to "find colleagues" who know more? or to "find people who have different perspectives"?

generate_personas_prompt = """You are an ontology engineer tasked with creating an ontology on <topic>{topic}</topic>. You have decided to interview {group_name} ({group_description}). Find exactly {n} individuals belonging to the group whom you would like to interview to cover their perspective on the domain. Ensure you cover a wide range of experiences, backgrounds, geographical locations, lived experiences, ... 

The goal is to avoid over-representing a single demographic and instead capture a broad spectrum of voices that could contribute valuable insights on this topic."""


class Education(BaseModel):
    institution: str
    degree: str | None
    field_of_study: str | None
    description: str | None


class WorkExperience(BaseModel):
    company: str
    position: str
    description: str | None


class Skill(BaseModel):
    name: str
    proficiency: Literal["beginner", "intermediate", "advanced"] | None


class Persona(BaseModel):
    name: str
    bio: str = Field(
        description='Short bio written by the person themselves. ("I am ...")'
    )
    education: list[Education] | None
    work_experience: list[WorkExperience] | None
    skills: list[Skill] | None


class Personas(BaseModel):
    items: list[Persona]


PROPORTIONAL_TO_PRIORITY = -1

# consider changing these values
_priority_to_n = {
    Priority.Value.HIGH: 5,
    Priority.Value.MEDIUM: 3,
    Priority.Value.LOW: 1,
}


def generate_personas(
    topic: str, group_def: GroupDef, n: int = PROPORTIONAL_TO_PRIORITY
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
                    topic=topic,
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
