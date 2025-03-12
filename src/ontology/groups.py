from enum import StrEnum

import openai
from pydantic import BaseModel, Field

from ontology.utils import MODEL

# todo: hierarchies

generate_groups_prompt = """You are an ontology engineer tasked with creating an ontology on <domain>{domain}</domain>. As a first step, you are tasked with finding appropriate individuals to interview. What kind of people would you like to interview? Provide an exhaustive JSON list and nothing else."""


class Priority(BaseModel):
    class Value(StrEnum):
        # todo: figure out if there is a better way to measure importance of groups
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"

    reason: str = Field(..., description="Reason for the priority")
    value: Value


class Group(BaseModel):
    name: str
    description: str
    priority: Priority


class Groups(BaseModel):
    items: list[Group]


def generate_groups_for_domain(domain: str):
    response = openai.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": generate_groups_prompt.format(domain=domain)},
        ],
        response_format=Groups,
    )

    obj = response.choices[0].message.parsed

    if obj is None:
        raise ValueError("Failed to generate group definitions")

    return obj
