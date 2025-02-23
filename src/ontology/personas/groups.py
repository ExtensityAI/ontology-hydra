from enum import StrEnum

import openai
from pydantic import BaseModel, Field

from ontology.utils import MODEL

find_groups_prompt = """You are an ontology engineer tasked with creating an ontology on <topic>{topic}</topic>. As a first step, you are tasked with finding appropriate individuals to interview to get a grasp for the domain. What kind of people would you like to interview? Provide an exhaustive JSON list and nothing else."""


class Priority(BaseModel):
    class Value(StrEnum):
        # todo: figure out if there is a better way to measure importance of groups
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"

    reason: str = Field(..., description="Reason for the priority")
    value: Value


class GroupDef(BaseModel):
    name: str
    description: str
    priority: Priority


class GroupDefs(BaseModel):
    items: list[GroupDef]


def generate_group_defs(topic: str):
    response = openai.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {"role": "system", "content": find_groups_prompt.format(topic=topic)},
        ],
        response_format=GroupDefs,
    )

    obj = response.choices[0].message.parsed

    if obj is None:
        raise ValueError("Failed to generate group definitions")

    return obj
