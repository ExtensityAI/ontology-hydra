from typing import Literal

import openai
from pydantic import BaseModel, Field

from ontology.personas import Persona
from ontology.utils import MODEL, rng

discussion_system_prompt = """You are <persona>{participant}</persona>, a domain expert invited to a discussion about {topic}. The other participants are <participants>{participants}</participants>

# Objective
The host is creating an ontology for {topic}. Your task is to propose and refine the key questions the ontology should address.

# Guidelines

- Stay on topic: Keep the conversation focused on {topic}.
- Be concise: Provide clear, brief statements.
- Avoid repetition: Do not restate your points or the points made by others.
- No answered questions: Do not re-ask questions that have already been resolved.

# Discussion Flow

- Present the questions you believe are crucial for the ontology.
- Respond to other participants' ideas, suggesting improvements or clarifications.
- Refine and prioritize questions collaboratively.

# Work Sheet
Your progress is tracked in this shared worksheet: {worksheet}

Remember: This is an interactive panel of experts. Engage in genuine back-and-forth, build upon one another's ideas, and stay succinct. Let the discussion unfold naturally until you collectively converge on the most essential questions for the ontology."""


class Memory(BaseModel):
    pass  # TODO


class WorkSheet(BaseModel):
    questions: list[str] = Field(
        ..., description="The questions that have been proposed so far."
    )


class Participant(BaseModel):
    persona: Persona
    speaking_budget: int

    def deduct(self, cost: int):
        # todo raise error or sth if exhausted?
        self.speaking_budget = max(0, self.speaking_budget - cost)


class ProposeAction(BaseModel):
    """Propose a new question"""

    type: Literal["propose"]
    question: str = Field(
        ...,
        description="The ontology question you are proposing. It should be clear and concise.",
    )


# TODO revising


Action = ProposeAction


class Turn(BaseModel):
    thought: str = Field(
        ..., description="Think through what you want to say and do next."
    )
    response: str = Field(..., description="Respond to the other participants.")
    speak_to_participant: str | None = Field(
        None,
        description="If you directly address another participant, specify them here.",
    )
    actions: list[Action]


class ProtocolItem(BaseModel):
    participant: Participant
    turn: Turn
    cost: int

    @property
    def text(self):
        return self.turn.response


class Discussion:
    def __init__(self, topic: str, participants: list[Participant]):
        self.topic = topic
        self.participants = participants
        self.protocol = list[ProtocolItem]()
        self.work_sheet = WorkSheet(questions=[])

    @property
    def remaining_budget(self):
        return sum(p.speaking_budget for p in self.participants)

    def _get_participant_by_name(self, name: str):
        for p in self.participants:
            if p.persona.name == name:
                return p

    def _get_next_speaker(self):
        if len(self.protocol) == 0:
            # it's the first turn, choose a random speaker
            idx = rng.choice(len(self.participants))
            return self.participants[idx]

        if self.protocol[-1].turn.speak_to_participant is not None:
            # the last speaker addressed someone, so the next speaker should be that person
            p = self._get_participant_by_name(
                self.protocol[-1].turn.speak_to_participant
            )

            if p is not None:
                return p

        # choose the next speaker based on a weighted random choice
        other_participants = [
            p for p in self.participants if p != self.protocol[-1].participant
        ]
        probs = [p.speaking_budget for p in other_participants]
        probs = [p / sum(probs) for p in probs]

        idx = rng.choice(len(other_participants), p=probs)
        return other_participants[idx]

    def _build_messages(self, speaker: Participant):
        return [
            {
                "role": "system",
                "content": discussion_system_prompt.format(
                    participant=speaker.persona.model_dump_json(),
                    topic=self.topic,
                    participants=", ".join(
                        [
                            p.persona.model_dump_json()
                            for p in self.participants
                            if p != speaker
                        ]
                    ),
                    worksheet=self.work_sheet.model_dump_json(),
                ),
            },
            *[
                {
                    "role": "assistant" if item.participant == speaker else "user",
                    "content": item.text,
                }
                for item in self.protocol
            ],
        ]

    def proceed(self):
        if self.remaining_budget == 0:
            return False

        speaker = self._get_next_speaker()

        response = openai.beta.chat.completions.parse(
            model=MODEL, response_format=Turn, messages=self._build_messages(speaker)
        )

        if response.choices[0].message.parsed is None or response.usage is None:
            raise ValueError("Failed to generate turn")

        turn = response.choices[0].message.parsed
        cost = response.usage.completion_tokens

        for action in turn.actions:
            if action.type == "propose":
                self.work_sheet.questions.append(action.question)

        self.protocol.append(ProtocolItem(participant=speaker, turn=turn, cost=cost))
        print(turn.model_dump_json(indent=2))

        return True
