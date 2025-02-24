import openai
from pydantic import BaseModel, Field

from ontology.personas.personas import Persona
from ontology.utils import MODEL, rng


class Participant(BaseModel):
    persona: Persona
    speaking_budget: int

    def deduct(self, cost: int):
        # todo raise error or sth if exhausted?
        self.speaking_budget = max(0, self.speaking_budget - cost)


class Thought(BaseModel):
    response: str = Field(..., description="What you want to say")
    # allow to name the next speaker


class Turn(BaseModel):
    participant: Participant
    thought: Thought
    cost: int

    @property
    def text(self):
        return self.thought.response


# ideally, we allocate some budget for the discussion, which is spent on each turn, it ends when the budget is exhausted. also, it should always yield a result. consider giving different speaking time based on priority of the expert's group?

discussion_system_prompt = """You are {participant}, a domain expert invited to a discussion about {topic}.

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

Remember: This is an interactive panel of experts. Engage in genuine back-and-forth, build upon one another's ideas, and stay succinct. Let the discussion unfold naturally until you collectively converge on the most essential questions for the ontology."""


class Discussion:
    def __init__(self, topic: str, participants: list[Participant]):
        self.topic = topic
        self.participants = participants
        self.protocol: list[Turn] = []

    @property
    def remaining_budget(self):
        return sum(p.speaking_budget for p in self.participants)

    @property
    def is_finished(self):
        return self.remaining_budget == 0

    @property
    def _previous_speaker(self):
        return self.protocol[-1].participant if self.protocol else None

    def _get_potential_next_speakers(self):
        previous_speaker = self._previous_speaker

        # return all speakers except the previous one who still have budget
        return [
            p
            for p in self.participants
            if p != previous_speaker and p.speaking_budget > 0
        ]

    def _choose_next_speaker(self):
        # if someone was asked a question, they should be the next speaker TODO

        potential_speakers = self._get_potential_next_speakers()

        # choose speaker probabilistically based on share of total remaining budget
        remaining_budget = sum(p.speaking_budget for p in potential_speakers)

        idx = rng.choice(
            len(potential_speakers),
            p=[p.speaking_budget / remaining_budget for p in potential_speakers],
        )

        return potential_speakers[idx]

    def _prepare_messages(self, participant: Participant):
        return [
            {
                "role": "system",
                "content": discussion_system_prompt.format(
                    participant=participant.persona.name, topic=self.topic
                ),
            },  # TODO: a lot of potential in better prompting, etc.
            *[
                {
                    # current participant has role "assistant" (not sure if better for the model!)
                    "role": "user" if turn.participant != participant else "assistant",
                    "content": turn.text,
                }
                for turn in self.protocol
            ],
        ]

    def _speak(self, participant: Participant):
        response = openai.beta.chat.completions.parse(
            model=MODEL,
            messages=self._prepare_messages(participant),
            response_format=Thought,
        )

        thought = response.choices[0].message.parsed

        if not thought or not response.usage:
            # TODO: find better way to handle errors in general
            raise ValueError("No response from the model")

        # cost is just the number of completion tokens for now
        cost = response.usage.completion_tokens

        return Turn(participant=participant, thought=thought, cost=cost)

    def proceed(self):
        speaker = self._choose_next_speaker()
        print("next speaker is", speaker.persona.name)

        turn = self._speak(speaker)
        speaker.deduct(turn.cost)

        print("response:", turn.thought.response, "\ncost:", turn.cost)

        self.protocol.append(turn)

        return self.is_finished
