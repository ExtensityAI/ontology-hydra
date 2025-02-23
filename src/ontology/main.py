import os
from pathlib import Path

import openai
from dotenv import load_dotenv

from ontology.discussions.discussion import ActResult, Discussion, Participant, Thought
from ontology.personas.population import Population
from ontology.utils import MODEL

topic = "European history of the 19th century"

# need a moderator to guide the discussion probably!


def test_act(discussion: Discussion, participant: Participant):
    response = openai.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": f"""You are {participant.persona.model_dump_json()} and you are in a discussion about {discussion.state.topic}.

                # Guidelines
                - Keep the conversation on topic.
                - KEEP YOUR RESPONSES CONCISE.
                - Do not repeat yourself. Do not repeat what the other people say.
                - Do not ask questions that are already answered.
                
                # Goal
                Figure out the most important questions about the topic together with the other participants.""",
            },
            *[
                {
                    "role": "assistant" if turn.speaker != participant else "user",
                    "content": turn.text,
                }
                for turn in discussion.state.protocol
            ],
        ],
        response_format=Thought,
    )

    thought = response.choices[0].message.parsed

    if thought is None or response.usage is None:
        raise Exception("something went wrong, no response")

    return ActResult(thought=thought, cost=response.usage.completion_tokens)


def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

    sample_path = Path("sample-population.json")

    # population = generate_population(topic)
    population = Population.model_validate_json(sample_path.read_text(encoding="utf-8"))

    print(population.size)

    sample = population.sample(5)

    test_discussion = Discussion(
        topic=topic,
        participants=[
            Participant(persona=persona, speaking_budget=1000) for persona in sample
        ],
        act=test_act,
    )

    print(test_discussion.state.model_dump_json(indent=2))

    for i in range(16):
        if not test_discussion.proceed():
            print("finished!")

    with open("sample-discussion", "wb") as f:
        f.write(test_discussion.state.model_dump_json(indent=2).encode("utf-8"))

    # with open("sample-population2.json", "wb") as f:
    #     f.write(population.model_dump_json(indent=2).encode("utf-8"))


if __name__ == "__main__":
    main()
