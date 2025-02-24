import os
from pathlib import Path

import openai
from dotenv import load_dotenv

from ontology.discussions.discussion import Discussion, Participant
from ontology.personas.population import Population

topic = "European history of the 19th century"

# need a moderator to guide the discussion probably!


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
    )

    for i in range(16):
        if test_discussion.proceed():
            print("finished!")

    # with open("sample-population2.json", "wb") as f:
    #     f.write(population.model_dump_json(indent=2).encode("utf-8"))


if __name__ == "__main__":
    main()
