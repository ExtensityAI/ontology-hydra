from pydantic import BaseModel

from ontology.personas.groups import GroupDef, generate_group_defs
from ontology.personas.personas import Persona, generate_personas
from ontology.utils import rng


class Group(GroupDef):
    personas: list[Persona]


class Population(BaseModel):
    groups: list[Group]

    @property
    def size(self):
        return sum(len(group.personas) for group in self.groups)

    def sample(self, n: int) -> list[Persona]:
        """Returns a random sample of n personas from the population."""

        all_personas = [persona for group in self.groups for persona in group.personas]
        indices = rng.choice(len(all_personas), n, replace=False)

        return [all_personas[i] for i in indices]


def generate_population(topic: str):
    # consider adding a parameter to set the number of personas to generate

    population = Population(groups=[])

    group_defs = generate_group_defs(topic)

    for group_def in group_defs.items:
        print(group_def.model_dump_json(indent=2))
        personas = generate_personas(topic, group_def)

        group = Group(**group_def.model_dump(), personas=personas.items)

        population.groups.append(group)

    return population
