from random import Random

from pydantic import BaseModel

from ontopipe.cqs.groups import Group, generate_groups_for_domain
from ontopipe.cqs.personas import Persona, generate_personas_for_group

rng = Random(42)  # for reproducibility


class ComitteeMember(BaseModel):
    persona: Persona
    group: Group


class Comittee(BaseModel):
    members: list[ComitteeMember]

    def sample(self, k: int):
        """Returns a random sample of n members from the comittee."""
        return rng.sample(self.members, k)

    def divide_into_groups(self, group_size: int):
        """Returns a list of groups, each containing a sample of the comittee members."""

        return [
            self.members[i : i + group_size]
            for i in range(0, len(self.members), group_size)
        ]


def generate_comittee_for_domain(domain: str):
    """Generate a comittee of personas belonging to different groups based on the given domain"""
    # consider adding a parameter to set the number of personas to generate

    comittee = Comittee(members=[])

    groups = generate_groups_for_domain(domain)

    for group in groups.items:
        personas = generate_personas_for_group(domain, group)

        for persona in personas.items:
            comittee.members.append(ComitteeMember(persona=persona, group=group))

    # shuffle once to randomize order, useful for sampling into groups
    rng.shuffle(comittee.members)

    return comittee
