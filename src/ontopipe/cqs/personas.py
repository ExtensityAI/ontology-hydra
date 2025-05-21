from logging import getLogger

from pydantic import Field
from symai import Expression
from symai.components import MetadataTracker
from symai.strategy import LLMDataModel, contract

from ontopipe.cqs.groups import Group
from ontopipe.prompts import prompt_registry

logger = getLogger("ontopipe.cqs")

# note: if personas seem low-dimensional, or if they "overfit" to our goal, maybe prompt to modify them without stating our goal, just to make them more realistic

# idea: if we want even more diverse personas, prompt the generated personas from the groups to "find colleagues" who know more? or to "find people who have different perspectives"?


class Persona(LLMDataModel):
    name: str
    description: str


class Personas(LLMDataModel):
    items: list[Persona]


class PersonasGeneratorInput(LLMDataModel):
    domain: str = Field(..., description="The domain of the ontology")
    group: Group = Field(..., description="The group to generate personas for")
    existing_personas: list[Persona] = Field(
        description="Personas that have already been found"
    )
    n: int = Field(..., description="Number of personas to generate")


PROPORTIONAL_TO_PRIORITY = -1
ZERO_OR_ONE = 0

# consider changing these values
_priority_to_n = {
    "high": 3,
    "medium": 2,
    "low": 1,
}


@contract(
    pre_remedy=False,
    post_remedy=True,
    accumulate_errors=False,
    verbose=True,
    remedy_retry_params=dict(
        tries=25, delay=0.5, max_delay=15, jitter=0.1, backoff=2, graceful=False
    ),
)
class PersonasGenerator(Expression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input: PersonasGeneratorInput, **kwargs) -> Personas:
        if self.contract_result is None:
            raise ValueError("Contract failed!")

        return self.contract_result

    def post(self, output: Personas) -> bool:
        return True

    @property
    def prompt(self) -> str:
        return prompt_registry.instruction("generate_personas")


def generate_personas_for_group(
    domain: str, group_def: Group, n: int = PROPORTIONAL_TO_PRIORITY
):
    if n == PROPORTIONAL_TO_PRIORITY:
        # set the number of personas to generate based on the priority of the group
        n = _priority_to_n[group_def.priority.value]

    personas = list[Persona]()

    # TODO maybe provide all personas here, so it can generate more diverse ones - right now we only provide the ones from the current group!

    generator = PersonasGenerator()
    with MetadataTracker() as tracker:
        while len(personas) < n:
            n_remaining = n - len(personas)
            new_personas = generator(
                input=PersonasGeneratorInput(
                    domain=domain,
                    group=group_def,
                    n=n_remaining,
                    existing_personas=personas,
                )
            )

            # add at most n_remaining personas
            personas.extend(new_personas.items[:n_remaining])

            generator.contract_perf_stats()
            logger.debug("API Usage: %s", tracker.usage)

    return personas
