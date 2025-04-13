"""Defines data structures for the SQuAD v2 dataset (re: https://huggingface.co/datasets/rajpurkar/squad_v2)"""

from pathlib import Path

from pydantic import BaseModel


class SquadAnswer(BaseModel):
    text: str
    """The answer text"""

    answer_start: int
    """The starting character index of the answer in the context"""


class SquadQAPair(BaseModel):
    id: str
    """Unique identifier for the question-answer pair"""

    question: str
    """The question text"""

    answers: list[SquadAnswer]
    """List of answer dictionaries with text field"""

    is_impossible: bool = False
    """Indicates if the question has no answer"""


class SquadParagraph(BaseModel):
    context: str
    """Context text for the paragraph"""

    qas: list[SquadQAPair]
    """List of question-answer pairs in the paragraph"""


class SquadTopic(BaseModel):
    title: str

    paragraphs: list[SquadParagraph]
    """List of paragraphs with context and question-answer pairs"""

    @property
    def contexts(self):
        """Get all contexts from the paragraphs"""

        # unique and sorted
        return sorted({p.context for p in self.paragraphs})

    @property
    def qas(self):
        """Get all question-answer pairs from the paragraphs"""

        # unique and sorted by id
        return sorted([qa for p in self.paragraphs for qa in p.qas], key=lambda x: x.id)


class SquadDataset(BaseModel):
    """Represents the SQuAD dataset"""

    version: str
    data: list[SquadTopic]

    def find_topic(self, title: str):
        """Find a topic by its title"""
        return next((item for item in self.data if item.title == title), None)


def load_squad_dataset(path: Path):
    """Load the SQuAD dataset from a JSON file"""

    if not path.exists():
        raise FileNotFoundError(f"SQuAD dataset not found at {path}. Please read the README for instructions.")

    return SquadDataset.model_validate_json(path.read_text(encoding="utf-8"))
