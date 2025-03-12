import os
import random
from pathlib import Path

DOTENV_PATH = Path(".env")
MODEL = os.environ["NEUROSYMBOLIC_ENGINE_MODEL"]


def load_dotenv():
    """Load env variables from .env file"""

    if not DOTENV_PATH.exists():
        raise FileNotFoundError(f"File not found: {DOTENV_PATH}")

    for k, v in [
        line.strip().split("=") for line in DOTENV_PATH.read_text().splitlines()
    ]:
        os.environ[k] = v


load_dotenv()


rng = random.Random(42)
