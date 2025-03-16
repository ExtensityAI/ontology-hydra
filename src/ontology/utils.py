import os
import random
from pathlib import Path

DOTENV_PATH = Path(".env")

rng = random.Random(42)


if not DOTENV_PATH.exists():
    raise FileNotFoundError(f"File not found: {DOTENV_PATH}")

# load environment variables from .env file
for k, v in [line.strip().split("=") for line in DOTENV_PATH.read_text().splitlines()]:
    os.environ[k] = v

MODEL = os.environ["NEUROSYMBOLIC_ENGINE_MODEL"]


def chunked(lst: list, n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
