import os
from pathlib import Path

import openai

DOTENV = Path(".env")

# load .env file
for line in DOTENV.read_text(encoding="utf-8").splitlines():
    if line.startswith("#") or not line.strip():
        continue
    key, value = line.split("=", 1)
    os.environ[key.strip()] = value.strip()

MODEL = os.getenv("NEUROSYMBOLIC_ENGINE_MODEL")
openai.api_key = os.getenv("NEUROSYMBOLIC_ENGINE_API_KEY")
