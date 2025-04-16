import logging
import sys
from pathlib import Path

from loguru import logger

from ontopipe.eval.utils import InterceptHandler


def init_logging(log_dir_path: Path):
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    logging.getLogger("openai._base_client").setLevel(
        logging.WARN
    )  # ignore unnecessary OpenAI logs for both stderr and file logs

    def _only_ontopipe(record) -> bool:
        # filter log records to only show those from ontopipe (used only in stderr)
        return "ontopipe" in record["name"]

    logger.remove()

    logger.add(sys.stderr, colorize=True, filter=_only_ontopipe)
    logger.add(log_dir_path / "eval_{time}.log", rotation="300 MB")
