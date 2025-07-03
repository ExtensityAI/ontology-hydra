"""
File-only logging with no console output.
"""
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from loguru import logger

# Remove ALL handlers (including default console)
logger.remove()

# Only run once
if not hasattr(logger, "_file_configured"):
    ROOT = Path(__file__).resolve().parent.parent
    LOG_DIR = ROOT / "assets" / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE = LOG_DIR / f"search_{datetime.now():%Y-%m-%d}.log"

    # File sink
    logger.add(
        LOG_FILE,
        level="DEBUG",
        colorize=False,
        enqueue=True,
        rotation="30 MB",
        retention="14 days",
        compression="zip",
    )

    # Console sink for logger messages
    logger.add(
        sys.stderr,
        level="DEBUG",
        colorize=True,
        enqueue=False,
    )

    # Simple print capture - direct to file, no logger calls
    orig_stdout = sys.stdout
    _ansi_re = re.compile(r"\x1b\[[0-9;]*[A-Za-z]|\[[0-9;]*[mGKF]")

    class _StdCapture:
        def write(self, msg):
            orig_stdout.write(msg) # clean console
            # Direct file write - bypass logger completely
            if msg.strip():
                clean = _ansi_re.sub("", msg.rstrip())
                with open(LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(clean + "\n")

        def flush(self): orig_stdout.flush()
        def isatty(self): return True
        def __getattr__(self, name): return getattr(orig_stdout, name)

    sys.stdout = _StdCapture()
    logger._file_configured = True

__all__ = ["logger"]
