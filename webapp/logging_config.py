import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class StructuredFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        })


def setup_logging():
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(levelname)-8s %(name)s  %(message)s"))

    file_handler = RotatingFileHandler(
        LOG_DIR / "webapp.log", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s  %(message)s")
    )

    json_handler = RotatingFileHandler(
        LOG_DIR / "webapp.jsonl", maxBytes=10 * 1024 * 1024, backupCount=5
    )
    json_handler.setFormatter(StructuredFormatter())

    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[console, file_handler, json_handler],
    )
