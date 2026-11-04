import logging
import os
from pathlib import Path


logger = logging.getLogger("adult_income")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_dir = Path(
        os.environ.get(
            "APP_LOG_DIR",
            str(Path(__file__).resolve().parents[1] / "logs"),
        )
    )
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        logger.exception("File logging could not be initialized at %s", log_dir)
