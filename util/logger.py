import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

def setup_logging():
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(__name__)


def setup_logging() -> None:
    level = logging.INFO

    logger = logging.getLogger()
    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"seed_{timestamp}.log"
    file_handler = RotatingFileHandler(os.path.join("logs", log_filename))
    file_handler.setLevel(level)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
