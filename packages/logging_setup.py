from datetime import datetime
import os
import logging


def setup_logging():
    """
    Structured logging for the pipeline.

    Features
    --------
    - Creates logs/ directory automatically
    - Each run gets its own log file
    - Logs go to console AND file
    - Each run has a unique run_id
    """

    # Directory
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_id = f"RUN_{timestamp}"

    # Log file path
    log_file = os.path.join(log_dir, f"pipeline_{timestamp}.log")

    log_format = "%(asctime)s | %(levelname)s | %(message)s"

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))

    logger = logging.getLogger("data_pipeline")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger, run_id
