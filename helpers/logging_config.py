import os
import logging
import arrow
from typing import Optional
from helpers.helper import get_root_directory

def setup_logging(log_dir: str = f'{get_root_directory()}/logs',
                  log_file: str = f"{arrow.now().format('YYYY-MM-DD-HH-mm')}.log",
                  new_dir_name: Optional[str] = None):
    """
    Set up logging configuration.
    """

    if new_dir_name:
        log_directory = os.path.join(log_dir, new_dir_name)
    else:
        log_directory = log_dir

    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    logger = logging.getLogger()

    if logger.hasHandlers():
        logger.handlers.clear()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_directory, log_file)),
            logging.StreamHandler()
        ]
    )