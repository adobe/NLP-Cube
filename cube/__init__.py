from cube.api import _load
from cube.version import __version__

import logging
logger = logging.getLogger('cube')

if logger.level == 0:
    logger.setLevel(logging.INFO)

log_handler = logging.StreamHandler()
log_formatter = logging.Formatter(fmt="%(asctime)s %(levelname)s: %(message)s", datefmt='%Y-%m-%d %H:%M:%S')
log_handler.setFormatter(log_formatter)