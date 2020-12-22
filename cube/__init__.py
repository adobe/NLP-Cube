from cube.api2 import load

import logging
logger = logging.getLogger("cube")

log_handler = logging.StreamHandler()
log_formatter = logging.Formatter(fmt="[%(levelname)8s | %(asctime)s | %(filename)-20s:%(lineno)3s | %(funcName)-26s] %(message)s",
                              datefmt='%Y-%m-%d %H:%M:%S')
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)
logger.setLevel(logging.DEBUG)
