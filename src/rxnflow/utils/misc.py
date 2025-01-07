import logging
import sys


_WORKER = {}


def set_worker_env(key, val):
    _WORKER[key] = val


def get_worker_env(key):
    return _WORKER[key]


def create_logger(name="logger", loglevel=logging.INFO, logfile=None, streamHandle=True):
    logger = logging.getLogger(name)
    logger.setLevel(loglevel)
    formatter = logging.Formatter(
        fmt=f"%(asctime)s - %(levelname)s - {name} - %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
    )

    handlers = []
    if logfile is not None:
        handlers.append(logging.FileHandler(logfile, mode="a"))
    if streamHandle:
        handlers.append(logging.StreamHandler(stream=sys.stdout))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
