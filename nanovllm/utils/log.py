import logging

logger = logging.getLogger("nanovllm")
logger.setLevel(logging.INFO)
# logger.setLevel(logging.DEBUG)
_handler = logging.StreamHandler()
_handler.setLevel(logging.DEBUG)
_formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
_handler.setFormatter(_formatter)
if not logger.handlers:
    logger.addHandler(_handler)
# 避免向根 logger 传播导致重复打印，且在未配置根 logger 时也能输出
logger.propagate = False

def log_set_level(level):
    assert level in ['DEBUG', 'INFO', 'ERROR']
    global logger
    if level == 'DEBUG':
        logger.setLevel(logging.DEBUG)
    elif level == 'INFO':
        logger.setLevel(logging.INFO)
    elif level == 'ERROR':
        logger.setLevel(logging.ERROR)

def log_info(message):
    global logger
    logger.info(message)

def log_debug(message):
    global logger
    logger.debug(message)

def log_error(message):
    global logger
    logger.error(message)
    