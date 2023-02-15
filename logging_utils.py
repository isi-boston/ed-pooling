import logging


def init_logging(log_file, log_level):
    log_level = log_level.upper()
    logging.getLogger().setLevel(log_level)
    # When debugging, it's useful to see which file and line prints
    # each log message, but too verbose for general use.
    format_str = '{asctime}:{levelname}:{pathname}:{lineno}:{message}'
    # format_str = '{asctime}:{message}'
    log_formatter = logging.Formatter(format_str, style='{')
    root_logger = logging.getLogger()

    # Console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    # log file
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf8')
        file_handler.setFormatter(log_formatter)
        root_logger.addHandler(file_handler)
        logging.info('Logging to %s', log_file)
