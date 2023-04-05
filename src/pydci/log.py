from rich.logging import RichHandler
from rich.text import Text
from rich.console import Console
from loguru import logger


def log_table(rich_table):
    """Generate an ascii formatted presentation of a Rich table
    Eliminates any column styling
    """
    console = Console(width=70)
    with console.capture() as capture:
        console.print(rich_table)
    return Text.from_ansi(capture.get())


def enable_log(file=None, level='INFO', fmt=None, serialize=False):
    """
    Turn on logging for module with appropriate message format
    """
    if file is None:
        fmt = "{message}" if fmt is None else fmt
        logger.configure(handlers=[
            {"sink": RichHandler(markup=True, rich_tracebacks=True),
             "level": level, "format": fmt}])
    else:
        def_fmt = "{message}"
        if not serialize:
            def_fmt = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}"
        fmt = def_fmt if fmt is None else fmt
        logger.configure(handlers=[
            {"sink": file, "serialize": serialize,
             "level": level, "format": fmt}])
    logger.enable('pydci')
    logger.info('Logger initialized')

    return logger


def disable_log():
    """
    Turn of logging
    """
    logger.disable('pydci')
    return logger


logger = disable_log()
