"""
pyDCI Logging Module

pyDCI uses `Link loguru <https://loguru.readthedocs.io/en/stable/index.html>`_
in combination with `Link rich <https://github.com/Textualize/rich>`_ to do
logging in a nice and conise way. By default logging is disabled, but can
be enable by importing the :func: `enable_log` to turn logging on.
"""
from loguru import logger
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from rich.traceback import install
install(show_locals=True)


def log_table(rich_table):
    """
    Log rich table

    Generate an ascii formatted presentation of a Rich table eliminates
    any column styling. This function is to be used for logging a table that
    to loguru that is created by rich table.

    Parameters
    ----------
    rich_table: rich.Table
        rich table to output to file.

    Returns
    -------
    Text to output to loguru logging handler.
    """
    console = Console(width=70)
    with console.capture() as capture:
        console.print(rich_table)
    return Text.from_ansi(capture.get())


def enable_log(file=None, level="INFO", fmt=None, serialize=False):
    """
    Turn on logging for module with appropriate message format

    Parameters
    ----------
    file: str, optional
        If specified, name of file to log output to. If none specified, then
        logging is done to stdout.
    level: str, default="INFO"
        `Link log-level <https://docs.python.org/3/library/logging.html>`_ to
        use. Max is "DEBUG". Minimum is "CRITICAL"
    fmt: str, default=NONE
        Format to use for logging, in logguru's simplified syntax. See loguru
        documentation for more info.
    serialize: bool, default=False
        If set to True, output log in json form.
    """
    if file is None:
        fmt = "{message}" if fmt is None else fmt
        logger.configure(
            handlers=[
                {
                    "sink": RichHandler(markup=True, rich_tracebacks=True),
                    "level": level,
                    "format": fmt,
                }
            ]
        )
    else:
        def_fmt = "{message}"
        if not serialize:
            def_fmt = "{time:YYYY-MM-DD at HH:mm:ss} | {level} | {message}"
        fmt = def_fmt if fmt is None else fmt
        logger.configure(
            handlers=[
                {"sink": file, "serialize": serialize, "level": level, "format": fmt}
            ]
        )
    logger.enable("pydci")
    logger.info("Logger initialized")

    return logger


def disable_log():
    """
    Turn of logging
    """
    logger.disable("pydci")
    return logger


_ = disable_log()
