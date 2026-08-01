"""Package logging.

Legacy code used `print` unconditionally, even in ``verbosity="silent"`` mode.
This module provides a single logger factory; callers should use it instead of
``print`` so that log level is respected.
"""
from __future__ import annotations

import logging

_LOGGER_NAME = "mdwater"


def get_logger(name: str | None = None) -> logging.Logger:
    """Return the package logger (or a submodule child).

    Configuration is left to the application. The library only attaches a
    NullHandler so that unconfigured use is silent.
    """
    root = logging.getLogger(_LOGGER_NAME)
    if not root.handlers:
        root.addHandler(logging.NullHandler())
    if name is None or name == _LOGGER_NAME:
        return root
    return root.getChild(name)


def enable_stderr_logging(level: int = logging.INFO) -> None:
    """Attach a stderr StreamHandler at ``level`` if not already attached.

    Provided for interactive use (notebooks). Applications should configure
    logging themselves.
    """
    root = get_logger()
    if any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(name)s %(levelname)s] %(message)s"))
    root.addHandler(handler)
    root.setLevel(level)
