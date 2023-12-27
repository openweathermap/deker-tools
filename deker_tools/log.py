import logging

from typing import List, NamedTuple, Union


class LoggerNode(NamedTuple):
    """Class describes logger node."""

    name: str
    logger: Union[logging.Logger, logging.PlaceHolder]
    children: List['LoggerNode']


def tree() -> LoggerNode:
    """Return a tree of tuples representing the logger layout.

    Each tuple looks like ``('logger-name', <Logger>, [...])`` where the
    third element is a list of zero or more child tuples that share the
    same layout.

    """
    root = LoggerNode('', logging.root, [])
    nodes = {}
    items = list(logging.root.manager.loggerDict.items())  # for Python 2 and 3
    items.sort()
    for name, logger in items:
        nodes[name] = node = LoggerNode(name, logger, [])
        i = name.rfind('.', 0, len(name) - 1)  # same formula used in `logging`
        if i == -1:
            parent = root
        else:
            parent = nodes[name[:i]]
        parent[2].append(node)
    return root


def set_logger(format_string: str) -> None:
    """Set format for all loggers in the tree.

    :param format_string: Which format we set.
    """
    loggers = tree()
    fmt = logging.Formatter(fmt=format_string)

    def set_format_for_loggers(node: LoggerNode) -> None:
        name, logger, children = node
        if not isinstance(logger, logging.PlaceHolder):
            for handler in logger.handlers:
                handler.setFormatter(fmt)

        for node in children:
            set_format_for_loggers(node)

    logging.basicConfig(format=format_string)
    set_format_for_loggers(loggers)
