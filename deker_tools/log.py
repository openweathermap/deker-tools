# deker-tools - shared functions library for deker components
# Copyright (C) 2023-2026  OpenWeather
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import logging

from typing import NamedTuple


class LoggerNode(NamedTuple):
    """A logger in the tree of the loggers of the process, with its children."""

    name: str
    logger: logging.Logger | logging.PlaceHolder
    children: list["LoggerNode"]


def tree() -> LoggerNode:
    """Return a tree of tuples representing the logger layout of the whole process.

    Each tuple looks like ``('logger-name', <Logger>, [...])`` where the
    third element is a list of zero or more child tuples that share the
    same layout.

    :returns: the root logger with every logger of the process under it
    """
    root = LoggerNode("", logging.root, [])
    nodes = {}
    items = list(logging.root.manager.loggerDict.items())  # for Python 2 and 3
    items.sort()
    for name, logger in items:
        nodes[name] = node = LoggerNode(name, logger, [])
        i = name.rfind(".", 0, len(name) - 1)  # same formula used in `logging`
        if i == -1:
            parent = root
        else:
            parent = nodes[name[:i]]
        parent[2].append(node)
    return root


def _set_format(node: LoggerNode, fmt: logging.Formatter) -> None:
    """Set the format of every handler of the logger of the node and of the loggers under it.

    :param node: the logger and its children
    :param fmt: the format
    """
    if not isinstance(node.logger, logging.PlaceHolder):
        for handler in node.logger.handlers:
            handler.setFormatter(fmt)
    for child in node.children:
        _set_format(child, fmt)


def _find_node(node: LoggerNode, name: str) -> LoggerNode | None:
    """Find the node of a logger in the tree.

    :param node: tree to search in
    :param name: name of the logger
    :returns: the node, or ``None`` if there is no such logger
    """
    if node.name == name:
        return node
    for child in node.children:
        found = _find_node(child, name)
        if found is not None:
            return found
    return None


def set_logger(format_string: str, logger_name: str = "") -> None:
    """Set the format of every handler of a logger and of the loggers under it.

    With no logger name the whole process is meant: every handler of every logger gets the format, and the
    root logger is configured with it if it has no handler yet. With a name only that logger and its children
    are touched, the root logger is left alone.

    :param format_string: Which format we set.
    :param logger_name: name of the logger to start from; the root logger if empty
    """
    fmt = logging.Formatter(fmt=format_string)
    if not logger_name:
        logging.basicConfig(format=format_string)
        _set_format(tree(), fmt)
        return
    logging.getLogger(logger_name)  # a logger that was never asked for is not in the tree yet
    node = _find_node(tree(), logger_name)
    if node is None:  # pragma: no cover  # guard: getLogger() above has just registered it
        return
    _set_format(node, fmt)
