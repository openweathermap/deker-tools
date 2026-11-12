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
    """Class describes logger node."""

    name: str
    logger: logging.Logger | logging.PlaceHolder
    children: list["LoggerNode"]


def tree() -> LoggerNode:
    """Return a tree of tuples representing the logger layout.

    Each tuple looks like ``('logger-name', <Logger>, [...])`` where the
    third element is a list of zero or more child tuples that share the
    same layout.

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
