# deker-tools - shared functions library for deker components
# Copyright (C) 2023  OpenWeather
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

import errno
import os
import sys

from pathlib import Path
from typing import Union


__all__ = ["is_empty", "is_path_valid"]

Pathlike = Union[Path, str]


def is_empty(path: Pathlike) -> bool:
    """Check if directory is empty.

    :param path: Path to check
    """
    if isinstance(path, Path):
        path = str(path)

    if not os.path.isdir(path):
        raise IsADirectoryError(f"Path {path} is not a directory")

    with os.scandir(path) as iterator:
        if any(iterator):
            return False
    return True


def is_path_valid(path: Pathlike) -> None:
    """Check if directory path is valid.

    :param path: path to a directory
    """
    if isinstance(path, Path):
        path = str(path)

    if os.path.exists(path):
        if not os.path.isdir(path):
            raise IsADirectoryError(f"Path {path} is not a directory")

    _, path = os.path.splitdrive(path)
    root_dirname = os.environ.get("HOMEDRIVE", "C:") if sys.platform == "win32" else os.path.sep
    if not os.path.isdir(root_dirname):
        raise IsADirectoryError(f"Path {root_dirname} is not a directory")
    root_dirname = root_dirname.rstrip(os.path.sep) + os.path.sep
    for part in path.split(os.path.sep):
        try:
            os.lstat(root_dirname + part)
        except OSError as exc:
            if hasattr(exc, "winerror"):
                if exc.winerror == 123:
                    raise
            elif exc.errno in {errno.ENAMETOOLONG, errno.ERANGE}:
                raise
