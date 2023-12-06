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

import math


def convert_size_to_human(size_bytes: int) -> str:
    """Convert bytes to human size.

    :param size_bytes: size in bytes
    """
    if size_bytes == 0:
        return "0 b"
    size_name = ("b", "Kb", "Mb", "Gb", "Tb", "Pb", "Eb", "Zb", "Yb")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"
