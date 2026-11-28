# deker-tools - shared functions library for deker components
# Copyright (C) 2023-2026  Openweather Ltd.
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

"""Types shared by the DEKER packages."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeIs

__all__ = ("Sentinel", "SentinelType", "is_sentinel")


class SentinelType:
    """Type of ``Sentinel``, the value of an argument that is not given.

    The type has the single instance: calling it, copying the instance or loading it from a pickle gives the same
    ``Sentinel`` back, so ``is_sentinel`` can compare by identity in every process and every package.
    """

    __slots__ = ()

    def __new__(cls) -> "SentinelType":
        """Give the single instance instead of a new one.

        :returns: ``Sentinel``
        """
        return Sentinel

    def __reduce__(self) -> str:
        """Pickle and copy the instance by its name in this module.

        :returns: the name of the module attribute that holds the instance
        """
        return "Sentinel"

    def __repr__(self) -> str:
        """Show the name of the instance.

        :returns: ``"Sentinel"``
        """
        return "Sentinel"


Sentinel: SentinelType = object.__new__(SentinelType)


def is_sentinel(value: object) -> "TypeIs[SentinelType]":
    """Tell whether the value is ``Sentinel``, that is, the argument is not given.

    :param value: any value of an argument
    :returns: ``True`` for ``Sentinel``, ``False`` for any other value, ``None`` included
    """
    return value is Sentinel
