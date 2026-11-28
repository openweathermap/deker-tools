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

from datetime import datetime, timezone

from deker_tools.typings import Sentinel, SentinelType, is_sentinel


def get_utc(dt: str | int | float | datetime | SentinelType = Sentinel) -> datetime:
    """Convert datetime with any timezone or without it to UTC.

    A naive datetime or ISO string is taken as UTC already; a timestamp is seconds since the epoch.
    Without a moment, the current UTC time is returned.

    :param dt: ``datetime.datetime`` object, timestamp or datetime iso-string; not given - the current moment
    :returns: timezone-aware ``datetime.datetime`` in UTC
    :raises ValueError: if the string is not an ISO datetime
    """
    if is_sentinel(dt):
        return datetime.now(timezone.utc)

    if isinstance(dt, datetime):
        dt_object = dt
    elif isinstance(dt, (float, int)):
        return datetime.fromtimestamp(dt, timezone.utc)
    else:
        dt_object = datetime.fromisoformat(dt)

    if dt_object.tzinfo is None:
        return dt_object.replace(tzinfo=timezone.utc)
    return dt_object.astimezone(timezone.utc)
