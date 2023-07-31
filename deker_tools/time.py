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

from datetime import datetime, timezone
from typing import Optional, Union


def get_utc(dt: Optional[Union[str, int, float, datetime]] = None) -> datetime:
    """Convert datetime with any timezone or without it to UTC.

    If dt is ``None`` - UTC current time will be returned

    :param dt: ``datetime.datetime`` object, timestamp, datetime iso-string or ``None``;
    """
    if dt is None:
        return datetime.utcnow().replace(tzinfo=timezone.utc)

    if isinstance(dt, datetime):
        dt = dt.isoformat()  # convert any timezone objects to native format
    elif isinstance(dt, (float, int)):
        dt = datetime.utcfromtimestamp(dt).isoformat()

    dt_object = datetime.fromisoformat(dt)

    if dt_object.tzinfo is None:
        dt_object = dt_object.replace(tzinfo=timezone.utc)
    elif dt_object.tzinfo != timezone.utc:
        tm = dt_object.timestamp()
        dt_object = datetime.utcfromtimestamp(tm).replace(tzinfo=timezone.utc)
    return dt_object
