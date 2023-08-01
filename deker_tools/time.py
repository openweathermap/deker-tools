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