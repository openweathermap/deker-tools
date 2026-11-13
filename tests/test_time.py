from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from deker_tools.time import get_utc

UTC_DT = datetime(2026, 11, 12, 10, 30, 15, 123456, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "dt",
    [
        UTC_DT,
        UTC_DT.replace(tzinfo=None),
        UTC_DT.astimezone(timezone(timedelta(hours=3))),
        UTC_DT.astimezone(timezone(timedelta(hours=-7, minutes=-30))),
        UTC_DT.astimezone(ZoneInfo("Europe/Nicosia")),
        UTC_DT.astimezone(ZoneInfo("UTC")),
        UTC_DT.isoformat(),
        UTC_DT.replace(tzinfo=None).isoformat(),
        UTC_DT.astimezone(timezone(timedelta(hours=3))).isoformat(),
        UTC_DT.timestamp(),
        int(UTC_DT.timestamp()),
    ],
)
def test_get_utc_returns_the_same_moment_in_utc(dt):
    result = get_utc(dt)
    expected = UTC_DT.replace(microsecond=0) if isinstance(dt, int) else UTC_DT
    assert result == expected
    assert result.tzinfo is timezone.utc


def test_get_utc_none_is_now():
    before = datetime.now(timezone.utc)
    result = get_utc()
    after = datetime.now(timezone.utc)
    assert before <= result <= after
    assert result.tzinfo is timezone.utc


@pytest.mark.parametrize("dt", [b"2026-11-12T10:30:15", object(), [UTC_DT]])
def test_get_utc_raises_on_invalid_type(dt):
    with pytest.raises(TypeError):
        get_utc(dt)


def test_get_utc_raises_on_invalid_string():
    with pytest.raises(ValueError):
        get_utc("not a date")


if __name__ == "__main__":
    pytest.main()
