import copy
import pickle

import pytest

from deker_tools.typings import Sentinel, SentinelType, is_sentinel


def pickle_round_trip(value):
    return pickle.loads(pickle.dumps(value))


def test_sentinel_is_the_instance_of_its_type():
    assert isinstance(Sentinel, SentinelType)


def test_sentinel_type_gives_the_same_instance():
    assert SentinelType() is Sentinel


def test_sentinel_repr():
    assert repr(Sentinel) == "Sentinel"


@pytest.mark.parametrize(
    "duplicate",
    [
        pytest.param(pickle_round_trip, id="pickle"),
        pytest.param(copy.copy, id="copy"),
        pytest.param(copy.deepcopy, id="deepcopy"),
    ],
)
def test_sentinel_stays_the_same_object(duplicate):
    assert duplicate(Sentinel) is Sentinel


def test_sentinel_in_a_container_stays_the_same_object():
    restored = pickle_round_trip({"timeout": Sentinel})
    assert restored["timeout"] is Sentinel


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(Sentinel, True, id="sentinel"),
        pytest.param(None, False, id="none"),
        pytest.param(0, False, id="zero"),
        pytest.param("", False, id="empty-string"),
        pytest.param(False, False, id="false"),
        pytest.param([], False, id="empty-list"),
        pytest.param(object(), False, id="object"),
        pytest.param(SentinelType, False, id="the-type"),
    ],
)
def test_is_sentinel(value, expected):
    assert is_sentinel(value) is expected
