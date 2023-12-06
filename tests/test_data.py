import pytest

from deker_tools.data import convert_size_to_human


@pytest.mark.parametrize(
    ("bytes", "expected"),
    [
        (1052810, "1.0 MB"),
        (10489994, "10.0 MB"),
        (104861834, "100.0 MB"),
        (1073746058, "1.0 GB"),
    ],
)
def test_convert_size_to_human(bytes, expected):
    assert convert_size_to_human(bytes) == expected
