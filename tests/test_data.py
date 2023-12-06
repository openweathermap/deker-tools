import pytest

from deker_tools.data import convert_size_to_human


@pytest.mark.parametrize(
    ("bytes", "expected"),
    [
        (1052810, "1.0 Mb"),
        (10489994, "10.0 Mb"),
        (104861834, "100.0 Mb"),
        (1073746058, "1.0 Gb"),
    ],
)
def test_convert_size_to_human(bytes, expected):
    assert convert_size_to_human(bytes) == expected
