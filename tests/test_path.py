import tempfile

import pytest

from deker_tools.path import is_empty, is_path_valid


class TestIsEmpty:
    dir = "."

    def test_is_not_empty(self):
        assert not is_empty(self.dir)

    def test_is_empty(self):
        with tempfile.TemporaryDirectory(dir=self.dir) as dir:
            assert is_empty(dir)

    def test_is_empty_raises(self):
        with pytest.raises(IsADirectoryError):
            assert is_empty(__file__)


class TestIsPathValid:
    def test_is_path_valid(self):
        assert is_path_valid(".") is None

    def test_is_path_valid_raises_for_file(self):
        with pytest.raises(IsADirectoryError):
            assert is_path_valid(__file__)


if __name__ == "__main__":
    pytest.main()
