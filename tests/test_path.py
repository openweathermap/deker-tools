import errno
import os
import tempfile

from pathlib import Path

import pytest

from deker_tools.path import is_empty, is_path_valid


class TestIsEmpty:
    dir = "."

    def test_is_not_empty(self):
        assert not is_empty(self.dir)

    def test_is_not_empty_path_instance(self):
        assert not is_empty(Path(self.dir))

    def test_is_empty(self):
        with tempfile.TemporaryDirectory(dir=self.dir) as dir:
            assert is_empty(dir)

    def test_is_empty_path_instance(self):
        with tempfile.TemporaryDirectory(dir=self.dir) as dir:
            assert is_empty(Path(dir))

    def test_is_empty_raises(self):
        with pytest.raises(NotADirectoryError):
            assert is_empty(__file__)


class TestIsPathValid:
    def test_is_path_valid(self):
        assert is_path_valid(".") is None

    def test_is_path_valid_path_instance(self):
        assert is_path_valid(Path(".")) is None

    def test_is_path_valid_not_existing_path(self):
        assert is_path_valid("/no/such/directory/anywhere") is None

    def test_is_path_valid_raises_for_file(self):
        with pytest.raises(NotADirectoryError):
            assert is_path_valid(__file__)

    def test_is_path_valid_raises_for_too_long_name(self):
        with pytest.raises(OSError) as exc_info:
            is_path_valid("/" + "x" * 300)
        assert exc_info.value.errno in {errno.ENAMETOOLONG, errno.ERANGE}

    def test_is_path_valid_raises_for_invalid_root(self, monkeypatch):
        monkeypatch.setattr(os.path, "isdir", lambda path: False)
        with pytest.raises(NotADirectoryError):
            is_path_valid("/no/such/directory")

    @staticmethod
    def _lstat_raising(winerror: int):
        def lstat(path):
            exc = OSError("emulated Windows error")
            exc.winerror = winerror
            raise exc

        return lstat

    def test_is_path_valid_raises_for_windows_invalid_name(self, monkeypatch):
        monkeypatch.setattr(os, "lstat", self._lstat_raising(123))
        with pytest.raises(OSError) as exc_info:
            is_path_valid("/some/path")
        assert exc_info.value.winerror == 123

    def test_is_path_valid_ignores_other_windows_errors(self, monkeypatch):
        monkeypatch.setattr(os, "lstat", self._lstat_raising(2))
        assert is_path_valid("/some/path") is None

    def test_is_path_valid_ignores_other_os_errors(self, monkeypatch):
        def lstat(path):
            raise OSError(errno.EACCES, "permission denied")

        monkeypatch.setattr(os, "lstat", lstat)
        assert is_path_valid("/some/path") is None


if __name__ == "__main__":
    pytest.main()
