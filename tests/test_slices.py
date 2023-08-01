from datetime import datetime, timezone

import numpy as np
import pytest

from deker_tools.slices import SliceConversionError, create_shape_from_slice, match_slice_size, slice_converter


class TestSliceConverter:
    @pytest.mark.parametrize(
        ("exp", "exp_str"),
        [
            (1, "[1]"),
            ((-1, 0, 1), "[-1, 0, 1]"),
            (..., "[...]"),
            (slice(None, None, None), "[:]"),
            (slice(1, None, None), "[1:]"),
            (slice(None, 0, None), "[:0]"),
            ((..., slice(None, None, 1)), "[..., ::1]"),
            ((slice(None, None, None), slice(1, None, None)), "[:, 1:]"),
            ((slice(1, 1, 1), slice(None, -2, None)), "[1:1:1, :-2]"),
            ((slice(None, None, None), 1, ..., slice(None, None, 4)), "[:, 1, ..., ::4]"),
            (None, "[:]"),
            (np.index_exp[1], "[1]"),
            (np.index_exp[:], "[:]"),
            (np.index_exp[...], "[...]"),
            (np.index_exp[-1, 0, 1], "[-1, 0, 1]"),
            (np.index_exp[..., ::1], "[..., ::1]"),
            (np.index_exp[:, 1:], "[:, 1:]"),
            (np.index_exp[1:1:1, :-2], "[1:1:1, :-2]"),
            (np.index_exp[:, 1, ..., ::4], "[:, 1, ..., ::4]"),
            (np.index_exp[()], "[()]"),
            (
                np.index_exp[
                    datetime(2023, 6, 12, 16, 29, 18, 317633, tzinfo=timezone.utc).timestamp() : datetime.fromisoformat(
                        "2023-06-12T16:29:18.317633+00:00"
                    ) : -1.4,
                    -0.26,
                    "mama":"-mama":"+mama",
                    -1:1,
                ],
                "[1686587358.317633:`2023-06-12T16:29:18.317633+00:00`:-1.4, -0.26, `mama`:`-mama`:`+mama`, -1:1]",
            ),
            (np.index_exp["2023-06-11T00:00:00.000001-03:00"], "[`2023-06-11T00:00:00.000001-03:00`]"),
            (
                slice("2023-06-12T00:00:00.000001-03:00", "2023-06-12T16:29:18.317633+00:00", None),
                "[`2023-06-12T00:00:00.000001-03:00`:`2023-06-12T16:29:18.317633+00:00`]",
            ),
            (
                slice(
                    "2023-06-12T00:00:00.000001-03:00",
                    "2023-06-12T16:29:18.317633+00:00",
                    "2023-06-12T16:29:18.317633+00:00",
                ),
                "[`2023-06-12T00:00:00.000001-03:00`:`2023-06-12T16:29:18.317633+00:00`:`2023-06-12T16:29:18.317633+00:00`]",
            ),
            (
                np.index_exp[
                    "2023-06-11T00:00:00.000001-03:00",
                    "2023-06-12T00:00:00.000001-03:00":"2023-06-12T16:29:18.317633+00:00",
                    "2023-06-12T00:00:00.000001-03:00":"2023-06-12T16:29:18.317633+00:00":"2023-06-12T16:29:18.317633+00:00",
                ],
                "[`2023-06-11T00:00:00.000001-03:00`, `2023-06-12T00:00:00.000001-03:00`:`2023-06-12T16:29:18.317633+00:00`, `2023-06-12T00:00:00.000001-03:00`:`2023-06-12T16:29:18.317633+00:00`:`2023-06-12T16:29:18.317633+00:00`]",
            ),
        ],
    )
    def test_slice_converter_slice_to_str(self, exp, exp_str):
        assert slice_converter[exp] == exp_str

    @pytest.mark.parametrize(
        ("exp_str", "exp"),
        [
            ("[1]", 1),
            ("[-1, 0, 1]", (-1, 0, 1)),
            ("[...]", ...),
            ("[:]", slice(None, None, None)),
            ("[1:]", slice(1, None, None)),
            ("[:0]", slice(None, 0, None)),
            ("[..., ::1]", (..., slice(None, None, 1))),
            ("[:, 1:]", (slice(None, None, None), slice(1, None, None))),
            ("[1:1:1, :-2]", (slice(1, 1, 1), slice(None, -2, None))),
            ("[:, 1, ..., ::4]", (slice(None, None, None), 1, ..., slice(None, None, 4))),
            ("[()]", ()),
            (
                "[1686587358.317633:2023-06-12T16:29:18.317633+00:00:-1.4, -0.26, mama:-mama:+mama, -1:1]",
                (
                    slice(1686587358.317633, "2023-06-12T16:29:18.317633+00:00", -1.4),
                    -0.26,
                    slice("mama", "mama", "+mama"),
                    slice(-1, 1, None),
                ),
            ),
            (
                "[`2023-06-11T00:00:00.000001-03:00`]",
                "2023-06-11T00:00:00.000001-03:00",
            ),
            (
                "[`2023-06-12T00:00:00.000001-03:00`:`2023-06-12T16:29:18.317633+00:00`]",
                slice("2023-06-12T00:00:00.000001-03:00", "2023-06-12T16:29:18.317633+00:00", None),
            ),
            (
                "[`2023-06-12T00:00:00.000001-03:00`:`2023-06-12T16:29:18.317633+00:00`:`2023-06-12T16:29:18.317633+00:00`]",
                slice(
                    "2023-06-12T00:00:00.000001-03:00",
                    "2023-06-12T16:29:18.317633+00:00",
                    "2023-06-12T16:29:18.317633+00:00",
                ),
            ),
            (
                "[`2023-06-11T00:00:00.000001-03:00`, `2023-06-12T00:00:00.000001-03:00`:`2023-06-12T16:29:18.317633+00:00`, `2023-06-12T00:00:00.000001-03:00`:`2023-06-12T16:29:18.317633+00:00`:`2023-06-12T16:29:18.317633+00:00`]",
                (
                    "2023-06-11T00:00:00.000001-03:00",
                    slice("2023-06-12T00:00:00.000001-03:00", "2023-06-12T16:29:18.317633+00:00", None),
                    slice(
                        "2023-06-12T00:00:00.000001-03:00",
                        "2023-06-12T16:29:18.317633+00:00",
                        "2023-06-12T16:29:18.317633+00:00",
                    ),
                ),
            ),
        ],
    )
    def test_slice_converter_str_to_slice(self, exp, exp_str):
        assert slice_converter[exp_str] == exp

    @pytest.mark.parametrize(
        "string",
        [
            "1,3,:",
            "hello, world",
            "",
            " ",
            "[]",
            "[ ]",
            "()",
            "( )",
        ],
    )
    def test_slice_converter_error_on_strings(self, string):
        with pytest.raises(SliceConversionError):
            assert slice_converter[string]

    @pytest.mark.parametrize(
        "index",
        [
            np.index_exp[..., ...],
            np.index_exp[..., 0.52, 1:8:5, ...],
        ],
    )
    def test_slice_converter_error_on_index(self, index):
        with pytest.raises(SliceConversionError):
            assert slice_converter[index]


@pytest.mark.parametrize(
    ("slice_", "result"),
    [
        ((slice(1, None, None),), (360, 720, 4)),
        ((slice(1, None, None), slice(1, None, None)), (360, 719, 4)),
        ((slice(None, None, None), slice(None, None, None)), (361, 720, 4)),
        ((slice(None, None, None), slice(None, None, None), 0), (361, 720)),
        (0, (720, 4)),
    ],
)
def test_create_shape_from_slice(slice_, result):
    shape = (361, 720, 4)
    new_shape = create_shape_from_slice(shape, slice_)
    assert new_shape == result


@pytest.mark.parametrize(
    ("dim_size", "slice_", "result"),
    [
        (10, None, (0, 10, 1)),
        (10, slice(10), (0, 10, 1)),
        (10, slice(1, 5, 3), (1, 5, 3)),
        (2, slice(1, 5, 3), (1, 5, 3)),
    ],
)
def test_match_slice_size(dim_size, slice_, result):
    assert match_slice_size(dim_size, slice_) == result


if __name__ == "__main__":
    pytest.main()
