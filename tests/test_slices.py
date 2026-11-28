from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from deker_tools.slices import (
    SliceConversionError,
    create_shape_from_slice,
    expand_index_exp,
    match_slice_size,
    positive_bounds,
    slice_converter,
    slices_to_str,
    str_to_slices,
)


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
                "[`2023-06-11T00:00:00.000001-03:00`, `2023-06-12T00:00:00.000001-03:00`:`2023-06-12T16:29:18.317633+00:00`, `2023-06-12T00:00:00.000001-03:00`:`2023-06-12T16:29:18.317633+00:00`:`2023-06-12T16:29:18.317633+00:00`]",  # noqa: E501
            ),
        ],
    )
    def test_slice_converter_slice_to_str(self, exp, exp_str):
        assert slice_converter[exp] == exp_str

    @pytest.mark.parametrize(
        ("exp", "exp_str"),
        [
            (1.5, "[1.5]"),
            ((1.5, -2), "[1.5, -2]"),
            (("abc",), "[`abc`]"),
            (datetime(2023, 6, 12, 16, 29, 18, tzinfo=timezone.utc), "[`2023-06-12T16:29:18+00:00`]"),
            (slice(1.5, 2.5, None), "[1.5:2.5]"),
            (slice(None, None, 2), "[::2]"),
            ((), "[()]"),
        ],
    )
    def test_slice_converter_slice_to_str_special_values(self, exp, exp_str):
        assert slice_converter[exp] == exp_str

    @pytest.mark.parametrize("index", [object(), (1, object()), (("abc", 1),)])
    def test_slice_converter_error_on_invalid_type(self, index):
        with pytest.raises(SliceConversionError):
            assert slice_converter[index]

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
                    slice("mama", "-mama", "+mama"),  # labels keep their leading minus
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
                "[`2023-06-11T00:00:00.000001-03:00`, `2023-06-12T00:00:00.000001-03:00`:`2023-06-12T16:29:18.317633+00:00`, `2023-06-12T00:00:00.000001-03:00`:`2023-06-12T16:29:18.317633+00:00`:`2023-06-12T16:29:18.317633+00:00`]",  # noqa: E501
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
        ("exp_str", "exp"),
        [
            ("[`abc`]", "abc"),
            ("[`abc`, 1]", ("abc", 1)),
            ("[`a``b`]", "`a``b`"),
            ("[1.5]", 1.5),
            ("[-1.5]", -1.5),
            ("[-1]", -1),
            ("[None]", None),
            ("[`None`]", "None"),
            (
                "[:`2023-06-12T16:29:18+00:00`]",
                slice(None, "2023-06-12T16:29:18+00:00", None),
            ),
            (
                "[`2023-06-12T16:29:18+00:00`:]",
                slice("2023-06-12T16:29:18+00:00", None, None),
            ),
            (
                "[:`2023-06-12T16:29:18+00:00`:]",
                slice(None, "2023-06-12T16:29:18+00:00", None),
            ),
            (
                "[1:`2023-06-12T16:29:18+00:00`:2]",
                slice(1, "2023-06-12T16:29:18+00:00", 2),
            ),
            (
                "[`2023-06-11T00:00:00+00:00`:`2023-06-12T00:00:00+00:00`:`2023-06-12T06:00:00+00:00`]",
                slice("2023-06-11T00:00:00+00:00", "2023-06-12T00:00:00+00:00", "2023-06-12T06:00:00+00:00"),
            ),
        ],
    )
    def test_slice_converter_str_to_slice_special_values(self, exp_str, exp):
        assert slice_converter[exp_str] == exp

    @pytest.mark.parametrize(
        ("exp_str", "exp"),
        [
            # a numeric step after two datetime bounds
            (
                "[`2023-06-11T00:00:00+00:00`:`2023-06-12T00:00:00+00:00`:3]",
                slice("2023-06-11T00:00:00+00:00", "2023-06-12T00:00:00+00:00", 3),
            ),
            # datetimes without escape symbols
            ("[2023-06-11T00:00:00+00:00]", "2023-06-11T00:00:00+00:00"),
            (
                "[2023-06-11T00:00:00+00:00:2023-06-12T00:00:00+00:00:3]",
                slice("2023-06-11T00:00:00+00:00", "2023-06-12T00:00:00+00:00", 3),
            ),
            ("[:2023-06-12T16:29:18+00:00]", slice(None, "2023-06-12T16:29:18+00:00", None)),
            # separators inside escape symbols are not separators
            ("[`a:b`]", "a:b"),
            ("[`a,b`]", "a,b"),
            ("[`a:b`:`c,d`:2]", slice("a:b", "c,d", 2)),
            ("[`a,b`, `c:d`]", ("a,b", "c:d")),
            ("[`1:2`, 1:2]", ("1:2", slice(1, 2, None))),
        ],
    )
    def test_slice_converter_str_to_slice_escaped_separators(self, exp_str, exp):
        assert slice_converter[exp_str] == exp

    @pytest.mark.parametrize("exp_str", ["[1:2:3:4]", "[`a`:`b`:`c`:`d`]", "[1:2, 1:2:3:4]"])
    def test_slice_converter_error_on_too_many_slice_parts(self, exp_str):
        with pytest.raises(SliceConversionError):
            slice_converter[exp_str]

    @pytest.mark.parametrize(
        ("exp_str", "exp"),
        [
            # literals: ints, floats, one unary minus; exponent/hex/underscore spellings are valid literals too
            ("[5]", 5),
            ("[-5]", -5),
            ("[0.5]", 0.5),
            ("[-0.5]", -0.5),
            ("[1e3]", 1000.0),
            ("[0x10]", 16),
            ("[1_000]", 1000),
            ("[None]", None),
            ("[...]", ...),
            ("[()]", ()),
            # not literals: labels are kept as they are, minus included
            ("[abc]", "abc"),
            ("[-abc]", "-abc"),
            ("[-None]", "-None"),
            ("[-...]", "-..."),
            ("[--1]", "--1"),
            ("[+1]", "+1"),
            ("[1.2.3]", "1.2.3"),
            ("[True]", "True"),
            ("[nan]", "nan"),
            ("[inf]", "inf"),
            ("[(1)]", 1),  # parentheses are not escape symbols
            ("['abc']", "'abc'"),
            ("[1+2j]", "1+2j"),
            # escaped tokens are always strings
            ("[`5`]", "5"),
            ("[`None`]", "None"),
            ("[`...`]", "..."),
            ("[`2023-06-11T00:00:00+00:00`]", "2023-06-11T00:00:00+00:00"),
            ("[`PT3H`]", "PT3H"),
        ],
    )
    def test_slice_converter_str_to_slice_tokens(self, exp_str, exp):
        assert slice_converter[exp_str] == exp

    @pytest.mark.parametrize(
        ("exp_str", "exp"),
        [
            ("[PT3H]", timedelta(hours=3)),
            ("[-PT3H]", timedelta(hours=-3)),
            ("[P1D]", timedelta(days=1)),
            ("[P1DT12H30M]", timedelta(days=1, hours=12, minutes=30)),
            ("[PT0.5S]", timedelta(milliseconds=500)),
            ("[PT90M]", timedelta(minutes=90)),
            ("[PT0S]", timedelta(0)),
            ("[::PT3H]", slice(None, None, timedelta(hours=3))),
            (
                "[`2023-06-12T00:00:00+00:00`:`2023-06-11T00:00:00+00:00`:-PT3H]",
                slice("2023-06-12T00:00:00+00:00", "2023-06-11T00:00:00+00:00", timedelta(hours=-3)),
            ),
            # not durations: no components, unknown units, lower case
            ("[P]", "P"),
            ("[PT]", "PT"),
            ("[P1DT]", "P1DT"),
            ("[-P]", "-P"),
            ("[P1W]", "P1W"),
            ("[P1M]", "P1M"),
            ("[pt3h]", "pt3h"),
        ],
    )
    def test_slice_converter_str_to_slice_durations(self, exp_str, exp):
        assert slice_converter[exp_str] == exp

    @pytest.mark.parametrize(
        ("exp", "exp_str"),
        [
            (timedelta(hours=3), "[PT3H]"),
            (timedelta(hours=-3), "[-PT3H]"),
            (timedelta(days=1), "[P1D]"),
            (timedelta(days=1, hours=12, minutes=30), "[P1DT12H30M]"),
            (timedelta(milliseconds=500), "[PT0.5S]"),
            (timedelta(seconds=1, microseconds=250), "[PT1.00025S]"),
            (timedelta(minutes=90), "[PT1H30M]"),
            (timedelta(0), "[PT0S]"),
            (timedelta(days=-1, hours=-12), "[-P1DT12H]"),
            (slice(None, None, timedelta(hours=3)), "[::PT3H]"),
            (
                (slice(datetime(2023, 6, 12, tzinfo=timezone.utc), None, timedelta(hours=-3)), 0),
                "[`2023-06-12T00:00:00+00:00`::-PT3H, 0]",
            ),
        ],
    )
    def test_slice_converter_slice_to_str_durations(self, exp, exp_str):
        assert slice_converter[exp] == exp_str
        assert slice_converter[exp_str] == (
            exp
            if not isinstance(exp, tuple)
            else tuple(
                slice(e.start.isoformat() if isinstance(e.start, datetime) else e.start, e.stop, e.step)
                if isinstance(e, slice)
                else e
                for e in exp
            )
        )

    @pytest.mark.parametrize(
        ("exp", "exp_str"),
        [
            (np.int64(361), "[361]"),
            (np.int32(-7), "[-7]"),
            (np.uint8(255), "[255]"),
            (np.float64(-87.5), "[-87.5]"),
            (np.float32(0.25), "[0.25]"),
            ((np.int64(1), np.float64(0.5)), "[1, 0.5]"),
            ((slice(np.int64(1), np.int64(10), np.int64(2)),), "[1:10:2]"),
            ((slice(np.float64(-90.0), np.float64(90.0), np.float64(0.5)),), "[-90.0:90.0:0.5]"),
            (np.datetime64("2023-06-11T10:30:15.123456789"), "[`2023-06-11T10:30:15.123456`]"),
            (np.timedelta64(3, "h"), "[PT3H]"),
            (np.timedelta64(1500, "ms"), "[PT1.5S]"),
            (
                (slice(np.datetime64("2023-06-11"), np.datetime64("2023-06-12"), np.timedelta64(6, "h")),),
                "[`2023-06-11T00:00:00`:`2023-06-12T00:00:00`:PT6H]",
            ),
            ([1, slice(None, 5)], "[1, :5]"),
            (np.index_exp[1:5, 0], "[1:5, 0]"),
            (np.s_[::2], "[::2]"),
            (1e20, "[1e+20]"),
            (("NoneSuch", slice("NoneSuch", "b")), "[`NoneSuch`, `NoneSuch`:`b`]"),
        ],
    )
    def test_slice_converter_slice_to_str_numpy_and_containers(self, exp, exp_str):
        assert slice_converter[exp] == exp_str

    def test_slice_converter_large_float_round_trip(self):
        assert slice_converter["[1e+20]"] == 1e20

    @pytest.mark.parametrize(
        "index",
        [
            True,
            np.bool_(True),
            (1, False),
            (slice(True, None),),
            float("nan"),
            float("inf"),
            float("-inf"),
            np.float32("nan"),
            np.float64("inf"),
            np.float16("-inf"),
            np.longdouble("nan"),
            (slice(np.nan, None),),
            np.datetime64("NaT"),
            np.timedelta64("NaT"),
            np.array([1, 2]),
            (np.array([1, 2]),),
            {1, 2},
            {"a": 1},
            iter([1, 2]),
            np.index_exp,
            b"bytes",
            (1 + 2j,),
        ],
    )
    def test_slice_converter_error_on_not_representable(self, index):
        with pytest.raises(SliceConversionError):
            slice_converter[index]

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


class TestConverterFunctions:
    """``slices_to_str`` and ``str_to_slices`` are the typed equivalents of ``slice_converter[...]``."""

    @pytest.mark.parametrize(
        ("exp", "exp_str"),
        [
            (1, "[1]"),
            (-1, "[-1]"),
            (1.5, "[1.5]"),
            (None, "[:]"),
            (..., "[...]"),
            ((), "[()]"),
            ("abc", "[`abc`]"),
            (slice(None, None, None), "[:]"),
            (slice(1, 1, 1), "[1:1:1]"),
            (slice(None, None, -2), "[::-2]"),
            (slice(1.5, 2.5, None), "[1.5:2.5]"),
            (slice("a", "b", 2), "[`a`:`b`:2]"),
            (np.int64(3), "[3]"),
            ([1, slice(None, 2, None)], "[1, :2]"),
            ((-1, 0, 1), "[-1, 0, 1]"),
            ((slice(None, None, None), 1, ..., slice(None, None, 4)), "[:, 1, ..., ::4]"),
            (datetime(2023, 6, 12, 16, 29, 18, tzinfo=timezone.utc), "[`2023-06-12T16:29:18+00:00`]"),
            (
                slice(datetime(2023, 1, 1), datetime(2023, 2, 1), timedelta(hours=3)),
                "[`2023-01-01T00:00:00`:`2023-02-01T00:00:00`:PT3H]",
            ),
        ],
    )
    def test_slices_to_str(self, exp, exp_str):
        assert slices_to_str(exp) == exp_str
        if not isinstance(exp, str):
            assert slices_to_str(exp) == slice_converter[exp]

    def test_slices_to_str_takes_a_bare_string_as_a_label(self):
        # ``slice_converter["abc"]`` parses the string; the typed function has no such ambiguity
        assert slices_to_str("abc") == "[`abc`]"
        with pytest.raises(SliceConversionError):
            slice_converter["abc"]

    @pytest.mark.parametrize(
        ("exp_str", "exp"),
        [
            ("[1]", 1),
            ("[-1]", -1),
            ("[1.5]", 1.5),
            ("[:]", slice(None, None, None)),
            ("[...]", ...),
            ("[()]", ()),
            ("[`abc`]", "abc"),
            ("[abc]", "abc"),
            ("[`10`, 10]", ("10", 10)),
            ("[1:1:1]", slice(1, 1, 1)),
            ("[::-2]", slice(None, None, -2)),
            ("[1.5:2.5]", slice(1.5, 2.5, None)),
            ("[`a`:`b`:2]", slice("a", "b", 2)),
            ("[1, :2]", (1, slice(None, 2, None))),
            ("[-1, 0, 1]", (-1, 0, 1)),
            ("[:, 1, ..., ::4]", (slice(None, None, None), 1, ..., slice(None, None, 4))),
            ("[`2023-06-12T16:29:18+00:00`]", "2023-06-12T16:29:18+00:00"),
            (
                "[`2023-01-01T00:00:00`:`2023-02-01T00:00:00`:-PT3H]",
                slice("2023-01-01T00:00:00", "2023-02-01T00:00:00", timedelta(hours=-3)),
            ),
        ],
    )
    def test_str_to_slices(self, exp_str, exp):
        assert str_to_slices(exp_str) == exp
        assert str_to_slices(exp_str) == slice_converter[exp_str]

    @pytest.mark.parametrize(
        "exp",
        [
            1,
            1.5,
            ...,
            (),
            "abc",
            slice(None, None, None),
            slice(1.5, 2.5, -1),
            (-1, 0, 1),
            (slice(None, None, None), 1, ..., slice(None, None, 4)),
            (slice("a", "b", 2), "c"),
        ],
    )
    def test_functions_round_trip(self, exp):
        assert str_to_slices(slices_to_str(exp)) == exp

    @pytest.mark.parametrize(
        "index",
        [
            object(),
            (1, object()),
            (("abc", 1),),
            {1, 2},
            (..., ...),
            (..., 0.52, slice(1, 8, 5), ...),
        ],
    )
    def test_slices_to_str_error(self, index):
        with pytest.raises(SliceConversionError):
            slices_to_str(index)

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
            "[1:2:3:4]",
            1,
            None,
            (1, 2),
            b"[1]",
        ],
    )
    def test_str_to_slices_error(self, string):
        with pytest.raises(SliceConversionError):
            str_to_slices(string)


@pytest.mark.parametrize(
    ("shape", "slice_", "result"),
    [
        ((361, 720, 4), (slice(1, None, None),), (360, 720, 4)),
        ((361, 720, 4), (slice(1, None, None), slice(1, None, None)), (360, 719, 4)),
        ((361, 720, 4), (slice(None, None, None), slice(None, None, None)), (361, 720, 4)),
        ((361, 720, 4), (slice(None, None, None), slice(None, None, None), 0), (361, 720)),
        ((361, 720, 4), 0, (720, 4)),
        ((1,), 0, ()),
        ((5, 5, 5), 0, (5, 5)),
        ((5, 5, 5), (0, 0), (5,)),
        ((5, 5, 5), ..., (5, 5, 5)),
        ((5, 5, 5), (), (5, 5, 5)),
        ((5, 5, 5), slice(None, None, None), (5, 5, 5)),
        ((5, 5, 5), slice(1, 3), (2, 5, 5)),
        ((5, 5, 5), (..., 1), (5, 5)),
        ((5, 5, 5), (1, ...), (5, 5)),
        ((5, 5, 5), (slice(1, 3), ..., slice(None, 2)), (2, 5, 2)),
        ((5, 5, 5), (slice(-3, None), slice(None, -1, 1)), (3, 4, 5)),
        ((5, 5, 5), np.index_exp[1:3, 0], (2, 5)),
        # steps: positive, negative, larger than the dimension, mixed
        ((10,), slice(None, None, 2), (5,)),
        ((10,), slice(1, None, 3), (3,)),
        ((10,), slice(None, None, -1), (10,)),
        ((10,), slice(8, 1, -3), (3,)),
        ((10,), slice(None, None, 100), (1,)),
        ((10,), slice(None, None, -100), (1,)),
        ((10, 5, 4), (slice(None, None, 2), slice(None, None, -2), 1), (5, 3)),
        ((361, 720), (slice(None, None, 4), slice(None, None, 4)), (91, 180)),
        # an ellipsis stands for every axis the other indices leave out, wherever it is
        ((3, 4, 5, 6), (..., 0), (3, 4, 5)),
        ((3, 4, 5, 6), (0, ...), (4, 5, 6)),
        ((3, 4, 5, 6), (0, ..., 0), (4, 5)),
        ((3, 4, 5, 6), (..., slice(None, None, -2)), (3, 4, 5, 3)),
        ((3, 4, 5, 6), (slice(None, None, 2), ..., 1), (2, 4, 5)),
        ((3, 4, 5, 6), (0, 1, 2, ...), (6,)),
        ((3, 4, 5, 6), (0, 1, 2, 3, ...), ()),
        ((3, 4, 5, 6), (..., 0, 1, 2, 3), ()),
    ],
)
def test_create_shape_from_slice(shape, slice_, result):
    new_shape = create_shape_from_slice(shape, slice_)
    assert new_shape == result


@pytest.mark.parametrize(
    "slice_",
    [
        (0, 0, 0, 0),  # too many indices
        (..., 0, 0, 0, 0),  # too many indices beside an ellipsis
        (..., 0, ...),  # two ellipses
        (slice(None, None, 0),),  # zero step
        (slice(None, None, 0), 0),
        (None, 0),  # None is not a valid index here
        ("a",),
        "a",
        1.5,
        np.index_exp,
    ],
)
def test_create_shape_from_slice_raises(slice_):
    with pytest.raises(IndexError):
        create_shape_from_slice((5, 5, 5), slice_)


@pytest.mark.parametrize(
    ("dim_size", "slice_", "result"),
    [
        (10, slice(None), (0, 10, 1)),
        (10, slice(10), (0, 10, 1)),
        (10, slice(1, 5, 3), (1, 5, 3)),
        (2, slice(1, 5, 3), (1, 2, 3)),  # bounds are clipped to the dimension, as range() would get them
        (10, slice(-3, None), (7, 10, 1)),
        (10, slice(None, -2), (0, 8, 1)),
        (10, slice(-8, -2, 2), (2, 8, 2)),
        # negative step: None means "from the end" / "past the beginning"
        (10, slice(None, None, -1), (9, -1, -1)),
        (10, slice(8, 1, -3), (8, 1, -3)),
        (10, slice(None, 2, -2), (9, 2, -2)),
        (10, slice(-1, None, -1), (9, -1, -1)),
        (10, slice(100, None, -1), (9, -1, -1)),
    ],
)
def test_match_slice_size(dim_size, slice_, result):
    assert match_slice_size(dim_size, slice_) == result


def test_match_slice_size_without_a_slice_is_the_whole_dimension():
    assert match_slice_size(10) == (0, 10, 1)


def test_match_slice_size_does_not_take_none():
    with pytest.raises(AttributeError):
        match_slice_size(10, None)


def test_match_slice_size_zero_step():
    with pytest.raises(ValueError, match="step"):
        match_slice_size(10, slice(None, None, 0))


if __name__ == "__main__":
    pytest.main()


@pytest.mark.parametrize(
    ("shape", "index_exp", "expected"),
    [
        ((5,), 0, (0,)),
        ((5,), slice(1, 3), (slice(1, 3),)),
        ((5, 5), ..., (slice(None), slice(None))),
        ((5, 5), (), (slice(None), slice(None))),
        ((5, 5), 0, (0, slice(None))),
        ((5, 5), (..., 0), (slice(None), 0)),
        ((5, 5, 5), (0, ...), (0, slice(None), slice(None))),
        ((5, 5, 5), (0, ..., -1), (0, slice(None), -1)),
        ((5, 5, 5), (slice(None, None, -2), ...), (slice(None, None, -2), slice(None), slice(None))),
        ((5, 5, 5), (0, 1, 2, ...), (0, 1, 2)),
    ],
)
def test_expand_index_exp(shape, index_exp, expected):
    assert expand_index_exp(shape, index_exp) == expected


@pytest.mark.parametrize(
    "index_exp",
    [(0, 0, 0, 0), (..., 0, 0, 0, 0), (..., 0, ...), (None,), ("a",), 1.5, (True,), (slice(None), False)],
)
def test_expand_index_exp_raises(index_exp):
    with pytest.raises(IndexError):
        expand_index_exp((5, 5, 5), index_exp)


@pytest.mark.parametrize(
    ("shape", "index_exp", "forward", "flipped"),
    [
        ((10,), slice(None), (slice(None),), ()),
        ((10,), slice(None, None, 2), (slice(None, None, 2),), ()),
        ((10,), slice(None, None, -1), (slice(0, 10, 1),), (0,)),
        ((10,), slice(None, None, -2), (slice(1, 10, 2),), (0,)),
        ((10,), slice(8, 2, -2), (slice(4, 9, 2),), (0,)),
        ((10,), slice(-1, None, -3), (slice(0, 10, 3),), (0,)),
        ((10,), slice(2, 8, -1), (slice(0, 0),), ()),  # nothing is selected
        ((10, 10), (3, slice(None, None, -3)), (3, slice(0, 10, 3)), (0,)),  # the integer drops the first axis
        ((10, 10), (slice(None, None, -1), 3), (slice(0, 10, 1), 3), (0,)),
        ((4, 5, 6), (..., slice(None, None, -2)), (slice(None), slice(None), slice(1, 6, 2)), (2,)),
        (
            (4, 5, 6),
            (slice(None, None, -1), 2, slice(4, 0, -1)),
            (slice(0, 4, 1), 2, slice(1, 5, 1)),
            (0, 1),
        ),
    ],
)
def test_positive_bounds(shape, index_exp, forward, flipped):
    assert positive_bounds(shape, index_exp) == (forward, flipped)


@pytest.mark.parametrize(
    ("shape", "index_exp"),
    [
        ((10,), np.index_exp[::-1]),
        ((10,), np.index_exp[8:2:-2]),
        ((10, 10), np.index_exp[::-1, ::2]),
        ((10, 10), np.index_exp[3, ::-3]),
        ((4, 5, 6), np.index_exp[::-1, 2, 4:0:-1]),
        ((4, 5, 6), np.index_exp[..., ::-2]),
        ((3, 4, 5, 6), np.index_exp[-1::-2, ::-1, 3:0:-1, ::-3]),
    ],
)
def test_positive_bounds_select_the_same_cells_reversed(shape, index_exp):
    """The forward twin, flipped along the reported axes, is exactly the numpy selection."""
    data = np.arange(np.prod(shape)).reshape(shape)
    forward, flipped = positive_bounds(shape, index_exp)
    assert np.array_equal(np.flip(data[forward], flipped), data[index_exp])
