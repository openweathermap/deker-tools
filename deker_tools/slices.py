# deker-tools - shared functions library for deker components
# Copyright (C) 2023-2026  Openweather Ltd.
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

import ast
import datetime
import math
import numbers
import re

from collections.abc import Iterable
from types import EllipsisType
from typing import TypeAlias, Union

import numpy as np

__all__ = [
    "match_slice_size",
    "create_shape_from_slice",
    "slice_converter",
    "slices_to_str",
    "str_to_slices",
    "SliceConversionError",
]

# what ``np.index_exp[...]`` produces: a single index or a tuple of them
Slice: TypeAlias = Union[slice, EllipsisType, int, tuple[Union[slice, int, EllipsisType, None], ...]]

# one unit of an index expression as the string form reads it
SliceUnit: TypeAlias = Union[int, float, str, datetime.datetime, datetime.timedelta, EllipsisType, tuple[()], None]
# one unit as the string form accepts it for writing: numpy scalars are converted to the plain types above
SliceUnitInput: TypeAlias = Union[SliceUnit, np.integer, np.floating, np.datetime64, np.timedelta64]

FancySlice: TypeAlias = Union[
    slice,
    SliceUnit,
    tuple[Union[slice, SliceUnit], ...],
]
FancySliceInput: TypeAlias = Union[
    slice,
    SliceUnitInput,
    tuple[Union[slice, SliceUnitInput], ...],
    list[Union[slice, SliceUnitInput]],
]


WHOLE_DIMENSION = slice(None)  # Slice of a whole dimension, the default of ``match_slice_size``.


def match_slice_size(dim_len: int, slice_: slice = WHOLE_DIMENSION) -> tuple[int, int, int]:
    """Convert slice into a sequence and get its length.

    :param dim_len: length of the corresponding Dimension
    :param slice_: slice to be converted; not given - the whole dimension
    :returns: ``(start, stop, step)`` resolved against ``dim_len`` exactly as Python does for ``range``:
      ``None`` and negative bounds are resolved with the sign of the step in mind, bounds are clipped to the dimension
    :raises ValueError: if the step is zero
    """
    return slice_.indices(dim_len)


def create_shape_from_slice(
    array_shape: tuple[int, ...],
    index_exp: Slice,
) -> tuple[int, ...]:
    """Calculate shape of a subset from the index expression passed to __getitem__.

    An integer index drops its axis; a slice keeps it with the number of the positions it selects, as
    ``range`` counts them, for any sign of the step; the axes the expression leaves out are taken whole.

    :param array_shape: shape of the parent array
    :param index_exp: index expression passed to the array __getitem__ method
    :returns: shape of the subset selected by ``index_exp``
    :raises IndexError: if an index is of another kind, there are too many of them, more than one ellipsis,
      or a slice step is zero
    """
    if (
        isinstance(index_exp, EllipsisType)
        or (isinstance(index_exp, slice) and index_exp == slice(None, None, None))
        or (isinstance(index_exp, tuple) and not index_exp)
    ):
        return array_shape

    # the same normalisation ``np.index_exp[...]`` does: a tuple stays as is, a single index becomes a 1-tuple
    expressions: list[slice | int | EllipsisType | None]
    if isinstance(index_exp, tuple):
        expressions = list(index_exp)
    elif isinstance(index_exp, (slice, int)):
        expressions = [index_exp]
    else:
        raise IndexError(f"Only integers, slices, ellipsis and tuples of them are valid indices, got {index_exp!r}")

    len_shape: int = len(array_shape)
    ellipses = sum(isinstance(exp, EllipsisType) for exp in expressions)
    if ellipses > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    len_item: int = len(expressions) - ellipses

    if len_item > len_shape:
        raise IndexError(f"Too many indices for array: array is {len_shape}-dimensional, but {len_item} were indexed")

    # an ellipsis stands for every axis the other indices leave out, as in numpy; without one they go to the end
    whole_axes = [slice(None, None, None) for _ in range(len_shape - len_item)]
    if ellipses:
        position = next(i for i, exp in enumerate(expressions) if isinstance(exp, EllipsisType))
        expressions[position : position + 1] = whole_axes
    else:
        expressions.extend(whole_axes)

    normalized: list[slice | None] = []
    for exp in expressions:
        if isinstance(exp, slice):
            if exp.step == 0:
                raise IndexError("slice step cannot be zero")
            normalized.append(exp)
        elif isinstance(exp, int):
            normalized.append(None)  # an integer index removes the dimension
        else:
            raise IndexError(f"Only integers, slices and ellipsis are valid indices, got {exp!r}")

    return tuple(
        len(range(*match_slice_size(array_shape[i], exp))) for i, exp in enumerate(normalized) if exp is not None
    )


def expand_index_exp(array_shape: tuple[int, ...], index_exp: Slice) -> tuple[slice | int, ...]:
    """Expand an index expression to one slice or index per axis of the array.

    A single index becomes a 1-tuple, an ellipsis stands for every axis the other indices leave out and the axes
    left out without an ellipsis are taken whole, as in numpy. Nothing else is changed: negative indices
    and slice defaults stay as they are.

    :param array_shape: shape of the array
    :param index_exp: index expression passed to the array ``__getitem__`` method
    :returns: one slice or integer per axis
    :raises IndexError: if there are too many indices or more than one ellipsis
    """
    raw: tuple = index_exp if isinstance(index_exp, tuple) else (index_exp,)
    items: list[slice | int | EllipsisType] = []
    for item in raw:
        if not isinstance(item, (slice, int, EllipsisType)) or isinstance(item, bool):
            raise IndexError(f"Only integers, slices and ellipsis are valid indices, got {item!r}")
        items.append(item)

    ellipses = sum(isinstance(item, EllipsisType) for item in items)
    if ellipses > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    given = len(items) - ellipses
    if given > len(array_shape):
        raise IndexError(
            f"Too many indices for array: array is {len(array_shape)}-dimensional, but {given} were indexed"
        )
    whole_axes: list[slice | int | EllipsisType] = [slice(None, None, None) for _ in range(len(array_shape) - given)]
    if ellipses:
        position = next(i for i, item in enumerate(items) if isinstance(item, EllipsisType))
        items[position : position + 1] = whole_axes
    else:
        items.extend(whole_axes)
    return tuple(item for item in items if not isinstance(item, EllipsisType))


def positive_bounds(array_shape: tuple[int, ...], index_exp: Slice) -> tuple[tuple[slice | int, ...], tuple[int, ...]]:
    """Replace every slice with a negative step by the slice that selects the same cells forwards.

    A slice with a negative step selects the same cells as its forward twin, in the reversed order. Storages
    that select forwards only (HDF5 hyperslabs, for one) read the twin and flip the data along the axis;
    the data to write is flipped the same way first.

    :param array_shape: shape of the array
    :param index_exp: index expression passed to the array ``__getitem__`` method
    :returns: the forward index expression, one slice or index per axis, and the axes of the selected data
        (an integer index drops its axis) that have to be flipped
    """
    forward: list[slice | int] = []
    flipped_axes: list[int] = []
    data_axis = 0
    for size, item in zip(array_shape, expand_index_exp(array_shape, index_exp)):
        if isinstance(item, slice) and item.step is not None and item.step < 0:
            start, stop, step = item.indices(size)
            selected = range(start, stop, step)
            if selected:
                forward.append(slice(selected[-1], selected[0] + 1, -step))
                flipped_axes.append(data_axis)
            else:
                forward.append(slice(0, 0))
        else:
            forward.append(item)
        if isinstance(item, slice):
            data_axis += 1
    return tuple(forward), tuple(flipped_axes)


class SliceConversionError(Exception):
    """If something goes wrong during slice conversion."""

    pass


class _StringEscape(type):
    """Sets escape symbol for string."""

    _string_escape = "`"
    _isostring_regex = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?((\+|-)(\d{2}:\d{2}))?)")
    # ISO 8601 duration with days, hours, minutes and seconds; the lookaheads reject "P", "PT" and a trailing "T"
    _duration_regex = re.compile(
        r"(?P<sign>-)?P(?!T?$)(?:(?P<days>\d+)D)?"
        r"(?:T(?!$)(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+(?:\.\d+)?)S)?)?"
    )

    @classmethod
    def _parse_duration(cls, string: str) -> datetime.timedelta:
        """Parse an ISO 8601 duration (days, hours, minutes, seconds; optional leading minus).

        :param string: duration string, e.g. ``"PT3H"``, ``"P1DT12H30M"``, ``"-PT0.5S"``
        :returns: the duration as ``timedelta``
        :raises ValueError: if the string is not a duration or has no components
        """
        match = cls._duration_regex.fullmatch(string)
        if not match:  # pragma: no cover  # guard: callers check the regex first
            raise ValueError(f"Invalid ISO 8601 duration: {string!r}")
        duration = datetime.timedelta(
            days=int(match["days"] or 0),
            hours=int(match["hours"] or 0),
            minutes=int(match["minutes"] or 0),
            seconds=float(match["seconds"] or 0),
        )
        return -duration if match["sign"] else duration

    @classmethod
    def _format_duration(cls, duration: datetime.timedelta) -> str:
        """Write a ``timedelta`` as an ISO 8601 duration (days, hours, minutes, seconds).

        :param duration: duration to write
        :returns: duration string, e.g. ``"PT3H"``, ``"P1DT12H30M"``, ``"-PT0.5S"``; zero is ``"PT0S"``
        """
        sign = "-" if duration < datetime.timedelta(0) else ""
        duration = abs(duration)
        hours, remainder = divmod(duration.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        date_part = f"{duration.days}D" if duration.days else ""
        time_part = ""
        if hours:
            time_part += f"{hours}H"
        if minutes:
            time_part += f"{minutes}M"
        if seconds or duration.microseconds:
            time_part += f"{seconds + duration.microseconds / 1_000_000:g}S" if duration.microseconds else f"{seconds}S"
        if not date_part and not time_part:
            time_part = "0S"
        return f"{sign}P{date_part}{'T' + time_part if time_part else ''}"

    @classmethod
    def _wrap_with_escape(cls, string: str) -> str:
        """Enclose string with escape symbols.

        :param string: string to be wrapped
        :returns: string enclosed in escape symbols
        """
        return f"{cls._string_escape}{string}{cls._string_escape}"

    @classmethod
    def _is_wrapped_with_escape(cls, string: str) -> bool:
        """Check if string is wrapped with escape symbols.

        :param string: string to be checked
        :returns: ``True`` if the string starts and ends with the escape symbol
        """
        return string.startswith(cls._string_escape) and string.endswith(cls._string_escape)

    @classmethod
    def _unwrap(cls, string: str) -> str:
        """Disclose string from escape symbols.

        :param string: string to be unwrapped
        :returns: string without the enclosing escape symbols
        """
        return string.strip(cls._string_escape)


class _StringToSliceMixin(type, metaclass=_StringEscape):
    """Converts a slice string to slice objects.

    The parser only translates tokens into Python values; whether a value makes sense for a particular
    dimension is decided by the array that receives the slice.
    """

    # quoted segments and datetime isostrings are swapped for placeholders while splitting,
    # so that the separators inside them (":" and ",") do not split
    _placeholder = "\x00"
    _protected_regex = re.compile(r"`[^`]*`|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:[+-]\d{2}:\d{2})?")

    @classmethod
    def _parse_literal(cls, token: str) -> int | float | EllipsisType | tuple[()] | None:
        """Parse a token as a Python literal of one of the allowed forms.

        Allowed forms: an ``int`` or ``float`` constant with an optional single unary minus, ``None``, ``...``
        and the empty tuple ``()``. Anything else (``bool``, names, containers, quoted strings, double signs)
        is not a literal of a slice string.

        :param token: stripped token
        :returns: parsed value
        :raises ValueError: if the token is not a literal of an allowed form
        """
        try:
            node = ast.parse(token, mode="eval").body
        except SyntaxError:
            raise ValueError(f"Not a literal: {token!r}")

        if isinstance(node, ast.Constant):
            value = node.value
            if value is None or value is Ellipsis:
                return value
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            operand = node.operand.value
            if isinstance(operand, (int, float)) and not isinstance(operand, bool):
                return -operand
        elif isinstance(node, ast.Tuple) and not node.elts:
            return ()
        raise ValueError(f"Not a literal: {token!r}")

    @classmethod
    def _process_string(cls, string: str) -> SliceUnit:
        """Convert one token of a slice string into a value.

        The order matters:

        1. a token enclosed in escape symbols is a string, whatever it contains;
        2. an ISO 8601 datetime (``2023-06-11T00:00:00+00:00``) is kept as a string;
        3. an ISO 8601 duration (``PT3H``, ``-P1DT12H``) becomes a ``timedelta``;
        4. an allowed literal becomes ``int``, ``float``, ``None``, ``...`` or ``()``;
        5. anything else is a label: a plain string, kept as is.

        :param string: token to be processed
        :returns: ``None``, ``Ellipsis``, ``()``, a number, a ``timedelta`` or a string
        """
        if not isinstance(string, str):  # pragma: no cover  # guard for non-str input
            raise TypeError(f"Invalid unit '{string}' type: {type(string)}")

        string = string.strip()
        if not string:
            return None

        if cls._is_wrapped_with_escape(string) and string.count(cls._string_escape) == 2:
            return cls._unwrap(string)

        if cls._isostring_regex.fullmatch(string):
            return string

        if cls._duration_regex.fullmatch(string):
            return cls._parse_duration(string)

        try:
            return cls._parse_literal(string)
        except ValueError:
            return string

    @classmethod
    def _protect(cls, string: str) -> tuple[str, list[str]]:
        """Replace quoted segments and ISO datetimes with placeholders.

        After that the string can be split by ``","`` and ``":"`` with plain ``str.split``: the separators
        inside quotes and datetimes are hidden in the placeholders.

        :param string: contents of the slice string without the brackets
        :returns: the string with placeholders and the list of the protected segments, in order
        """
        protected: list[str] = []
        pieces: list[str] = []
        last = 0
        for match in cls._protected_regex.finditer(string):
            pieces.append(string[last : match.start()])
            pieces.append(f"{cls._placeholder}{len(protected)}{cls._placeholder}")
            protected.append(match.group(0))
            last = match.end()
        pieces.append(string[last:])
        return "".join(pieces), protected

    @classmethod
    def _restore(cls, token: str, protected: list[str]) -> str:
        """Put the protected segments back into one token.

        :param token: token with placeholders
        :param protected: segments returned by :meth:`_protect`
        :returns: token with the original segments
        """
        for index, original in enumerate(protected):
            token = token.replace(f"{cls._placeholder}{index}{cls._placeholder}", original)
        return token

    @classmethod
    def _str_to_slices(cls, slice_: str) -> FancySlice:
        """Convert a slice string to a value, a slice or a tuple of them.

        :param slice_: string to convert
        :returns: slice, value or a tuple of them, parsed from the string
        :raises SliceConversionError: if the string is not enclosed in square brackets, is empty,
          or an element has more than three ``:``-separated parts
        """
        if not slice_.startswith("[") or not slice_.endswith("]"):
            raise SliceConversionError(f"Invalid slice string: {slice_}; shall be enclosed in brackets: '[{slice_}]'")
        body: str = slice_[1:-1]
        if not body or body.isspace():
            raise SliceConversionError(f"Invalid slice string: {slice_} is empty")

        body, protected = cls._protect(body)
        res: list = []
        for element in body.split(","):
            parts = element.split(":")
            if len(parts) > 3:
                element = cls._restore(element, protected)
                raise SliceConversionError(f"Invalid slice string: {element}; a slice has at most start, stop and step")
            tokens = [cls._process_string(cls._restore(part, protected)) for part in parts]
            # a colon outside quotes and datetimes means a slice, anything else is a value
            res.append(slice(*tokens) if len(tokens) > 1 else tokens[0])

        return tuple(res) if len(res) != 1 else res[0]


class _SliceToStringMixin(type, metaclass=_StringEscape):
    """Converts values and slices to a slice string."""

    # maps default non-numeric slices values
    _slice_to_str = {None: ":", ...: "...", (): "()"}

    @classmethod
    def _normalize_unit(cls, value: SliceUnitInput) -> SliceUnit:
        """Bring one index unit to the Python type the string form is written from.

        numpy scalars are accepted and converted: integers and floats to ``int``/``float``, ``datetime64``
        to ``datetime`` (microsecond precision, naive), ``timedelta64`` to ``timedelta``.

        :param value: unit of an index expression
        :returns: the same value as a plain Python type
        :raises TypeError: for ``bool``, ``nan``/``inf``, ``NaT`` and any type the string form cannot represent
        """
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("bool is not a valid index")
        if value is None or value is Ellipsis:
            return value
        if isinstance(value, (str, datetime.datetime, datetime.timedelta)):
            return value
        # numpy date/time scalars go BEFORE the numbers check: np.timedelta64 is registered as numbers.Integral
        if isinstance(value, np.datetime64):
            if np.isnat(value):
                raise TypeError("NaT is not a valid index")
            return value.astype("datetime64[us]").astype(datetime.datetime)
        if isinstance(value, np.timedelta64):
            if np.isnat(value):
                raise TypeError("NaT is not a valid step")
            return value.astype("timedelta64[us]").item()
        if isinstance(value, numbers.Integral):  # int, np.int64, np.int32, ...
            return int(value)
        if isinstance(value, numbers.Real):  # float, np.float64, np.float32, ...
            value = float(value)
            if not math.isfinite(value):
                raise TypeError("nan and inf are not valid indices")
            return value
        raise TypeError(f"{type(value).__name__} is not representable in a slice string")

    @classmethod
    def _unit_to_str(cls, value: SliceUnit) -> str:
        """Write one normalized unit in the string form.

        :param value: normalized unit
        :returns: string representation of the unit
        """
        if value is None:
            return ""
        if value is Ellipsis or value == ():
            return cls._slice_to_str[value]
        if isinstance(value, str):
            return cls._wrap_with_escape(value)
        if isinstance(value, datetime.datetime):
            return cls._wrap_with_escape(value.isoformat())
        if isinstance(value, datetime.timedelta):
            return cls._format_duration(value)
        return str(value)

    @classmethod
    def _convert_element(cls, element: SliceUnitInput | slice) -> str:
        """Convert one element of the index expression to its string form.

        :param element: value or slice to convert
        :returns: string representation of one index expression element
        """
        if isinstance(element, slice):
            parts = [
                cls._unit_to_str(cls._normalize_unit(getattr(element, attr))) for attr in ("start", "stop", "step")
            ]
            slice_string = ":".join(parts)
            if slice_string.endswith(":"):
                slice_string = slice_string[:-1]
            return slice_string
        unit = cls._normalize_unit(element)
        if unit is None:
            return cls._slice_to_str[None]
        return cls._unit_to_str(unit)

    @classmethod
    def _slices_to_str(cls, slice_: FancySliceInput) -> str:
        """Convert a value, a slice or a tuple/list of them to a slice string.

        :param slice_: value, slice, or a tuple/list of values and slices
        :returns: slice string enclosed in square brackets
        :raises TypeError: if a container other than ``tuple``/``list`` is passed
        """
        if isinstance(slice_, (tuple, list)):
            if not slice_:
                return f"[{cls._slice_to_str[()]}]"
            return f"[{', '.join(cls._convert_element(element) for element in slice_)}]"
        if isinstance(slice_, Iterable) and not isinstance(slice_, str):
            raise TypeError(f"{type(slice_).__name__} is not representable in a slice string; use a tuple or a list")
        return f"[{cls._convert_element(slice_)}]"


class _SliceConverter(_SliceToStringMixin, _StringToSliceMixin):
    """Converts slices to string and vice versa. Check slice_converter class for interface."""

    def __getitem__(cls, item: FancySliceInput | str) -> str | FancySlice:
        """Call slices_to_str or str_to_slices depending on item type.

        :param item: item to convert
        :returns: slice string for a slice object, or slice object(s) for a slice string
        :raises SliceConversionError: if the item cannot be converted
        """
        if isinstance(item, str):
            return str_to_slices(item)
        return slices_to_str(item)


def slices_to_str(slice_: FancySliceInput) -> str:
    """Convert an index expression to its string form, the same way ``slice_converter[...]`` does.

    :param slice_: value, slice, or a tuple/list of values and slices
    :returns: slice string enclosed in square brackets
    :raises SliceConversionError: if the expression cannot be converted
    """
    try:
        if isinstance(slice_, (tuple, list)) and sum(unit is Ellipsis for unit in slice_) > 1:
            raise IndexError("An index can only have a single ellipsis ('...')")
        return _SliceConverter._slices_to_str(slice_)
    except Exception as e:
        raise SliceConversionError(e)


def str_to_slices(string: str) -> FancySlice:
    """Convert a slice string back to an index expression, the same way ``slice_converter["..."]`` does.

    :param string: slice string enclosed in square brackets
    :returns: a single index or a tuple of them
    :raises SliceConversionError: if the string cannot be parsed
    """
    try:
        return _SliceConverter._str_to_slices(string)
    except Exception as e:
        raise SliceConversionError(e)


class slice_converter(metaclass=_SliceConverter):  # noqa: N801
    """Converts slices to string and vice versa.

    ``slice_converter[index expression]`` returns the string form of the expression, enclosed in square brackets;
    ``slice_converter["[...]"]`` reads such a string back into a single index or a tuple of them. The string form
    is what the server adapters send. The same conversions are the functions :func:`slices_to_str` and
    :func:`str_to_slices`.

    Writing accepts ``int``, ``float``, ``str``, ``datetime``, ``timedelta``, NumPy numbers and dates, ``None``,
    ``...``, slices of them and a tuple or a list of all these. Booleans, ``nan``, infinity and containers other
    than a tuple or a list raise :class:`SliceConversionError`.

    Reading returns numbers as ``int`` or ``float``, an ISO 8601 duration (``PT3H``, ``-PT3H``, ``P1D``) as
    ``timedelta``, ``None``, ``...`` and ``()`` as themselves, and everything else, dates included, as a string
    label: a date is not turned into ``datetime``, the dimension of the array interprets the label, and a time
    dimension reads ISO 8601 strings. Backticks are needed only around a label that would otherwise read as
    a number, a date or a duration; any label may be quoted. A string that cannot be read raises
    :class:`SliceConversionError`.

    Standard index expressions
        >>> slice_converter[5]
        '[5]'
        >>> slice_converter[1, :]
        '[1, :]'
        >>> slice_converter["[1, :]"]
        (1, slice(None, None, None))

    Index expressions with floats (e.g. a dimension with a scale)
        >>> slice_converter[-90.0:90.0:0.5, 0.25]
        '[-90.0:90.0:0.5, 0.25]'
        >>> slice_converter["[-90.0:90.0:0.5, 0.25]"]
        (slice(-90.0, 90.0, 0.5), 0.25)

    Index expressions with datetime
        >>> slice_converter[datetime.datetime(2023,1,1):datetime.datetime(2023,2,1), 5]
        '[`2023-01-01T00:00:00`:`2023-02-01T00:00:00`, 5]'
        >>> slice_converter["[`2023-01-01T00:00:00`:`2023-02-01T00:00:00`, 5]"]
        (slice('2023-01-01T00:00:00', '2023-02-01T00:00:00', None), 5)

    Timezone and microseconds are also acceptable
        >>> slice_converter[datetime.datetime(2023,1,1,0,0,0,123456)]
        '[`2023-01-01T00:00:00.123456`]'
        >>> slice_converter['[`2023-01-01T00:00:00-03:30`]']
        '2023-01-01T00:00:00-03:30'
        >>> slice_converter['[`2023-01-01T00:00:00.123456+05:00`]']
        '2023-01-01T00:00:00.123456+05:00'

    Index expressions with a duration (ISO 8601) as a step
        >>> slice_converter[datetime.datetime(2023,1,1):datetime.datetime(2023,2,1):datetime.timedelta(hours=3)]
        '[`2023-01-01T00:00:00`:`2023-02-01T00:00:00`:PT3H]'
        >>> slice_converter["[`2023-01-01T00:00:00`:`2023-02-01T00:00:00`:-PT3H]"]
        slice('2023-01-01T00:00:00', '2023-02-01T00:00:00', datetime.timedelta(days=-1, seconds=75600))

    Index expressions with labels (strings)
        >>> slice_converter["temperature":"humidity":2, "pressure"]
        '[`temperature`:`humidity`:2, `pressure`]'
        >>> slice_converter['[`temperature`:`humidity`:2, `pressure`]']
        (slice('temperature', 'humidity', 2), 'pressure')

    Quotes are optional for reading, unless the label looks like a number, a date or a duration
        >>> slice_converter['[temperature:humidity:2, pressure]']
        (slice('temperature', 'humidity', 2), 'pressure')
        >>> slice_converter['[`10`, 10]']
        ('10', 10)
    """
