"""Tools for working with slices and index expressions."""

import builtins
import datetime
import re

from typing import Iterable, List, Optional, Tuple, Union

import numpy as np

from numpy.lib.index_tricks import IndexExpression


__all__ = ["match_slice_size", "create_shape_from_slice", "slice_converter", "SliceConversionError"]

Slice = Union[  # type: ignore[valid-type]
    IndexExpression, slice, type(Ellipsis), int, Tuple[Union[slice, int, type(Ellipsis), None], ...]
]

FancySlice = Union[  # type: ignore[valid-type]
    IndexExpression,
    slice,
    type(Ellipsis),
    int,
    float,
    str,
    datetime.datetime,
    Tuple[Union[slice, int, float, str, datetime.datetime, type(Ellipsis), None], ...],
]


def match_slice_size(dim_len: int, slice_: Optional[slice] = None) -> Tuple[int, int, int]:
    """Convert slice into a sequence and get its length.

    :param dim_len: length of the corresponding Dimension
    :param slice_: slice to be converted
    """
    if slice_ is None:
        start, stop, step = 0, dim_len, 1
    else:
        start = 0 if slice_.start is None else (slice_.start if slice_.start >= 0 else dim_len + slice_.start)
        stop = dim_len if slice_.stop is None else (slice_.stop if slice_.stop >= 0 else dim_len + slice_.stop)
        step = 1 if slice_.step is None else slice_.step
    return start, stop, step


# TODO: [Optimization]
def create_shape_from_slice(
    array_shape: Tuple[int, ...], index_exp: Slice  # type: ignore[valid-type]
) -> Tuple[int, ...]:
    """Calculate shape of a subset from the index expression passed to __getitem__.

    :param array_shape: shape of the parent array
    :param index_exp: index expression passed to the array __getitem__ method
    """
    if (
        isinstance(index_exp, type(builtins.Ellipsis))
        or (isinstance(index_exp, slice) and index_exp == slice(None, None, None))
        or (isinstance(index_exp, tuple) and not index_exp)
    ):
        return array_shape

    index_exp: List[Optional[slice]] = list(np.index_exp[index_exp])  # type: ignore[arg-type]

    len_item: int = len(index_exp)
    len_shape: int = len(array_shape)

    if len_item > len_shape:
        raise IndexError(f"Too many indices for array: array is {len_shape}-dimensional, but {len_item} were indexed")

    if len_item < len_shape:
        for _ in range(len_shape - len_item):
            index_exp.append(slice(None, None, None))

    exclude: List[int] = []
    if all(isinstance(i, (slice, type(builtins.Ellipsis), int)) for i in index_exp):
        for n, i in enumerate(index_exp):  # type: ignore[arg-type]
            if isinstance(i, slice) and i.step is not None and i.step != 1:
                raise IndexError("step should be equal to 1")
            if isinstance(i, type(builtins.Ellipsis)):
                index_exp[n] = slice(None, None, None)
            if isinstance(i, int):
                exclude.append(n)
        if exclude:
            exclude.sort()
            exclude.reverse()
            for p in exclude:
                index_exp[p] = None  # type: ignore[union-attr, index]
    del exclude
    shape = tuple(
        len(tuple(range(*match_slice_size(array_shape[i], exp)))) for i, exp in enumerate(index_exp) if exp is not None
    )

    return shape


class SliceConversionError(Exception):
    """If something goes wrong during slice conversion."""

    pass


class _StringEscape(type):
    """Sets escape symbol for string."""

    _string_escape = "`"

    @classmethod
    def _wrap_with_escape(cls, string: str) -> str:
        """Enclose string with escape symbols.

        :param string: string to be wrapped
        """
        return f"{cls._string_escape}{string}{cls._string_escape}"

    @classmethod
    def _is_wrapped_with_escape(cls, string: str) -> bool:
        """Check if string is wrapped with escape symbols.

        :param string: string to be checked
        """
        return string.startswith(cls._string_escape) and string.endswith(cls._string_escape)

    @classmethod
    def _unwrap(cls, string: str) -> str:
        """Disclose string from escape symbols.

        :param string: string to be unwrapped
        """
        return string.strip(cls._string_escape)


class _StringToSliceMixin(type, metaclass=_StringEscape):
    """Converts string to slices."""

    # maps default non-numeric slices values
    _str_to_slice = {"None": None, "": None, "...": ..., "()": ()}
    _isostring_regex = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?((\+|-)(\d{2}:\d{2}))?)")

    @classmethod
    def _process_string(cls, string: str) -> Optional[Union[int, float, str, slice]]:
        """Validate and get value from string.

        :param string: string to be processed
        """
        string = string.strip()
        str_for_check = string

        # try to validate negative numbers
        if string.startswith("-"):
            str_for_check = string[1:]

        # check if string is None or ...
        if str_for_check in cls._str_to_slice:
            return cls._str_to_slice[string]  # type: ignore[return-value]

        # check if string is empty
        if str_for_check is False or str_for_check.isspace():
            return None

        # check if string is integer
        if str_for_check.isdigit():
            return int(string)

        # check if string is float
        if str_for_check.replace(".", "").isdigit():
            return float(string)

        if isinstance(str_for_check, str):
            if cls._is_wrapped_with_escape(str_for_check):
                if str_for_check.count(cls._string_escape) == 2:
                    str_for_check = cls._unwrap(str_for_check)
            return str_for_check

        raise TypeError(f"Invalid unit '{string}' type: {type(string)}")

    @classmethod
    def _convert_str_to_slice(cls, slice_string: str) -> slice:
        """Convert a slice string to a slice object.

        :param slice_string: slice string to convert
        """
        slice_parameters = []
        for i in slice_string.split(":"):
            pos = cls._process_string(i.strip())
            slice_parameters.append(pos)
        return slice(*slice_parameters)

    @classmethod
    def _parse_datetime(cls, slice_string: str, match_list: list) -> Union[slice, str]:
        """Parse datetime isostrings and convert them to a slice object.

        :param slice_string: initial slice string to convert
        :param match_list: list of matched datetime isostrings to convert
        """
        slice_string = slice_string.replace(cls._string_escape, "").strip()
        slice_parameters = [i[0] for i in match_list if i]
        length = 3
        slice_parameters_length = len(slice_parameters)
        if len(slice_parameters) < length:
            first: str = slice_parameters[0]
            split_by_time = slice_string.split(first)
            if len(slice_parameters) == 1 and (
                len(split_by_time) == 1 or (len(split_by_time) == 2 and all(not s for s in split_by_time))
            ):
                return first
            if not slice_string.startswith(first):
                split = slice_string.split(first)
                start = split[0].strip(cls._string_escape).strip(":").strip(cls._string_escape).strip()
                slice_parameters.insert(0, cls._process_string(start))
                if split[-1] != split[0]:
                    step = split[-1].strip(cls._string_escape).strip(":").strip(cls._string_escape).strip()
                    slice_parameters.append(cls._process_string(step))
            else:
                slice_parameters.extend(None for _ in range(length - slice_parameters_length))

        return slice(*slice_parameters)

    @classmethod
    def _str_to_slices(cls, slice_: str) -> FancySlice:  # type: ignore[valid-type]
        """Convert a slice string to a list of slices.

        :param slice_: string to convert
        """
        # split the input string into slice strings and convert each one to a slice object
        if not slice_.startswith("[") or not slice_.endswith("]"):
            raise SliceConversionError(f"Invalid slice string: {slice_}; shall be enclosed in brackets: '[{slice_}]'")
        slices_str_no_brackets: str = slice_[1:-1]
        if not slices_str_no_brackets or slices_str_no_brackets.isspace():
            raise SliceConversionError(f"Invalid slice string: {slice_} is empty")

        slices_str = slices_str_no_brackets.split(",")

        res: list = []
        for slice_str in slices_str:
            # check if string is an ordinary string, datetime or slice
            # ":" may appear both in slices and datetime isostrings
            if ":" in slice_str:
                match = re.findall(cls._isostring_regex, slice_str)
                if match:  # dimension represents datetime
                    pos = cls._parse_datetime(slice_str, match)  # type: ignore[assignment]
                else:  # all other dimensions represent any other values except time
                    pos = cls._convert_str_to_slice(slice_str)  # type: ignore[assignment]
            else:
                pos = cls._process_string(slice_str)  # type: ignore[assignment]
            res.append(pos)

        return tuple(res) if len(res) != 1 else res[0]  # type: ignore[no-any-return]


class _SliceToStringMixin(type, metaclass=_StringEscape):
    """Converts indexes and slices to string."""

    # maps default non-numeric slices values
    _slice_to_str = {None: ":", ...: "...", (): "()"}

    @classmethod
    def _slices_to_str(cls, slice_: Optional[FancySlice]) -> str:  # type: ignore[valid-type]
        """Convert a slice object(s) to a slice string.

        :param slice_: slice or slices tuple to convert
        """

        def convert_slice(_slice_: Optional[FancySlice]) -> str:  # type: ignore[valid-type]
            """Convert a slice object to a slice string.

            :param _slice_: slice to convert
            """
            if _slice_ is None or _slice_ is Ellipsis:
                return cls._slice_to_str[_slice_]

            slice_parameters = []
            if isinstance(_slice_, slice):
                for attr in ("start", "stop", "step"):
                    el = getattr(_slice_, attr)
                    if isinstance(el, datetime.datetime):
                        el = cls._wrap_with_escape(el.isoformat())
                    elif isinstance(el, str):
                        el = cls._wrap_with_escape(el)
                    slice_parameters.append(str(el))

                slice_string = ":".join(slice_parameters).replace("None", "")
                if slice_string.endswith(":"):
                    slice_string = slice_string[:-1]
                return slice_string

            if isinstance(_slice_, str):
                return cls._wrap_with_escape(_slice_)
            if isinstance(_slice_, (int, float)):
                return str(_slice_)
            if isinstance(_slice_, datetime.datetime):
                return cls._wrap_with_escape(_slice_.isoformat())

            raise TypeError(f"Invalid unit '{_slice_}' type: {type(_slice_)}")

        if slice_ == ():
            return f"[{cls._slice_to_str[slice_]}]"

        if isinstance(slice_, Iterable):
            slice_strs = [convert_slice(sl) for sl in slice_]  # type: ignore[union-attr]
            return f"[{', '.join(slice_strs)}]"
        return f"[{convert_slice(slice_)}]"


class _SliceConverter(_SliceToStringMixin, _StringToSliceMixin):
    """Converts slices to string and vice versa. Check slice_converter class for interface."""

    def __getitem__(cls, item: Union[FancySlice, str]) -> Union[str, FancySlice]:  # type: ignore[valid-type]
        """Call slices_to_str or str_to_slices depending on item type.

        :param item: item to convert
        """
        try:
            if isinstance(item, tuple) and item.count(...) > 1:
                raise IndexError("An index can only have a single ellipsis ('...')")
            if isinstance(item, str):
                return cls._str_to_slices(item)
            return cls._slices_to_str(item)
        except Exception as e:
            raise SliceConversionError(e)


class slice_converter(object, metaclass=_SliceConverter):  # noqa: N801
    """Converts slices to string and vice versa.

    Standard index expressions
        >>> slice_converter[5]
        '[5]'
        >>> slice_converter[1, :]
        '[1, :]'
        >>> slice_converter["[1, :]"]
        (1, slice(None, None, None))

    Index expressions with datetime and floats
        >>> slice_converter[datetime.datetime(2023,1,1):datetime.datetime(2023,2,1), 0.1:0.9:0.05]
        '[`2023-01-01T00:00:00`:`2023-02-01T00:00:00`, 0.1:0.9:0.05]'
        >>> slice_converter["[`2023-01-01T00:00:00`:`2023-02-01T00:00:00`, 0.1:0.9:0.05]"]
        (slice('2023-01-01T00:00:00', '2023-02-01T00:00:00', None), slice(0.1, 0.9, 0.05))

    Timezone and microseconds are also acceptable
        >>> slice_converter[datetime.datetime(2023,1,1,0,0,0,123456)]
        '[`2023-01-01T00:00:00.123456`]'
        >>> slice_converter['[`2023-01-01T00:00:00-03:30`]']
        '2023-01-01T00:00:00-03:30'
        >>> slice_converter['[`2023-01-01T00:00:00.123456+05:00`]']
        '2023-01-01T00:00:00.123456+05:00'

    Index expressions with strings
        >>> slice_converter["1a":"10b":"5c"]
        '[`1a`:`10b`:`5c`]'
        >>> slice_converter['[1a:10b:5c]']
        slice('1a', '10b', '5c')
        >>> slice_converter["1":"10":"5.2"]
        '[`1`:`10`:`5.2`]'
        >>> slice_converter["1", "10", "5.2"]
        '[`1`, `10`, `5.2`]'
        >>> slice_converter['[`1`, `10`, `5.2`]']
        ('1', '10', '5.2')
        >>> slice_converter['[`1`:`10`:`5.2`]']
        slice('1', '10', '5.2')
    """
