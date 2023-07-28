Deker Tools
===========

Deker Tools is a collection of utility functions and classes designed to assist in common data processing tasks. It consists of modules that handle data conversion, path validation, and slice manipulation.

data
----

This module provides `convert_size_to_human` method for converting data size units::

    > convert_size_to_human(1052810)
    "1.0 MB"

path
----

This module provides functions to validate and handle filesystem paths::

    is_empty(path)
    is_path_valid(path)

slices
------
Calculate shape of a subset from the index expression::

    > shape = (361, 720, 4)
    > index_exp = (slice(None, None, None), slice(None, None, None), 0)
    > create_shape_from_slice(shape, index_exp)
    (361, 720)

Convert slice into a sequence and get its length::

    > match_slice_size(10, slice(10))
    (0, 10, 1)

Convert slices to string and vice versa with slice_converter::

    > slice_converter[5]
    '[5]'

    > slice_converter[datetime.datetime(2023,1,1):datetime.datetime(2023,2,1), 0.1:0.9:0.05]
    '[`2023-01-01T00:00:00`:`2023-02-01T00:00:00`, 0.1:0.9:0.05]'

time
------

This module provides `get_utc` function which returns timezone with UTC or current time by default::

    > get_utc()
    2023-07-26 15:42:05.539317+00:00

    > get_utc(datetime.now())
    2023-07-26 15:42:05.539317+00:00

.. toctree::
   :maxdepth: 4
   :hidden:

   Deker tools API <source/api/modules>
