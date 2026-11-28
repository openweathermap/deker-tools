# Changelog

All notable changes to deker-tools, the helpers shared by [DEKER™](https://github.com/openweathermap/deker)
and its adapters, are listed here, release by release, newest first. Only production releases are
listed. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), the versions
follow [Semantic Versioning](https://semver.org/).

## [2.0.0]

Released together with DEKER™ 2.0.0. The parsing of slices changed incompatibly, so every
consumer (`deker` and the adapters) pins `deker-tools>=2,<3` and is released with it.

### Breaking changes

- A step in a slice is allowed: any non-zero integer, negative included, in
  `create_shape_from_slice` and the other shape helpers. A zero step raises `ValueError`.
- The bounds of a slice follow the rules of Python (`slice.indices()`): empty and negative bounds,
  bounds beyond the dimension and the sign of the step are handled as NumPy does.
- `slices_to_str` accepts only what `str_to_slices` reads back: `int`, `float`, `str`,
  `datetime`, `timedelta`, NumPy numbers and `tuple` or `list` of them. Booleans, `nan`, infinity,
  `NaT`, arrays, sets and dicts raise `ValueError`. `None` inside labels is kept.
- `str_to_slices` splits the string outside quotes and dates only, parses every token with the
  Python parser against a whitelist of forms, turns an ISO 8601 duration (signed) into a
  `timedelta` and keeps everything unrecognised as a label. Negative numbers in labels keep their
  sign; a colon or a comma inside quotes does not split; the step after two dates is not lost.
- `is_empty` and `is_path_valid` raise `NotADirectoryError` for a path that is not a directory, instead
  of `IsADirectoryError`.
- An argument that is not given is `Sentinel`, not `None`: `get_utc()` without a moment returns the current
  time, `get_utc(None)` raises `TypeError`; `match_slice_size(dim_len)` without a slice covers the whole
  dimension, `None` is not accepted there anymore.

### Added

- `deker_tools.typings`: `Sentinel`, the value of an argument that is not given, its type `SentinelType` and
  `is_sentinel`. Every DEKER™ package takes them from here.
- `deker_tools.cancel`: the cancellation tokens of the operations, built on
  [cantok](https://github.com/pomponchik/cantok). `read_token` and `write_token` build the token of a read
  (its timeout and the cancellation scope) and of a write (the scope until the first file of the write is
  started, see `mark_started`); `check` and `cancellable` stop an operation at its cancellation points with
  `TokenCancelledError` or `TokenTimeoutError`; `cancellation_scope` lets an integration cancel the operations
  it starts; `resolve_timeout` takes the timeout of a call or the default.
- The dependency `cantok>=0.0.43,<1`.
- `deker_tools.hash_ring`: the consistent hash ring of a cluster, `HashRing`, and the key of an array on it,
  `get_hash_key` (by the primary attributes, or by the id when there are none), moved here from the server
  adapters. The placement of the keys is the same as before. `HashRing.get_node_pos` and `get_node` on a ring
  without nodes, and `get_hash_key` for an array with neither primary attributes nor an id, raise
  `HashRingError`.
- `deker_tools.log.set_logger(format_string, logger_name)` sets the format of one logger and its
  children; without a name it still sets every logger of the process, as before.
- `deker_tools.slices.slices_to_str` and `str_to_slices` are public: the server adapters use
  them instead of their own wrappers.
- An ellipsis in an index expression expands to all the missing dimensions.
- `py.typed`.

### Changed

- The length of a slice is computed without materialising its indices: one microsecond instead
  of 17 milliseconds for a million cells.
- The examples in the docstrings run as tests; the documentation links lead to docs.deker.io.

### Internals and tooling

- The build moved from poetry to uv, ruff and pydoclint replace black, flake8, isort and darglint;
  `make ci`, `make test-matrix`, `make install-matrix`; 100% test coverage required; tests in
  random order.

## [1.2.0] - 2026-11-11

### Added

- Python 3.13 and 3.14 are supported.
- NumPy 2 is supported (`numpy>=1.26,<3`): the slice helpers do not use the private
  `numpy.lib.index_tricks` anymore.
- `deker_tools.log`: the tree of the DEKER™ loggers and `set_logger` to set their format.

### Changed

- The dependency ranges are widened to the current majors; the dev and docs tooling is updated.
- CI runs the tests on every supported Python.

### Removed

- Python 3.9. Pin `deker-tools<1.2` to stay on it.

## [1.1.0] - 2023-12-28

### Changed

- All the configuration lives in `pyproject.toml`; the CI workflows moved to deker-actions; the
  README is rewritten; the license headers are added to the modules.

## [1.0.0] - 2023-08-01

The pilot release of deker-tools.

[2.0.0]: https://github.com/openweathermap/deker-tools/compare/v1.2.0...v2.0.0
[1.2.0]: https://github.com/openweathermap/deker-tools/compare/v1.1.0...v1.2.0
[1.1.0]: https://github.com/openweathermap/deker-tools/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/openweathermap/deker-tools/releases/tag/v1.0.0
