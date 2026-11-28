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

"""Consistent hash ring of a cluster and the keys of the arrays on it.

A cluster keeps every array on the node the ring gives for the key of the array. The placement is a contract:
the same nodes and keys shall give the same nodes in every version, or the arrays are looked for on wrong
nodes.
"""

import hashlib
import math

from bisect import bisect
from collections.abc import Collection
from datetime import datetime
from typing import Generic, Protocol, TypeVar

from deker_tools.time import get_utc
from deker_tools.typings import Sentinel, SentinelType, is_sentinel

__all__ = (
    "HashRing",
    "HashRingError",
    "HashedArray",
    "get_hash_by_primary_attrs",
    "get_hash_key",
    "get_id_and_primary_attributes",
)

NodeT = TypeVar("NodeT")

POINTS_PER_NODE = 40  # points of a node of the average weight on the ring
KEYS_PER_DIGEST = 3  # points taken from one md5 digest, four bytes each
BYTES_PER_KEY = 4


class HashRingError(Exception):
    """The ring cannot give a node: it has no nodes, or there is no key to look for."""


class HashedArray(Protocol):
    """An array as the ring sees it: its id and its primary attributes."""

    @property
    def id(self) -> str:
        """Id of the array."""

    @property
    def primary_attributes(self) -> dict:
        """Primary attributes of the array, empty if it has none."""


class HashRing(Generic[NodeT]):
    """Consistent hash ring over nodes of any type with a stable ``str()`` representation.

    Every node gets ``POINTS_PER_NODE`` points on the ring times its share of the weights; a key belongs to the
    node of the first point after the key, the ring wraps around after the last point.

    :param nodes: the nodes; each is placed by its ``str()``
    :param weights: weights of the nodes, a node not listed weighs 1; not given - all the nodes are equal
    """

    @staticmethod
    def _hash_digest(key: str) -> bytes:
        """Get the md5 digest of a key.

        :param key: the key
        :returns: 16 bytes
        """
        return hashlib.md5(key.encode()).digest()  # noqa: S324 - a placement hash, not a security one

    @staticmethod
    def _hash_val(digest: bytes, offset: int) -> int:
        """Build a 32-bit integer from four bytes of a digest, the last of them the highest.

        :param digest: the digest
        :param offset: index of the first of the four bytes
        :returns: the integer
        """
        return (digest[offset + 3] << 24) | (digest[offset + 2] << 16) | (digest[offset + 1] << 8) | digest[offset]

    def _generate_circle(self) -> None:
        """Place the nodes on the ring: the points of a node by its weight, three points per digest."""
        total_weight = sum(self.weights.get(node, 1) for node in self.nodes)

        for node in self.nodes:
            weight = self.weights.get(node, 1)
            factor = math.floor((POINTS_PER_NODE * len(self.nodes) * weight) / total_weight)

            for j in range(factor):
                digest = self._hash_digest(f"{node}-{j}")
                for i in range(KEYS_PER_DIGEST):
                    key = self._hash_val(digest, i * BYTES_PER_KEY)
                    self.ring[key] = node
                    self._sorted_keys.append(key)

        self._sorted_keys.sort()

    def __init__(self, nodes: Collection[NodeT], weights: dict[NodeT, int] | SentinelType = Sentinel) -> None:
        self.ring: dict[int, NodeT] = {}
        self._sorted_keys: list[int] = []

        self.nodes: Collection[NodeT] = nodes
        self.weights: dict[NodeT, int] = {} if is_sentinel(weights) else weights

        self._generate_circle()

    def gen_key(self, key: str) -> int:
        """Get the place of a key on the ring.

        :param key: the key
        :returns: the place, a 32-bit integer
        """
        return self._hash_val(self._hash_digest(key), 0)

    def get_node_pos(self, string_key: str) -> int:
        """Get the index of the point of the node responsible for a key.

        :param string_key: the key
        :returns: index of the point in the sorted points of the ring
        :raises HashRingError: if the ring has no nodes
        """
        if not self.ring:
            raise HashRingError(f"The ring has no nodes to place the key {string_key!r} on")

        pos = bisect(self._sorted_keys, self.gen_key(string_key))
        if pos == len(self._sorted_keys):
            return 0
        return pos

    def get_node(self, string_key: str) -> NodeT:
        """Get the node responsible for a key.

        :param string_key: the key
        :returns: the node
        :raises HashRingError: if the ring has no nodes
        """
        return self.ring[self._sorted_keys[self.get_node_pos(string_key)]]


def get_hash_by_primary_attrs(primary_attributes: dict) -> str:
    """Get the key of an array by its primary attributes.

    The values are joined with ``/`` in the order of the attributes; a ``v_position`` is joined with ``-``,
    a datetime is written in UTC.

    :param primary_attributes: primary attributes of the array
    :returns: the key, empty for no attributes
    """
    values = []
    for name, attribute in primary_attributes.items():
        if name == "v_position":
            value = "-".join(str(element) for element in attribute)
        elif isinstance(attribute, datetime):
            value = get_utc(attribute).isoformat()
        else:
            value = str(attribute)
        values.append(value)
    return "/".join(values)


def get_id_and_primary_attributes(array: dict | HashedArray) -> tuple[str | None, dict | None]:
    """Get the id and the primary attributes of an array or of its metadata.

    :param array: the array, or its metadata with the keys ``id_`` and ``primary_attributes``
    :returns: the id and the primary attributes; the metadata may lack either of them, then it is ``None``
    """
    if isinstance(array, dict):
        return array.get("id_"), array.get("primary_attributes")
    return array.id, array.primary_attributes


def get_hash_key(array: dict | HashedArray) -> str:
    """Get the key of an array on the ring: its primary attributes, or its id when it has none.

    :param array: the array, or its metadata with the keys ``id_`` and ``primary_attributes``
    :returns: the key
    :raises HashRingError: if the array has neither primary attributes nor an id
    """
    id_, primary_attributes = get_id_and_primary_attributes(array)
    if primary_attributes:
        return get_hash_by_primary_attrs(primary_attributes)
    if id_ is None:
        raise HashRingError("Array has neither primary attributes nor an id, nothing to hash")
    return id_
