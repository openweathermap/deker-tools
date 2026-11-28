import hashlib
import math

from bisect import bisect
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Generic, TypeVar

import pytest

from deker_tools.hash_ring import (
    HashRing,
    HashRingError,
    get_hash_by_primary_attrs,
    get_hash_key,
    get_id_and_primary_attributes,
)
from deker_tools.typings import Sentinel, SentinelType, is_sentinel

md5_constructor = hashlib.md5

NodeT = TypeVar("NodeT")

NODES = ["node-a", "node-b", "node-c"]

# The placement of the keys is the contract of the cluster: a change moves the arrays to other nodes.
PLAIN_PLACEMENT = {
    "key-0": "node-a",
    "key-1": "node-b",
    "key-2": "node-c",
    "key-3": "node-a",
    "key-4": "node-a",
    "key-5": "node-a",
    "key-6": "node-b",
    "key-7": "node-a",
    "key-8": "node-a",
    "key-9": "node-b",
    "key-10": "node-c",
    "key-11": "node-c",
}
WEIGHTED_PLACEMENT = {
    "key-0": "node-a",
    "key-1": "node-a",
    "key-2": "node-c",
    "key-3": "node-a",
    "key-4": "node-a",
    "key-5": "node-a",
    "key-6": "node-a",
    "key-7": "node-a",
    "key-8": "node-a",
    "key-9": "node-a",
    "key-10": "node-a",
    "key-11": "node-c",
}


@dataclass(frozen=True)
class Node:
    """A node of a cluster: the ring places it by its ``str()``."""

    name: str

    def __str__(self) -> str:
        return self.name


@dataclass
class Array:
    """An array as the ring sees it: its id and its primary attributes."""

    id: str
    primary_attributes: dict


class LegacyHashRing(Generic[NodeT]):
    """The ring of deker-server-adapters before the move, copied as it was: the reference for the ring of tools."""

    def _hash_val(self, b_key: list[int], entry_fn: Callable) -> int:
        """Build a 32-bit integer from four bytes of a digest.

        :param b_key: bytes of the digest
        :param entry_fn: maps the byte number 0-3 to its index in the digest
        :returns: the integer
        """
        return (b_key[entry_fn(3)] << 24) | (b_key[entry_fn(2)] << 16) | (b_key[entry_fn(1)] << 8) | b_key[entry_fn(0)]

    def _hash_digest(self, key: str) -> list[int]:
        """Get the md5 digest of a key as a list of bytes.

        :param key: the key
        :returns: 16 bytes
        """
        m = md5_constructor()
        m.update(key.encode())
        return list(m.digest())

    def _generate_circle(self) -> None:
        """Place the nodes on the ring: 40 points per node times its share of the weights, 3 keys per point."""
        total_weight = 0
        for node in self.nodes:
            total_weight += self.weights.get(node, 1)

        for node in self.nodes:
            weight = self.weights.get(node, 1)
            factor = math.floor((40 * len(self.nodes) * weight) / total_weight)

            for j in range(0, int(factor)):
                b_key = self._hash_digest(f"{node}-{j}")

                for i in range(0, 3):
                    key = self._hash_val(b_key, lambda x: x + i * 4)  # noqa: B023 - the legacy code as it was
                    self.ring[key] = node
                    self._sorted_keys.append(key)

        self._sorted_keys.sort()

    def __init__(
        self,
        nodes: list[NodeT] | tuple[NodeT, ...] | set[NodeT],
        weights: dict[NodeT, int] | SentinelType = Sentinel,
    ):
        """Generate instance of hash ring with given nodes.

        :param nodes: is a list of objects that have a proper __str__ representation.
        :param weights: is dictionary that sets weights to the nodes.  The default
        weight is that all nodes are equal.
        """
        self.ring: dict[int, NodeT] = {}
        self._sorted_keys: list[int] = []

        self.nodes: list[NodeT] | tuple[NodeT, ...] | set[NodeT] = nodes
        self.weights: dict[NodeT, int] = {} if is_sentinel(weights) else weights

        self._generate_circle()

    def get_node(self, string_key: str) -> NodeT:
        """Return the node responsible for a key.

        :param string_key: String key
        :returns: node responsible for the key
        :raises HashRingError: if the ring is empty
        """
        pos = self.get_node_pos(string_key)
        if pos is None:
            msg = f"Couldn't find a position in {self.ring}"
            raise HashRingError(msg)
        return self.ring[self._sorted_keys[pos]]

    def get_node_pos(self, string_key: str) -> int | None:
        """Return the position in the ring of the node responsible for a key.

        :param string_key: String key
        :returns: position of the node in the ring, or ``None`` if the ring is empty
        """
        if not self.ring:
            return None

        key = self.gen_key(string_key)

        nodes = self._sorted_keys
        pos = bisect(nodes, key)

        if pos == len(nodes):
            return 0
        return pos

    def gen_key(self, key: str) -> int:
        """Given a string key it returns a long value.

        This long value represents a place on the hash ring.
        md5 is currently used because it mixes well.

        :param key: a string key
        :returns: integer position of the key on the hash ring
        """
        b_key = self._hash_digest(key)
        return self._hash_val(b_key, lambda x: x)


KEYS = [f"key-{i}" for i in range(2000)] + ["", "array-id", "v/1-2-3", "2024-01-01T00:00:00+00:00/x"]


class TestHashRingAgainstTheLegacyRing:
    """The ring of deker-tools gives exactly what the legacy ring gave: the points, the positions, the nodes."""

    @pytest.mark.parametrize(
        ("nodes", "weights"),
        [
            pytest.param(["node-a"], {}, id="one-node"),
            pytest.param(NODES, {}, id="equal"),
            pytest.param(NODES, {"node-a": 3}, id="weighted"),
            pytest.param(NODES, {"node-a": 5, "node-b": 2, "node-c": 1}, id="all-weighted"),
            pytest.param(NODES, {"node-c": 0}, id="zero-weight"),
            pytest.param([f"http://10.0.0.{i}:8000" for i in range(10)], {}, id="ten-urls"),
            pytest.param([Node(name) for name in NODES], {}, id="objects"),
            pytest.param(set(NODES), {}, id="set"),
        ],
    )
    def test_same_ring(self, nodes, weights):
        legacy = LegacyHashRing(nodes, weights)
        ring = HashRing(nodes, weights=weights)
        assert ring.ring == legacy.ring
        assert ring._sorted_keys == legacy._sorted_keys
        assert [ring.gen_key(key) for key in KEYS] == [legacy.gen_key(key) for key in KEYS]
        assert [ring.get_node(key) for key in KEYS] == [legacy.get_node(key) for key in KEYS]

    @pytest.mark.parametrize("offset", [0, 4, 8, 12])
    def test_same_integer_from_the_bytes_of_a_digest(self, offset):
        legacy = LegacyHashRing([])
        for key in KEYS:
            digest = hashlib.md5(key.encode()).digest()  # noqa: S324 - the placement hash
            assert HashRing._hash_val(digest, offset) == legacy._hash_val(list(digest), lambda x: x + offset)


class TestHashRingPlacement:
    """The ring places the keys exactly as the earlier versions did."""

    @pytest.mark.parametrize(("key", "node"), list(PLAIN_PLACEMENT.items()))
    def test_nodes_of_equal_weights(self, key, node):
        assert HashRing(NODES).get_node(key) == node

    @pytest.mark.parametrize(("key", "node"), list(WEIGHTED_PLACEMENT.items()))
    def test_nodes_of_given_weights(self, key, node):
        assert HashRing(NODES, weights={"node-a": 3}).get_node(key) == node

    @pytest.mark.parametrize(
        ("key", "position"),
        [
            pytest.param("", 3649838548, id="empty"),
            pytest.param("key", 2316004924, id="word"),
            pytest.param("array-id", 114793548, id="id"),
            pytest.param("v/1-2-3", 1169771501, id="primary-attributes"),
        ],
    )
    def test_position_of_a_key(self, key, position):
        assert HashRing(NODES).gen_key(key) == position

    @pytest.mark.parametrize(
        ("weights", "first_points"),
        [
            pytest.param({}, [2992783, 51157997, 63710660], id="equal"),
            pytest.param({"node-a": 3}, [2992783, 51157997, 72638653], id="weighted"),
        ],
    )
    def test_points_of_the_nodes(self, weights, first_points):
        ring = HashRing(NODES, weights=weights)
        assert len(ring.ring) == 360
        assert ring._sorted_keys[:3] == first_points

    def test_nodes_are_placed_by_their_string(self):
        ring = HashRing([Node(name) for name in NODES])
        assert {key: str(ring.get_node(key)) for key in PLAIN_PLACEMENT} == PLAIN_PLACEMENT


class TestHashRing:
    """The ring over its nodes: points, weights, the wrap around and the empty ring."""

    def test_empty_ring_has_no_node(self):
        ring: HashRing[str] = HashRing([])
        with pytest.raises(HashRingError, match="no nodes"):
            ring.get_node_pos("key")
        with pytest.raises(HashRingError, match="no nodes"):
            ring.get_node("key")

    def test_keys_past_the_last_point_wrap_to_the_first_node(self):
        ring = HashRing(["a", "b"])
        last_point = ring._sorted_keys[-1]
        key = next(f"key-{i}" for i in range(1_000_000) if ring.gen_key(f"key-{i}") > last_point)
        assert ring.get_node_pos(key) == 0
        assert ring.get_node(key) == ring.ring[ring._sorted_keys[0]]

    def test_keys_before_the_last_point_take_the_next_point(self):
        ring = HashRing(["a", "b"])
        first, second = ring._sorted_keys[:2]
        keys = [f"key-{i}" for i in range(100_000)]
        before_first = next(key for key in keys if ring.gen_key(key) < first)
        between = next(key for key in keys if first < ring.gen_key(key) < second)
        assert ring.get_node_pos(before_first) == 0
        assert ring.get_node_pos(between) == 1

    def test_weights_change_the_share_of_points(self):
        ring = HashRing(["a", "b"], weights={"a": 3, "b": 1})
        points = Counter(ring.ring.values())
        assert points["a"] > points["b"]
        assert ring.weights == {"a": 3, "b": 1}

    def test_nodes_without_weights_are_equal(self):
        ring = HashRing(["a", "b"])
        points = Counter(ring.ring.values())
        assert points["a"] == points["b"]
        assert ring.weights == {}

    @pytest.mark.parametrize("nodes", [["a", "b"], ("a", "b"), {"a", "b"}], ids=["list", "tuple", "set"])
    def test_nodes_of_any_collection(self, nodes):
        assert HashRing(nodes).nodes == nodes


class TestHashKey:
    """The key of an array on the ring: its primary attributes, or its id when it has none."""

    @pytest.mark.parametrize(
        ("primary_attributes", "key"),
        [
            pytest.param({"vid": "v", "v_position": (1, 2, 3)}, "v/1-2-3", id="v-position"),
            pytest.param({"name": "x", "n": 5}, "x/5", id="values"),
            pytest.param(
                {"dt": datetime(2024, 1, 1, 3, tzinfo=timezone(timedelta(hours=3))), "name": "x"},
                "2024-01-01T00:00:00+00:00/x",
                id="datetime-in-utc",
            ),
            pytest.param({}, "", id="empty"),
        ],
    )
    def test_key_by_primary_attributes(self, primary_attributes, key):
        assert get_hash_by_primary_attrs(primary_attributes) == key

    @pytest.mark.parametrize(
        ("array", "key"),
        [
            pytest.param({"primary_attributes": {"name": "x"}, "id_": "abc"}, "x", id="dict-attributes"),
            pytest.param({"primary_attributes": {}, "id_": "abc"}, "abc", id="dict-id"),
            pytest.param({"id_": "abc"}, "abc", id="dict-without-attributes"),
            pytest.param(Array("abc", {"name": "x"}), "x", id="array-attributes"),
            pytest.param(Array("abc", {}), "abc", id="array-id"),
        ],
    )
    def test_key_of_an_array(self, array, key):
        assert get_hash_key(array) == key

    @pytest.mark.parametrize(
        "array",
        [
            pytest.param({"primary_attributes": {}, "id_": None}, id="none"),
            pytest.param({}, id="nothing"),
        ],
    )
    def test_key_requires_an_id_or_primary_attributes(self, array):
        with pytest.raises(HashRingError, match="nothing to hash"):
            get_hash_key(array)

    @pytest.mark.parametrize(
        ("array", "expected"),
        [
            pytest.param({"primary_attributes": {"a": 1}, "id_": "abc"}, ("abc", {"a": 1}), id="dict"),
            pytest.param({}, (None, None), id="empty-dict"),
            pytest.param(Array("abc", {"a": 1}), ("abc", {"a": 1}), id="array"),
        ],
    )
    def test_id_and_primary_attributes(self, array, expected):
        assert get_id_and_primary_attributes(array) == expected
