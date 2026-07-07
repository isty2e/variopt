"""Tests for structured-space candidate serialization codecs."""

from collections.abc import Mapping
from typing import cast

import numpy as np
import pytest

from variopt import CategoricalSpace, IntegerSpace, RecordSpace, TupleSpace
from variopt.json_types import JSONValue
from variopt.spaces import RecordCandidate
from variopt.spaces.serialization import (
    space_candidate_from_dict,
    space_candidate_to_dict,
)
from variopt.spaces.types import SpaceCandidateValue

_EXCESSIVE_CODEC_DEPTH = 300


def runtime_value(value: object) -> object:
    """Return a value through an object-typed boundary for runtime guard tests."""
    return value


def too_deep_tuple_candidate() -> SpaceCandidateValue:
    """Return a tuple candidate deeper than the codec policy allows."""
    candidate: SpaceCandidateValue = 0
    for _ in range(_EXCESSIVE_CODEC_DEPTH):
        candidate = (candidate,)
    return candidate


def too_deep_list_payload() -> JSONValue:
    """Return a JSON payload deeper than the codec policy allows."""
    payload: JSONValue = 0
    for _ in range(_EXCESSIVE_CODEC_DEPTH):
        payload = [payload]
    return payload


class SpaceCandidateSerializationTests:
    """Regression tests for canonical structured-candidate JSON codecs."""

    def test_record_payload_normalizes_back_to_canonical_value(self) -> None:
        space = RecordSpace(depth=IntegerSpace(0, 5), mode=CategoricalSpace(("a", "b")))
        candidate = space.normalize({"depth": 3, "mode": "b"})

        payload = space_candidate_to_dict(candidate)
        decoded = space_candidate_from_dict(payload)
        assert isinstance(decoded, Mapping)
        restored = space_candidate_from_dict(payload, record_candidates=True)

        assert isinstance(restored, RecordCandidate)
        assert restored == candidate
        space.validate(restored)

    def test_nested_record_payloads_normalize_through_tuples(self) -> None:
        item_space = RecordSpace(depth=IntegerSpace(0, 5))
        space = TupleSpace(
            item_space,
            TupleSpace(item_space, item_space),
        )
        candidate = space.normalize(
            [
                {"depth": 3},
                ({"depth": 1}, {"depth": 4}),
            ],
        )

        payload = space_candidate_to_dict(candidate)
        restored = space_candidate_from_dict(payload, record_candidates=True)

        assert isinstance(restored, tuple)
        assert isinstance(restored[0], RecordCandidate)
        assert isinstance(restored[1], tuple)
        assert all(isinstance(item, RecordCandidate) for item in restored[1])
        assert restored == candidate
        space.validate(restored)

    def test_record_payload_preserves_bytes_and_bytearray_values(self) -> None:
        candidate = RecordCandidate(
            entries=(
                ("raw_bytes", b"\x00\xff"),
                ("raw_bytearray", bytearray(b"\x01\x02")),
            ),
        )

        payload = space_candidate_to_dict(candidate)
        restored = space_candidate_from_dict(payload, record_candidates=True)

        assert restored == candidate

    def test_marker_shaped_mapping_payload_round_trips_as_mapping(self) -> None:
        candidate = {"__variopt_bytes__": "abcd"}

        payload = space_candidate_to_dict(candidate)
        restored = space_candidate_from_dict(payload)

        assert restored == candidate

    def test_nested_marker_shaped_mapping_payload_round_trips(self) -> None:
        candidate = {
            "outer": {
                "__variopt_bytearray__": "0102",
            },
        }

        payload = space_candidate_to_dict(candidate)
        restored = space_candidate_from_dict(payload)

        assert restored == candidate

    def test_escape_marker_shaped_mapping_payload_round_trips(self) -> None:
        candidate = {"__variopt_mapping__": "kept-as-user-data"}

        payload = space_candidate_to_dict(candidate)
        restored = space_candidate_from_dict(payload)

        assert restored == candidate

    def test_legacy_escape_marker_mapping_without_format_stays_plain_mapping(self) -> None:
        restored = space_candidate_from_dict(
            {
                "__variopt_mapping__": {
                    "items": [["depth", 3]],
                },
            },
        )

        assert restored == {"__variopt_mapping__": {"items": (("depth", 3),)}}

    def test_malformed_escaped_mapping_payload_raises_type_error(self) -> None:
        payload: JSONValue = {
            "__variopt_mapping__": {
                "format": "variopt.mapping",
                "items": [["depth"]],
            },
        }

        with pytest.raises(TypeError, match="two-item arrays"):
            _ = space_candidate_from_dict(payload)

    def test_escaped_mapping_payload_rejects_duplicate_keys(self) -> None:
        payload: JSONValue = {
            "__variopt_mapping__": {
                "format": "variopt.mapping",
                "items": [["depth", 1], ["depth", 2]],
            },
        }

        with pytest.raises(ValueError, match="keys must be unique"):
            _ = space_candidate_from_dict(payload)

    def test_legacy_bytes_marker_payload_still_decodes_as_bytes(self) -> None:
        restored = space_candidate_from_dict({"__variopt_bytes__": "00ff"})

        assert restored == b"\x00\xff"

    def test_legacy_bytearray_marker_payload_still_decodes_as_bytearray(self) -> None:
        restored = space_candidate_from_dict({"__variopt_bytearray__": "0102"})

        assert restored == bytearray(b"\x01\x02")

    def test_to_dict_rejects_unsupported_list_candidate(self) -> None:
        raw_candidate = runtime_value([1, 2])
        unsupported_candidate = cast(SpaceCandidateValue, raw_candidate)

        with pytest.raises(TypeError, match="supported structured candidate"):
            _ = space_candidate_to_dict(unsupported_candidate)

    def test_to_dict_rejects_null_candidate(self) -> None:
        raw_candidate = runtime_value(None)
        unsupported_candidate = cast(SpaceCandidateValue, raw_candidate)

        with pytest.raises(TypeError, match="supported structured candidate"):
            _ = space_candidate_to_dict(unsupported_candidate)

    def test_to_dict_rejects_non_finite_float_candidate(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            _ = space_candidate_to_dict(float("inf"))

    def test_from_dict_rejects_non_finite_float_payload(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            _ = space_candidate_from_dict(float("nan"))

        with pytest.raises(ValueError, match="finite"):
            _ = space_candidate_from_dict(float("inf"))

    def test_to_dict_rejects_numpy_float_candidate(self) -> None:
        raw_candidate = runtime_value(np.float64(1.0))
        unsupported_candidate = cast(SpaceCandidateValue, raw_candidate)

        with pytest.raises(TypeError, match="supported structured candidate"):
            _ = space_candidate_to_dict(unsupported_candidate)

    def test_to_dict_rejects_bytes_subclass_candidate(self) -> None:
        class DerivedBytes(bytes):
            """Bytes subclass used to verify exact scalar canonicality."""

        raw_candidate = runtime_value(DerivedBytes(b"abc"))
        unsupported_candidate = cast(SpaceCandidateValue, raw_candidate)

        with pytest.raises(TypeError, match="supported structured candidate"):
            _ = space_candidate_to_dict(unsupported_candidate)

    def test_from_dict_rejects_numpy_float_payload(self) -> None:
        raw_payload = runtime_value(np.float64(1.0))
        unsupported_payload = cast(JSONValue, raw_payload)

        with pytest.raises(TypeError, match="supported structured candidate payload"):
            _ = space_candidate_from_dict(unsupported_payload)

    def test_to_dict_allows_shared_noncyclic_mapping_candidate(self) -> None:
        shared_child: dict[str, SpaceCandidateValue] = {"value": 1}
        candidate: dict[str, SpaceCandidateValue] = {
            "left": shared_child,
            "right": shared_child,
        }

        payload = space_candidate_to_dict(candidate)

        assert payload == {
            "left": {"value": 1},
            "right": {"value": 1},
        }

    def test_from_dict_allows_shared_noncyclic_list_payload(self) -> None:
        shared_child: list[JSONValue] = [1]
        payload: JSONValue = [shared_child, shared_child]

        restored = space_candidate_from_dict(payload)

        assert restored == ((1,), (1,))

    def test_to_dict_rejects_cyclic_mapping_candidate(self) -> None:
        raw_candidate: dict[str, object] = {}
        raw_candidate["self"] = raw_candidate
        candidate = cast(SpaceCandidateValue, raw_candidate)

        with pytest.raises(ValueError, match="cycle"):
            _ = space_candidate_to_dict(candidate)

    def test_from_dict_rejects_cyclic_mapping_payload(self) -> None:
        raw_payload: dict[str, object] = {}
        raw_payload["self"] = raw_payload
        payload = cast(JSONValue, raw_payload)

        with pytest.raises(ValueError, match="cycle"):
            _ = space_candidate_from_dict(payload)

    def test_from_dict_rejects_escaped_mapping_item_cycle(self) -> None:
        raw_items: list[object] = []
        raw_items.append(["self", raw_items])
        payload = cast(
            JSONValue,
            {
                "__variopt_mapping__": {
                    "format": "variopt.mapping",
                    "items": raw_items,
                },
            },
        )

        with pytest.raises(ValueError, match="cycle"):
            _ = space_candidate_from_dict(payload)

    def test_to_dict_rejects_excessively_deep_candidate(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            _ = space_candidate_to_dict(too_deep_tuple_candidate())

    def test_from_dict_rejects_excessively_deep_payload(self) -> None:
        with pytest.raises(ValueError, match="depth"):
            _ = space_candidate_from_dict(too_deep_list_payload())
