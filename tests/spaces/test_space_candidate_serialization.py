"""Tests for structured-space candidate serialization codecs."""

from collections.abc import Mapping

from variopt import CategoricalSpace, IntegerSpace, RecordSpace, TupleSpace
from variopt.spaces import RecordCandidate
from variopt.spaces.serialization import (
    space_candidate_from_dict,
    space_candidate_to_dict,
)


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
