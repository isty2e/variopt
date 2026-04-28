"""Tests for shared continuous optimizer codecs."""

from collections.abc import Mapping
from dataclasses import dataclass
from math import exp, isclose, log

import numpy as np
import pytest
from typing_extensions import override

from tests.numeric_support import approx_equal
from variopt.spaces import (
    IntegerSpace,
    LeafPath,
    RealSpace,
    RecordCandidate,
    RecordSpace,
    SpaceBoundaryValue,
    StructuredLeafSpace,
    StructuredSearchSpace,
)
from variopt.spaces.projections import ContinuousStructuredSpaceCodec
from variopt.spaces.types import SpaceCandidateValue

ConditionalRealCandidate = tuple[float, float]


@dataclass(frozen=True)
class ConditionalRealPairSpace(
    StructuredSearchSpace[ConditionalRealCandidate, ConditionalRealCandidate],
):
    """Test-only real structured space with candidate-conditioned topology."""

    head_space: RealSpace
    tail_space: RealSpace

    @override
    def normalize(
        self,
        raw_candidate: ConditionalRealCandidate,
    ) -> ConditionalRealCandidate:
        return (
            self.head_space.normalize(raw_candidate[0]),
            self.tail_space.normalize(raw_candidate[1]),
        )

    @override
    def validate(self, candidate: ConditionalRealCandidate) -> None:
        self.head_space.validate(candidate[0])
        self.tail_space.validate(candidate[1])

    @override
    def sample(self, random_state: np.random.RandomState) -> ConditionalRealCandidate:
        return (
            self.head_space.sample(random_state),
            self.tail_space.sample(random_state),
        )

    @override
    def leaf_paths(self) -> tuple[LeafPath, ...]:
        return (("head",), ("tail",))

    @override
    def has_static_topology(self) -> bool:
        return False

    @override
    def active_leaf_paths(
        self,
        candidate: ConditionalRealCandidate,
    ) -> tuple[LeafPath, ...]:
        self.validate(candidate)
        if candidate[0] > 0.0:
            return (("head",), ("tail",))
        return (("head",),)

    @override
    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        if path == ("head",):
            return self.head_space
        if path == ("tail",):
            return self.tail_space
        msg = f"invalid conditional real pair path: {path!r}"
        raise TypeError(msg)

    @override
    def leaf_value_at_path(
        self,
        candidate: ConditionalRealCandidate,
        path: LeafPath,
    ) -> SpaceCandidateValue:
        self.validate(candidate)
        if path == ("head",):
            return candidate[0]
        if path == ("tail",):
            return candidate[1]
        msg = f"invalid conditional real pair path: {path!r}"
        raise TypeError(msg)

    @override
    def replace_leaf_values(
        self,
        candidate: ConditionalRealCandidate,
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> ConditionalRealCandidate:
        self.validate(candidate)
        head_value = candidate[0]
        tail_value = candidate[1]
        if ("head",) in replacements:
            replacement = replacements[("head",)]
            if type(replacement) is not float:
                msg = "conditional real head replacement must be a canonical float"
                raise TypeError(msg)
            head_value = self.head_space.normalize(replacement)
        if ("tail",) in replacements:
            replacement = replacements[("tail",)]
            if type(replacement) is not float:
                msg = "conditional real tail replacement must be a canonical float"
                raise TypeError(msg)
            tail_value = self.tail_space.normalize(replacement)
        return (head_value, tail_value)


class ContinuousStructuredSpaceCodecTests:
    """Regression tests for continuous structured-space codecs."""

    def test_round_trips_log_scaled_record_coordinates(self) -> None:
        space = RecordSpace(
            x=RealSpace(1e-4, 10.0, scale="log"),
            y=RealSpace(-5.0, 5.0),
        )
        codec: ContinuousStructuredSpaceCodec[
            Mapping[str, SpaceBoundaryValue] | RecordCandidate,
            RecordCandidate,
        ] = ContinuousStructuredSpaceCodec[
            Mapping[str, SpaceBoundaryValue] | RecordCandidate,
            RecordCandidate,
        ].from_space(space)
        candidate: RecordCandidate = space.normalize({"x": 0.1, "y": -3.0})

        coordinates = codec.coordinates_from_candidate(candidate)
        projected_candidate: RecordCandidate = codec.candidate_from_coordinates(
            candidate,
            (-0.5, 2.0),
        )

        assert codec.leaf_paths == (("x",), ("y",))
        assert isclose(coordinates[0], log(0.1))
        assert isclose(coordinates[1], -3.0)
        assert approx_equal(record_real(projected_candidate, "x"), exp(-0.5))
        assert approx_equal(record_real(projected_candidate, "y"), 2.0)

    def test_rejects_non_real_structured_space(self) -> None:
        space = RecordSpace(
            x=RealSpace(-1.0, 1.0),
            step=IntegerSpace(0, 10),
        )

        with pytest.raises(TypeError, match="RealSpace"):
            _ = ContinuousStructuredSpaceCodec[
                Mapping[str, SpaceBoundaryValue] | RecordCandidate,
                RecordCandidate,
            ].from_space(space)

    def test_rejects_dynamic_topology_space(self) -> None:
        space = ConditionalRealPairSpace(
            head_space=RealSpace(-1.0, 1.0),
            tail_space=RealSpace(-1.0, 1.0),
        )

        with pytest.raises(TypeError, match="static topology"):
            _ = ContinuousStructuredSpaceCodec[
                ConditionalRealCandidate,
                ConditionalRealCandidate,
            ].from_space(space)


def record_real(candidate: RecordCandidate, field_name: str) -> float:
    """Return one canonical real-valued record field."""
    value = candidate[field_name]
    if type(value) is not float:
        msg = f"record field {field_name!r} must be a canonical float"
        raise TypeError(msg)
    return value
