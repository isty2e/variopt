"""Tests for structured space-derived diversity metrics."""

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pytest
from typing_extensions import override

from tests.numeric_support import approx_equal
from variopt import (
    ArraySpace,
    CategoricalSpace,
    IntegerSpace,
    PermutationSpace,
    RealSpace,
    RecordSpace,
    TupleSpace,
)
from variopt.diversity import StructuredSpaceDiversityMetric
from variopt.diversity.space_metric import (
    structured_distance_between_validated_candidates,
)
from variopt.spaces import (
    LeafPath,
    RecordCandidate,
    StructuredLeafSpace,
    StructuredSearchSpace,
)
from variopt.spaces.geometry import (
    CompiledStructuredGeometryProvider,
    StructuredDistanceParts,
    StructuredSpaceGeometry,
    compile_structured_geometry,
    distance_parts,
    generic_distance_parts,
)
from variopt.spaces.types import SpaceCandidateValue, SpaceScalarValue

WrappedPairCandidate = tuple[int, str]
ConditionalBranchCandidate = tuple[str, int]


@dataclass(frozen=True)
class WrappedPairSpace(
    StructuredSearchSpace[WrappedPairCandidate, WrappedPairCandidate],
):
    """Minimal custom structured space that should use the generic geometry path."""

    depth_space: IntegerSpace
    mode_space: CategoricalSpace[str]

    @override
    def normalize(self, raw_candidate: WrappedPairCandidate) -> WrappedPairCandidate:
        return (
            self.depth_space.normalize(raw_candidate[0]),
            self.mode_space.normalize(raw_candidate[1]),
        )

    @override
    def validate(self, candidate: WrappedPairCandidate) -> None:
        self.depth_space.validate(candidate[0])
        self.mode_space.validate(candidate[1])

    @override
    def sample(self, random_state: np.random.RandomState) -> WrappedPairCandidate:
        return (
            self.depth_space.sample(random_state),
            self.mode_space.sample(random_state),
        )

    @override
    def leaf_paths(self) -> tuple[LeafPath, ...]:
        return ((0,), (1,))

    @override
    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        if path == (0,):
            return self.depth_space
        if path == (1,):
            return self.mode_space
        msg = f"invalid wrapped pair path: {path!r}"
        raise TypeError(msg)

    @override
    def leaf_value_at_path(
        self,
        candidate: WrappedPairCandidate,
        path: LeafPath,
    ) -> SpaceCandidateValue:
        self.validate(candidate)
        if path == (0,):
            return candidate[0]
        if path == (1,):
            return candidate[1]
        msg = f"invalid wrapped pair path: {path!r}"
        raise TypeError(msg)

    @override
    def replace_leaf_values(
        self,
        candidate: WrappedPairCandidate,
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> WrappedPairCandidate:
        self.validate(candidate)
        depth_value = candidate[0]
        mode_value = candidate[1]

        if (0,) in replacements:
            replacement = replacements[(0,)]
            if type(replacement) is not int:
                msg = "wrapped pair depth replacement must be a canonical integer"
                raise TypeError(msg)
            depth_value = self.depth_space.normalize(replacement)

        if (1,) in replacements:
            replacement = replacements[(1,)]
            if not isinstance(replacement, str):
                msg = "wrapped pair mode replacement must be a canonical string"
                raise TypeError(msg)
            mode_value = self.mode_space.normalize(replacement)

        return (depth_value, mode_value)


@dataclass(frozen=True)
class ConditionalBranchSpace(
    StructuredSearchSpace[ConditionalBranchCandidate, ConditionalBranchCandidate],
):
    """Minimal conditional structured space for active-topology regression."""

    mode_space: CategoricalSpace[str]
    depth_space: IntegerSpace

    @override
    def normalize(
        self,
        raw_candidate: ConditionalBranchCandidate,
    ) -> ConditionalBranchCandidate:
        return (
            self.mode_space.normalize(raw_candidate[0]),
            self.depth_space.normalize(raw_candidate[1]),
        )

    @override
    def validate(self, candidate: ConditionalBranchCandidate) -> None:
        self.mode_space.validate(candidate[0])
        self.depth_space.validate(candidate[1])

    @override
    def sample(self, random_state: np.random.RandomState) -> ConditionalBranchCandidate:
        return (
            self.mode_space.sample(random_state),
            self.depth_space.sample(random_state),
        )

    @override
    def leaf_paths(self) -> tuple[LeafPath, ...]:
        return (("mode",), ("depth",))

    @override
    def has_static_topology(self) -> bool:
        return False

    @override
    def active_leaf_paths(
        self,
        candidate: ConditionalBranchCandidate,
    ) -> tuple[LeafPath, ...]:
        self.validate(candidate)
        if candidate[0] == "tree":
            return (("mode",), ("depth",))
        return (("mode",),)

    @override
    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        if path == ("mode",):
            return self.mode_space
        if path == ("depth",):
            return self.depth_space
        msg = f"invalid conditional branch path: {path!r}"
        raise TypeError(msg)

    @override
    def leaf_value_at_path(
        self,
        candidate: ConditionalBranchCandidate,
        path: LeafPath,
    ) -> SpaceCandidateValue:
        self.validate(candidate)
        if path == ("mode",):
            return candidate[0]
        if path == ("depth",):
            return candidate[1]
        msg = f"invalid conditional branch path: {path!r}"
        raise TypeError(msg)

    @override
    def replace_leaf_values(
        self,
        candidate: ConditionalBranchCandidate,
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> ConditionalBranchCandidate:
        self.validate(candidate)
        mode_value = candidate[0]
        depth_value = candidate[1]

        if ("mode",) in replacements:
            replacement = replacements[("mode",)]
            if not isinstance(replacement, str):
                msg = "conditional mode replacement must be a canonical string"
                raise TypeError(msg)
            mode_value = self.mode_space.normalize(replacement)

        if ("depth",) in replacements:
            replacement = replacements[("depth",)]
            if type(replacement) is not int:
                msg = "conditional depth replacement must be a canonical integer"
                raise TypeError(msg)
            depth_value = self.depth_space.normalize(replacement)

        return (mode_value, depth_value)


@dataclass(frozen=True)
class WrappedPairCompiledGeometry(StructuredSpaceGeometry):
    """Custom compiled geometry used to verify provider opt-in behavior."""

    parts: StructuredDistanceParts

    @override
    def distance_parts(
        self,
        left: SpaceCandidateValue,
        right: SpaceCandidateValue,
    ) -> StructuredDistanceParts:
        return self.parts


@dataclass(frozen=True)
class ProviderWrappedPairSpace(
    WrappedPairSpace,
    CompiledStructuredGeometryProvider,
):
    """Wrapped pair space with an explicit sidecar compiled-geometry provider."""

    compiled_parts: StructuredDistanceParts

    @override
    def compile_structured_geometry(self) -> StructuredSpaceGeometry | None:
        """Return one custom compiled geometry for this space."""
        return WrappedPairCompiledGeometry(parts=self.compiled_parts)


class StructuredSpaceDiversityMetricTests:
    """Regression tests for generic space-derived diversity metrics."""

    def test_real_space_distance_is_linearly_normalized(self) -> None:
        metric = StructuredSpaceDiversityMetric(space=RealSpace(0.0, 10.0))

        distance = metric.distance(2.0, 7.0)

        assert distance == 0.5

    def test_real_space_distance_rejects_noncanonical_integer_leaf(self) -> None:
        metric = StructuredSpaceDiversityMetric(space=RealSpace(0.0, 10.0))

        with pytest.raises(TypeError):
            _ = metric.distance(2, 7.0)

    def test_real_space_distance_respects_log_scale(self) -> None:
        metric = StructuredSpaceDiversityMetric(
            space=RealSpace(1.0, 100.0, scale="log")
        )

        distance = metric.distance(1.0, 10.0)

        assert approx_equal(distance, 0.5)

    def test_categorical_space_distance_rejects_equal_noncanonical_leaf(self) -> None:
        space: CategoricalSpace[SpaceScalarValue] = CategoricalSpace((0, 1))
        metric = StructuredSpaceDiversityMetric(space=space)

        with pytest.raises(TypeError):
            _ = metric.distance(True, 1)

    def test_categorical_space_distance_rejects_unknown_choice(self) -> None:
        space: CategoricalSpace[str] = CategoricalSpace(("a", "b"))
        metric = StructuredSpaceDiversityMetric(space=space)

        with pytest.raises(ValueError, match="not in the declared choices"):
            _ = metric.distance("a", "c")

    def test_composite_space_distance_combines_leaf_distances_by_rms(self) -> None:
        space = RecordSpace(
            depth=IntegerSpace(1, 5),
            mode=CategoricalSpace(("a", "b")),
            rate=RealSpace(1.0, 100.0, scale="log"),
        )
        metric = StructuredSpaceDiversityMetric(space=space)
        left = space.normalize(
            {
                "depth": 1,
                "mode": "a",
                "rate": 1.0,
            },
        )
        right = space.normalize(
            {
                "depth": 3,
                "mode": "b",
                "rate": 10.0,
            },
        )

        distance = metric.distance(left, right)

        expected = math.sqrt((0.5 * 0.5 + 1.0 + 0.5 * 0.5) / 3.0)
        assert approx_equal(distance, expected)

    def test_record_space_distance_rejects_misordered_candidate_fields(self) -> None:
        space = RecordSpace(
            depth=IntegerSpace(1, 5),
            mode=CategoricalSpace(("a", "b")),
        )
        metric = StructuredSpaceDiversityMetric(space=space)
        left = RecordCandidate(entries=(("mode", "a"), ("depth", 1)))
        right = space.normalize({"depth": 3, "mode": "b"})

        with pytest.raises(ValueError, match="keys must exactly match"):
            _ = metric.distance(left, right)

    def test_tuple_space_distance_rejects_equal_noncanonical_leaf(self) -> None:
        space = TupleSpace(CategoricalSpace((0, 1)), RealSpace(0.0, 1.0))
        metric = StructuredSpaceDiversityMetric(space=space)

        with pytest.raises(TypeError):
            _ = metric.distance((True, 0.0), (1, 1.0))

    def test_array_space_distance_combines_element_leaf_distances_by_rms(self) -> None:
        space = ArraySpace(RealSpace(0.0, 10.0), length=3)
        metric = StructuredSpaceDiversityMetric(space=space)

        distance = metric.distance(
            space.normalize((0.0, 2.0, 10.0)),
            space.normalize((5.0, 7.0, 10.0)),
        )

        expected = math.sqrt((0.5 * 0.5 + 0.5 * 0.5 + 0.0) / 3.0)
        assert approx_equal(distance, expected)

    def test_categorical_array_distance_uses_mismatch_fraction(self) -> None:
        space = ArraySpace(CategoricalSpace(("a", "b")), length=4)
        metric = StructuredSpaceDiversityMetric(space=space)

        distance = metric.distance(
            space.normalize(("a", "a", "b", "b")),
            space.normalize(("a", "b", "a", "b")),
        )

        assert approx_equal(distance, math.sqrt(0.5))

    def test_binary_array_distance_uses_position_mismatch_fraction(self) -> None:
        space = ArraySpace(IntegerSpace(0, 1), length=4)
        metric = StructuredSpaceDiversityMetric(space=space)

        distance = metric.distance(
            space.normalize((0, 1, 1, 0)),
            space.normalize((0, 0, 1, 1)),
        )

        assert approx_equal(distance, math.sqrt(0.5))

    def test_integer_array_distance_respects_log_scale(self) -> None:
        space = ArraySpace(IntegerSpace(1, 100, scale="log"), length=2)
        metric = StructuredSpaceDiversityMetric(space=space)

        distance = metric.distance(
            space.normalize((1, 10)),
            space.normalize((10, 100)),
        )

        assert approx_equal(distance, 0.5)

    def test_permutation_space_distance_uses_position_mismatch_fraction(self) -> None:
        space = PermutationSpace(size=4)
        metric = StructuredSpaceDiversityMetric(space=space)

        distance = metric.distance(
            space.normalize((0, 1, 2, 3)),
            space.normalize((0, 2, 1, 3)),
        )

        assert approx_equal(distance, math.sqrt(0.5))

    def test_permutation_space_distance_handles_diagonal_full_and_size_one(
        self,
    ) -> None:
        metric = StructuredSpaceDiversityMetric(space=PermutationSpace(size=3))

        assert metric.distance((0, 1, 2), (0, 1, 2)) == 0.0
        assert metric.distance((0, 1, 2), (1, 2, 0)) == 1.0
        assert (
            StructuredSpaceDiversityMetric(
                space=PermutationSpace(size=1),
            ).distance((0,), (0,))
            == 0.0
        )

    def test_permutation_space_distance_ignores_label_gap_magnitude(self) -> None:
        space = PermutationSpace(size=5)
        metric = StructuredSpaceDiversityMetric(space=space)
        left = space.normalize((0, 1, 2, 3, 4))

        adjacent_swap = metric.distance(left, space.normalize((1, 0, 2, 3, 4)))
        far_swap = metric.distance(left, space.normalize((4, 1, 2, 3, 0)))

        assert approx_equal(adjacent_swap, math.sqrt(2.0 / 5.0))
        assert approx_equal(far_swap, adjacent_swap)

    def test_permutation_generic_geometry_uses_position_mismatch_parts(self) -> None:
        space = PermutationSpace(size=4)
        left = space.normalize((0, 1, 2, 3))
        right = space.normalize((3, 1, 0, 2))

        parts = generic_distance_parts(space, left, right)

        assert parts == StructuredDistanceParts(
            overlap_squared_distance=3.0,
            shared_leaf_count=4,
        )

    def test_permutation_compiled_and_generic_geometry_parts_match(self) -> None:
        space = PermutationSpace(size=4)
        geometry = compile_structured_geometry(space)
        left = space.normalize((0, 1, 2, 3))
        right = space.normalize((0, 3, 2, 1))

        assert geometry is not None
        assert geometry.distance_parts(left, right) == generic_distance_parts(
            space,
            left,
            right,
        )

    def test_composite_permutation_generic_geometry_uses_child_mismatch_parts(
        self,
    ) -> None:
        space = TupleSpace(PermutationSpace(size=3), CategoricalSpace(("x", "y")))
        left = space.normalize(((0, 1, 2), "x"))
        right = space.normalize(((2, 1, 0), "y"))

        parts = generic_distance_parts(space, left, right)

        assert parts == StructuredDistanceParts(
            overlap_squared_distance=3.0,
            shared_leaf_count=4,
        )

    def test_nested_permutation_generic_geometry_uses_child_mismatch_parts(
        self,
    ) -> None:
        space = TupleSpace(
            PermutationSpace(size=3),
            TupleSpace(PermutationSpace(size=3), CategoricalSpace(("x", "y"))),
        )
        left = space.normalize(((0, 1, 2), ((0, 1, 2), "x")))
        right = space.normalize(((2, 1, 0), ((1, 2, 0), "y")))

        parts = generic_distance_parts(space, left, right)

        assert parts == StructuredDistanceParts(
            overlap_squared_distance=6.0,
            shared_leaf_count=7,
        )

    def test_array_permutation_generic_geometry_uses_element_mismatch_parts(
        self,
    ) -> None:
        space = ArraySpace(PermutationSpace(size=3), length=2)
        left = space.normalize(((0, 1, 2), (0, 1, 2)))
        right = space.normalize(((2, 1, 0), (1, 2, 0)))

        parts = generic_distance_parts(space, left, right)

        assert parts == StructuredDistanceParts(
            overlap_squared_distance=5.0,
            shared_leaf_count=6,
        )

    def test_array_permutation_compiled_and_generic_geometry_parts_match(self) -> None:
        space = ArraySpace(PermutationSpace(size=3), length=2)
        geometry = compile_structured_geometry(space)
        left = space.normalize(((0, 1, 2), (0, 1, 2)))
        right = space.normalize(((2, 1, 0), (1, 2, 0)))

        assert geometry is not None
        assert geometry.distance_parts(left, right) == generic_distance_parts(
            space,
            left,
            right,
        )

    def test_non_permutation_integer_leaf_keeps_numeric_distance_law(self) -> None:
        space = TupleSpace(IntegerSpace(0, 10), PermutationSpace(size=3))
        left = space.normalize((0, (0, 1, 2)))
        right = space.normalize((10, (2, 1, 0)))

        parts = generic_distance_parts(space, left, right)

        assert parts == StructuredDistanceParts(
            overlap_squared_distance=3.0,
            shared_leaf_count=4,
        )

    def test_permutation_distance_rejects_noncanonical_candidate(self) -> None:
        space = PermutationSpace(size=3)
        metric = StructuredSpaceDiversityMetric(space=space)

        with pytest.raises(ValueError):
            _ = metric.distance((0, 1, 1), (0, 2, 1))

    def test_custom_structured_space_uses_generic_geometry_fallback(self) -> None:
        space = WrappedPairSpace(
            depth_space=IntegerSpace(1, 5),
            mode_space=CategoricalSpace(("a", "b")),
        )
        metric = StructuredSpaceDiversityMetric(space=space)

        distance = metric.distance(
            space.normalize((1, "a")),
            space.normalize((3, "b")),
        )

        expected = math.sqrt((0.5 * 0.5 + 1.0) / 2.0)
        assert approx_equal(distance, expected)

    def test_custom_structured_space_can_opt_into_compiled_geometry_provider(
        self,
    ) -> None:
        compiled_parts = StructuredDistanceParts(
            overlap_squared_distance=0.25,
            shared_leaf_count=2,
        )
        space = ProviderWrappedPairSpace(
            depth_space=IntegerSpace(1, 5),
            mode_space=CategoricalSpace(("a", "b")),
            compiled_parts=compiled_parts,
        )
        left = space.normalize((1, "a"))
        right = space.normalize((3, "b"))

        geometry = compile_structured_geometry(space)
        metric = StructuredSpaceDiversityMetric(space=space)

        assert geometry is not None
        assert metric.geometry == geometry
        assert metric.part_values_geometry is None
        assert metric.validated_part_values_geometry is None
        assert distance_parts(space, left, right) == compiled_parts
        assert approx_equal(
            math.sqrt(0.125),
            metric.distance(left, right),
        )
        assert approx_equal(
            structured_distance_between_validated_candidates(metric, left, right),
            metric.distance(left, right),
        )

    def test_metric_caches_builtin_raw_distance_part_geometry(self) -> None:
        space = RecordSpace(
            depth=IntegerSpace(1, 5),
            mode=CategoricalSpace(("a", "b")),
        )
        metric = StructuredSpaceDiversityMetric(space=space)

        assert metric.geometry is not None
        assert metric.part_values_geometry is metric.geometry
        assert metric.validated_part_values_geometry is metric.geometry
        left = space.normalize({"depth": 1, "mode": "a"})
        right = space.normalize({"depth": 3, "mode": "b"})
        assert approx_equal(
            metric.distance(left, right),
            math.sqrt((0.25 + 1.0) / 2.0),
        )

    def test_validated_record_distance_matches_public_distance(self) -> None:
        space = RecordSpace(
            depth=IntegerSpace(1, 5),
            mode=CategoricalSpace(("a", "b")),
            rate=RealSpace(0.0, 10.0),
        )
        metric = StructuredSpaceDiversityMetric(space=space)
        left = space.normalize({"depth": 1, "mode": "a", "rate": 0.0})
        right = space.normalize({"depth": 3, "mode": "b", "rate": 5.0})

        public_distance = metric.distance(left, right)
        validated_distance = structured_distance_between_validated_candidates(
            metric,
            left,
            right,
        )

        assert approx_equal(validated_distance, public_distance)

    def test_validated_tuple_distance_matches_public_nested_distance(self) -> None:
        space = TupleSpace(
            CategoricalSpace(("left", "right")),
            TupleSpace(IntegerSpace(0, 4), CategoricalSpace((b"a", b"b"))),
        )
        metric = StructuredSpaceDiversityMetric(space=space)
        left = space.normalize(("left", (0, b"a")))
        right = space.normalize(("right", (2, b"b")))

        public_distance = metric.distance(left, right)
        validated_distance = structured_distance_between_validated_candidates(
            metric,
            left,
            right,
        )

        assert approx_equal(validated_distance, public_distance)

    def test_validated_mixed_tuple_distance_matches_public_distance(self) -> None:
        space = TupleSpace(
            TupleSpace(CategoricalSpace(("a", "b")), CategoricalSpace(("a", "b"))),
            RealSpace(0.0, 10.0),
        )
        metric = StructuredSpaceDiversityMetric(space=space)
        left = space.normalize((("a", "a"), 0.0))
        right = space.normalize((("b", "a"), 5.0))

        public_distance = metric.distance(left, right)
        validated_distance = structured_distance_between_validated_candidates(
            metric,
            left,
            right,
        )

        assert approx_equal(validated_distance, public_distance)

    def test_all_categorical_record_validated_distance_preserves_exact_types(
        self,
    ) -> None:
        space = RecordSpace(
            integer_choice=CategoricalSpace((1, 2)),
            float_choice=CategoricalSpace((1.0, 2.0)),
            bytes_choice=CategoricalSpace((b"a", b"b")),
            bool_choice=CategoricalSpace((False, True)),
        )
        metric = StructuredSpaceDiversityMetric(space=space)
        left = space.normalize(
            {
                "integer_choice": 1,
                "float_choice": 1.0,
                "bytes_choice": b"a",
                "bool_choice": False,
            },
        )
        right = space.normalize(
            {
                "integer_choice": 2,
                "float_choice": 2.0,
                "bytes_choice": b"a",
                "bool_choice": True,
            },
        )

        public_distance = metric.distance(left, right)
        validated_distance = structured_distance_between_validated_candidates(
            metric,
            left,
            right,
        )

        assert approx_equal(public_distance, math.sqrt(3.0 / 4.0))
        assert approx_equal(validated_distance, public_distance)

    def test_all_categorical_tuple_validated_distance_handles_one_field(self) -> None:
        space = TupleSpace(CategoricalSpace(("only", "other")))
        metric = StructuredSpaceDiversityMetric(space=space)
        left = space.normalize(("only",))
        right = space.normalize(("other",))

        validated_distance = structured_distance_between_validated_candidates(
            metric,
            left,
            right,
        )

        assert validated_distance == 1.0

    def test_public_all_categorical_distance_rejects_noncanonical_types(self) -> None:
        space = RecordSpace(
            integer_choice=CategoricalSpace((1, 2)),
            float_choice=CategoricalSpace((1.0, 2.0)),
            bytes_choice=CategoricalSpace((b"a", b"b")),
        )
        metric = StructuredSpaceDiversityMetric(space=space)
        canonical = space.normalize(
            {
                "integer_choice": 1,
                "float_choice": 1.0,
                "bytes_choice": b"a",
            },
        )
        bool_for_integer = RecordCandidate(
            entries=(
                ("integer_choice", True),
                ("float_choice", 1.0),
                ("bytes_choice", b"a"),
            ),
        )
        int_for_float = RecordCandidate(
            entries=(
                ("integer_choice", 1),
                ("float_choice", 1),
                ("bytes_choice", b"a"),
            ),
        )
        bytearray_for_bytes = RecordCandidate(
            entries=(
                ("integer_choice", 1),
                ("float_choice", 1.0),
                ("bytes_choice", bytearray(b"a")),
            ),
        )

        with pytest.raises(TypeError, match="declared choice type"):
            _ = metric.distance(bool_for_integer, canonical)
        with pytest.raises(TypeError, match="declared choice type"):
            _ = metric.distance(int_for_float, canonical)
        with pytest.raises(TypeError, match="declared choice type"):
            _ = metric.distance(bytearray_for_bytes, canonical)

    def test_categorical_space_rejects_equal_choices_with_different_types(self) -> None:
        with pytest.raises(ValueError, match="choices must be unique"):
            _ = CategoricalSpace((1, 1.0))

    def test_empty_composite_spaces_are_rejected_before_geometry_compilation(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="TupleSpace requires at least one"):
            _ = TupleSpace()
        with pytest.raises(ValueError, match="RecordSpace requires at least one"):
            _ = RecordSpace()

    def test_generic_geometry_returns_distance_parts_for_active_topology_mismatch(
        self,
    ) -> None:
        space = ConditionalBranchSpace(
            mode_space=CategoricalSpace(("tree", "mlp")),
            depth_space=IntegerSpace(1, 5),
        )
        parts = distance_parts(
            space,
            space.normalize(("tree", 2)),
            space.normalize(("mlp", 2)),
        )

        assert parts == StructuredDistanceParts(
            overlap_squared_distance=1.0,
            shared_leaf_count=1,
            topology_mismatch_leaf_count=1,
        )

    def test_conditional_space_reports_inactive_leaf_path(self) -> None:
        space = ConditionalBranchSpace(
            mode_space=CategoricalSpace(("tree", "mlp")),
            depth_space=IntegerSpace(1, 5),
        )
        candidate = space.normalize(("mlp", 2))

        assert space.is_active_leaf_path(candidate, ("mode",))
        assert not space.is_active_leaf_path(candidate, ("depth",))

    def test_metric_collapses_topology_mismatch_as_full_leaf_penalty(self) -> None:
        space = ConditionalBranchSpace(
            mode_space=CategoricalSpace(("tree", "mlp")),
            depth_space=IntegerSpace(1, 5),
        )
        metric = StructuredSpaceDiversityMetric(space=space)

        distance = metric.distance(
            space.normalize(("tree", 2)),
            space.normalize(("mlp", 2)),
        )

        assert distance == 1.0

    def test_conditional_branch_space_reports_non_static_topology(self) -> None:
        space = ConditionalBranchSpace(
            mode_space=CategoricalSpace(("tree", "mlp")),
            depth_space=IntegerSpace(1, 5),
        )

        assert not (space.has_static_topology())

    def test_fast_real_space_distance_rejects_out_of_bounds_candidate(self) -> None:
        metric = StructuredSpaceDiversityMetric(space=RealSpace(0.0, 10.0))

        with pytest.raises(ValueError):
            _ = metric.distance(-1.0, 2.0)

    def test_fast_array_space_distance_rejects_wrong_length(self) -> None:
        metric = StructuredSpaceDiversityMetric(
            space=ArraySpace(RealSpace(0.0, 10.0), length=2),
        )

        with pytest.raises(ValueError):
            _ = metric.distance((1.0,), (2.0,))

    def test_fast_binary_array_distance_rejects_non_binary_value(self) -> None:
        metric = StructuredSpaceDiversityMetric(
            space=ArraySpace(IntegerSpace(0, 1), length=2),
        )

        with pytest.raises(ValueError):
            _ = metric.distance((0, 2), (0, 1))
