"""Reusable conformance test cases for variopt contracts."""

from typing import Generic, TypeVar, cast

import numpy as np
import pytest

from variopt import (
    Observation,
    RunResult,
    SearchSpace,
)
from variopt.artifacts import Trace, TraceEvent
from variopt.randomness import normalize_random_state

RawCandidateT = TypeVar("RawCandidateT")
CandidateT = TypeVar("CandidateT")
ResultT = TypeVar("ResultT")


class SearchSpaceConformanceCase(Generic[RawCandidateT, CandidateT]):
    """Reusable contract tests for SearchSpace implementations."""

    __test__: bool = False

    def make_space(self) -> SearchSpace[RawCandidateT, CandidateT]:
        """Return the concrete search space under test."""
        raise NotImplementedError

    def make_boundary_candidate(self) -> RawCandidateT:
        """Return a valid raw boundary candidate."""
        raise NotImplementedError

    def make_expected_candidate(self) -> CandidateT:
        """Return the expected canonical candidate for the boundary input."""
        raise NotImplementedError

    def make_invalid_boundary_candidate(self) -> object:
        """Return a boundary input that must fail during normalization."""
        raise NotImplementedError

    def assert_candidate_equal(self, left: CandidateT, right: CandidateT) -> None:
        """Assert semantic equality for canonical candidates."""
        assert left == right

    def boundary_exception_types(self) -> tuple[type[BaseException], ...]:
        """Return the exception types allowed for invalid boundary input."""
        return (TypeError, ValueError)

    def sample_seed(self) -> int:
        """Return the deterministic seed used for sample conformance checks."""
        return 17

    def test_conformance_normalize_produces_expected_candidate(self) -> None:
        space = self.make_space()

        candidate = space.normalize(self.make_boundary_candidate())

        self.assert_candidate_equal(candidate, self.make_expected_candidate())

    def test_conformance_normalize_is_idempotent_for_canonical_candidate(self) -> None:
        space = self.make_space()
        canonical_candidate = self.make_expected_candidate()

        normalized_candidate = space.normalize(cast(RawCandidateT, canonical_candidate))

        self.assert_candidate_equal(normalized_candidate, canonical_candidate)

    def test_conformance_normalize_produces_valid_candidate(self) -> None:
        space = self.make_space()

        candidate = space.normalize(self.make_boundary_candidate())

        space.validate(candidate)

    def test_conformance_sample_produces_valid_candidate(self) -> None:
        space = self.make_space()

        candidate = space.sample(normalize_random_state(self.sample_seed()))

        space.validate(candidate)

    def test_conformance_sample_is_reproducible_for_same_seed(self) -> None:
        space = self.make_space()

        candidate_one = space.sample(normalize_random_state(self.sample_seed()))
        candidate_two = space.sample(normalize_random_state(self.sample_seed()))

        self.assert_candidate_equal(candidate_one, candidate_two)

    def test_conformance_candidates_equal_accepts_expected_candidate(self) -> None:
        space = self.make_space()
        candidate = space.normalize(self.make_boundary_candidate())

        assert space.candidates_equal(candidate, self.make_expected_candidate())

    def test_conformance_candidates_equal_rejects_invalid_candidate(self) -> None:
        space = self.make_space()

        with pytest.raises(self.boundary_exception_types()):
            _ = space.candidates_equal(
                cast(CandidateT, self.make_invalid_boundary_candidate()),
                self.make_expected_candidate(),
            )

    def test_conformance_invalid_boundary_candidate_fails(self) -> None:
        space = self.make_space()

        with pytest.raises(self.boundary_exception_types()):
            _ = space.normalize(
                cast(RawCandidateT, self.make_invalid_boundary_candidate())
            )


class ExplicitRandomnessConformanceCase(Generic[ResultT]):
    """Reusable contract tests for explicit-RNG stochastic components."""

    __test__: bool = False

    def exercise_with_rng(self, _random_state: np.random.RandomState) -> ResultT:
        """Run the stochastic component once with the supplied RNG."""
        raise NotImplementedError

    def assert_result_equal(self, left: ResultT, right: ResultT) -> None:
        """Assert semantic equality for component outputs."""
        assert left == right

    def component_seed(self) -> int:
        """Return the deterministic seed used for component conformance."""
        return 11

    def test_conformance_repeated_seed_reproduces_same_result(self) -> None:
        result_one = self.exercise_with_rng(
            normalize_random_state(self.component_seed())
        )
        result_two = self.exercise_with_rng(
            normalize_random_state(self.component_seed())
        )

        self.assert_result_equal(result_one, result_two)

    def test_conformance_component_does_not_touch_global_rng(self) -> None:
        np.random.seed(999)

        _ = self.exercise_with_rng(normalize_random_state(self.component_seed()))
        after = np.random.random_sample(8)

        np.random.seed(999)
        expected = np.random.random_sample(8)

        np.testing.assert_array_equal(after, expected)


class ArtifactConformanceCase(Generic[CandidateT]):
    """Reusable contract tests for runtime artifacts."""

    __test__: bool = False

    def make_refined_observation(self) -> Observation[CandidateT]:
        """Return an observation whose evaluated candidate differs from the proposal."""
        raise NotImplementedError

    def make_worse_observation(self) -> Observation[CandidateT]:
        """Return an observation with a worse minimization score."""
        raise NotImplementedError

    def make_better_observation(self) -> Observation[CandidateT]:
        """Return an observation with a better minimization score."""
        raise NotImplementedError

    def make_trace_event(self) -> TraceEvent:
        """Return a representative diagnostics event."""
        raise NotImplementedError

    def test_conformance_observation_separates_proposal_and_evaluated_candidate(
        self,
    ) -> None:
        observation = self.make_refined_observation()

        assert observation.proposal.candidate != observation.candidate
        assert isinstance(observation.value, float)
        assert isinstance(observation.score, float)

    def test_conformance_run_result_selects_best_observation(self) -> None:
        worse_observation = self.make_worse_observation()
        better_observation = self.make_better_observation()
        observations = (worse_observation, better_observation)

        result: RunResult[CandidateT] = RunResult[CandidateT].from_observations(
            observations=observations,
        )

        assert worse_observation.score > better_observation.score
        assert result.best_observation == better_observation
        assert result.observations == observations

    def test_conformance_trace_is_append_only(self) -> None:
        trace = Trace()
        event = self.make_trace_event()
        updated_trace = trace.append(event)

        assert trace.events == ()
        assert updated_trace.events == (event,)
