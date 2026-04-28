"""Tests for structured discrete local-search kernels."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, TypeVar

import numpy as np
import pytest
from typing_extensions import override

from variopt import (
    ArraySpace,
    CategoricalSpace,
    EvaluationOutcome,
    IntegerSpace,
    Objective,
    Observation,
    Problem,
    Proposal,
    RealSpace,
)
from variopt.algorithms.local_search import (
    StructuredHillClimbKernel,
    StructuredIteratedLocalSearchKernel,
    StructuredKickPolicy,
    StructuredScheduledLocalSearchKernel,
    StructuredStochasticNeighborhoodKernel,
    StructuredVariableNeighborhoodKernel,
    StructuredVariableNeighborhoodStage,
)
from variopt.execution import (
    ExecutionResources,
    NestedParallelismPolicy,
)
from variopt.kernel import (
    KernelStatus,
    ProposalBatchQuery,
    ProposalKernelHint,
    ProposalLocalSearchContext,
)
from variopt.spaces import (
    LeafPath,
    RecordCandidate,
    RecordSpace,
    SpaceBoundaryValue,
    StructuredLeafSpace,
    StructuredSearchSpace,
)
from variopt.spaces.types import SpaceCandidateValue

Color: TypeAlias = Literal["red", "green", "blue"]
BoundaryRunnerT = TypeVar("BoundaryRunnerT")
CandidateRunnerT = TypeVar("CandidateRunnerT")
ConditionalDiscreteCandidate = tuple[int, int]


@dataclass(frozen=True, slots=True)
class DummyProposalKernelHint(ProposalKernelHint):
    """Unexpected generic kernel hint used to test local-search validation."""


def evaluate_query_directly(
    query: ProposalBatchQuery[BoundaryRunnerT, CandidateRunnerT],
) -> tuple[EvaluationOutcome[CandidateRunnerT], ...]:
    """Evaluate one proposal batch directly through the problem objective."""
    return tuple(
        EvaluationOutcome(
            observation=Observation.from_objective_value(
                proposal=proposal,
                candidate=proposal.candidate,
                value=query.problem.objective.evaluate(proposal.candidate),
                direction=query.problem.direction,
            ),
            evaluation_count=1,
        )
        for proposal in query.proposals
    )


class IntegerObjective(Objective[int]):
    """One-dimensional integer objective with a known optimum."""

    @override
    def evaluate(self, candidate: int) -> float:
        return float((candidate - 2) ** 2)


class CategoricalObjective(Objective[Color]):
    """Categorical objective with an ordered preference."""

    @override
    def evaluate(self, candidate: Color) -> float:
        if candidate == "blue":
            return 0.0
        if candidate == "green":
            return 1.0
        return 2.0


class FlatCategoricalObjective(Objective[int]):
    """Categorical objective where the incumbent is already globally best."""

    @override
    def evaluate(self, candidate: int) -> float:
        if candidate == 0:
            return 0.0
        return 1.0


class ImproveAwayFromZeroObjective(Objective[int]):
    """Categorical objective where any non-zero sampled move improves."""

    @override
    def evaluate(self, candidate: int) -> float:
        if candidate == 0:
            return 1.0
        return 0.0


class ArrayIntegerObjective(Objective[tuple[int, ...]]):
    """Small array objective used to exercise recursive leaf traversal."""

    @override
    def evaluate(self, candidate: tuple[int, ...]) -> float:
        return float(abs(candidate[0] - 1) + abs(candidate[1] - 2))


class MixedRecordObjective(Objective[RecordCandidate]):
    """Mixed discrete record objective over integer and categorical fields."""

    @override
    def evaluate(self, candidate: RecordCandidate) -> float:
        level = record_int(candidate, "level")
        color = record_str(candidate, "color")
        color_penalty = 2.0
        if color == "green":
            color_penalty = 1.0
        elif color == "blue":
            color_penalty = 0.0
        return float(abs(level - 2) + color_penalty)


class UnsupportedRecordObjective(Objective[RecordCandidate]):
    """Dummy record objective used to reject unsupported leaf spaces."""

    @override
    def evaluate(self, candidate: RecordCandidate) -> float:
        _ = candidate
        return 0.0


class PairMoveObjective(Objective[tuple[int, ...]]):
    """Two-leaf objective where only a coordinated pair move improves."""

    @override
    def evaluate(self, candidate: tuple[int, ...]) -> float:
        if candidate == (1, 1):
            return 0.0
        if candidate == (0, 0):
            return 1.0
        return 3.0


class PrioritizedPairMoveObjective(Objective[tuple[int, ...]]):
    """Three-leaf objective where only one prioritized pair move improves."""

    @override
    def evaluate(self, candidate: tuple[int, ...]) -> float:
        if candidate == (0, 1, 1):
            return 0.0
        if candidate == (0, 0, 0):
            return 1.0
        return 3.0


class VariableNeighborhoodResetObjective(Objective[tuple[int, ...]]):
    """Three-leaf objective that requires stage reset after a pair improvement."""

    @override
    def evaluate(self, candidate: tuple[int, ...]) -> float:
        if candidate == (1, 1, 1):
            return 0.0
        if candidate == (1, 1, 0):
            return 1.0
        if candidate == (0, 0, 0):
            return 2.0
        return 4.0


class StrictKickAcceptanceObjective(Objective[tuple[int, ...]]):
    """Three-leaf objective where one-leaf kicks improve only back to the incumbent."""

    @override
    def evaluate(self, candidate: tuple[int, ...]) -> float:
        if candidate == (0, 0, 0):
            return 1.0
        if candidate == (1, 1, 0):
            return 0.0
        return 3.0


class ConditionalDiscreteObjective(Objective[ConditionalDiscreteCandidate]):
    """Two-leaf objective used only to trigger dynamic-topology rejection."""

    @override
    def evaluate(self, candidate: ConditionalDiscreteCandidate) -> float:
        if candidate == (1, 1):
            return 0.0
        if candidate == (0, 0):
            return 1.0
        return 3.0


@dataclass(frozen=True)
class ConditionalDiscretePairSpace(
    StructuredSearchSpace[ConditionalDiscreteCandidate, ConditionalDiscreteCandidate],
):
    """Test-only discrete structured space with candidate-conditioned topology."""

    head_space: IntegerSpace
    tail_space: IntegerSpace

    @override
    def normalize(
        self,
        raw_candidate: ConditionalDiscreteCandidate,
    ) -> ConditionalDiscreteCandidate:
        return (
            self.head_space.normalize(raw_candidate[0]),
            self.tail_space.normalize(raw_candidate[1]),
        )

    @override
    def validate(self, candidate: ConditionalDiscreteCandidate) -> None:
        self.head_space.validate(candidate[0])
        self.tail_space.validate(candidate[1])

    @override
    def sample(self, random_state: np.random.RandomState) -> ConditionalDiscreteCandidate:
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
        candidate: ConditionalDiscreteCandidate,
    ) -> tuple[LeafPath, ...]:
        self.validate(candidate)
        if candidate[0] > 0:
            return (("head",), ("tail",))
        return (("head",),)

    @override
    def leaf_space_at_path(self, path: LeafPath) -> StructuredLeafSpace:
        if path == ("head",):
            return self.head_space
        if path == ("tail",):
            return self.tail_space
        msg = f"invalid conditional discrete pair path: {path!r}"
        raise TypeError(msg)

    @override
    def leaf_value_at_path(
        self,
        candidate: ConditionalDiscreteCandidate,
        path: LeafPath,
    ) -> SpaceCandidateValue:
        self.validate(candidate)
        if path == ("head",):
            return candidate[0]
        if path == ("tail",):
            return candidate[1]
        msg = f"invalid conditional discrete pair path: {path!r}"
        raise TypeError(msg)

    @override
    def replace_leaf_values(
        self,
        candidate: ConditionalDiscreteCandidate,
        replacements: Mapping[LeafPath, SpaceCandidateValue],
    ) -> ConditionalDiscreteCandidate:
        self.validate(candidate)
        head_value = candidate[0]
        tail_value = candidate[1]
        if ("head",) in replacements:
            replacement = replacements[("head",)]
            if type(replacement) is not int:
                msg = "conditional discrete head replacement must be a canonical integer"
                raise TypeError(msg)
            head_value = self.head_space.normalize(replacement)
        if ("tail",) in replacements:
            replacement = replacements[("tail",)]
            if type(replacement) is not int:
                msg = "conditional discrete tail replacement must be a canonical integer"
                raise TypeError(msg)
            tail_value = self.tail_space.normalize(replacement)
        return (head_value, tail_value)


class StructuredHillClimbKernelTests:
    """Regression tests for discrete structured local-search kernels."""

    def make_execution_resources(self) -> ExecutionResources:
        """Return canonical evaluator-owned execution resources."""
        return ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
            owner_worker_count=1,
            owner_backend="sequential",
        )

    def test_integer_hill_climber_converges_to_local_optimum(self) -> None:
        problem = Problem(
            space=IntegerSpace(0, 5),
            objective=IntegerObjective(),
        )
        kernel = StructuredHillClimbKernel[int, int](max_steps=8)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=5, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert outcome.observation.candidate == 2
        assert outcome.observation.value == 0.0
        assert outcome.evaluation_count == 6
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.backend == "structured.local_search"
        assert outcome.kernel_diagnostics.method == "leafwise_first_improvement"
        assert outcome.kernel_diagnostics.status == KernelStatus.CONVERGED

    def test_categorical_hill_climber_moves_through_declared_alternatives(self) -> None:
        color_space: CategoricalSpace[Color] = CategoricalSpace(("red", "green", "blue"))
        initial_candidate: Color = "red"
        problem: Problem[Color, Color, Observation[Color]] = Problem(
            space=color_space,
            objective=CategoricalObjective(),
        )
        kernel = StructuredHillClimbKernel[Color, Color](max_steps=4)
        query: ProposalBatchQuery[Color, Color, Observation[Color]] = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal[Color](candidate=initial_candidate, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert outcome.observation.candidate == "blue"
        assert outcome.observation.value == 0.0
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.CONVERGED

    def test_hill_climber_recurses_over_array_leaves(self) -> None:
        problem = Problem(
            space=ArraySpace(IntegerSpace(0, 3), length=2),
            objective=ArrayIntegerObjective(),
        )
        kernel = StructuredHillClimbKernel[Sequence[int], tuple[int, ...]](max_steps=8)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert outcome.observation.candidate == (1, 2)
        assert outcome.observation.value == 0.0
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.CONVERGED

    def test_hill_climber_stops_when_step_budget_is_exhausted(self) -> None:
        space = RecordSpace(
            level=IntegerSpace(0, 5),
            color=CategoricalSpace(("red", "green", "blue")),
        )
        problem = Problem(
            space=space,
            objective=MixedRecordObjective(),
        )
        initial_candidate = space.normalize({"level": 0, "color": "red"})
        kernel = StructuredHillClimbKernel[
            Mapping[str, SpaceBoundaryValue] | RecordCandidate,
            RecordCandidate,
        ](max_steps=1)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=initial_candidate, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert record_int(outcome.observation.candidate, "level") == 1
        assert record_str(outcome.observation.candidate, "color") == "red"
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED

    def test_context_can_disable_local_search(self) -> None:
        problem = Problem(
            space=IntegerSpace(0, 5),
            objective=IntegerObjective(),
        )
        kernel = StructuredHillClimbKernel[int, int](max_steps=8)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=5, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
            proposal_kernel_hints=(
                ProposalLocalSearchContext(enabled=False),
            ),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert outcome.observation.candidate == 5
        assert outcome.observation.value == 9.0
        assert outcome.evaluation_count == 1
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert outcome.kernel_diagnostics.message == "local search disabled by run-method context"

    def test_context_can_override_step_budget(self) -> None:
        space = RecordSpace(
            level=IntegerSpace(0, 5),
            color=CategoricalSpace(("red", "green", "blue")),
        )
        problem = Problem(
            space=space,
            objective=MixedRecordObjective(),
        )
        initial_candidate = space.normalize({"level": 0, "color": "red"})
        kernel = StructuredHillClimbKernel[
            Mapping[str, SpaceBoundaryValue] | RecordCandidate,
            RecordCandidate,
        ](max_steps=8)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=initial_candidate, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
            proposal_kernel_hints=(
                ProposalLocalSearchContext(local_budget=1),
            ),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert record_int(outcome.observation.candidate, "level") == 1
        assert record_str(outcome.observation.candidate, "color") == "red"
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED

    def test_context_can_prioritize_leaf_order(self) -> None:
        space = RecordSpace(
            level=IntegerSpace(0, 5),
            color=CategoricalSpace(("red", "green", "blue")),
        )
        problem = Problem(
            space=space,
            objective=MixedRecordObjective(),
        )
        initial_candidate = space.normalize({"level": 0, "color": "red"})
        kernel = StructuredHillClimbKernel[
            Mapping[str, SpaceBoundaryValue] | RecordCandidate,
            RecordCandidate,
        ](max_steps=1)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=initial_candidate, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
            proposal_kernel_hints=(
                ProposalLocalSearchContext(prioritized_leaf_paths=(("color",),)),
            ),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert record_int(outcome.observation.candidate, "level") == 0
        assert record_str(outcome.observation.candidate, "color") == "green"
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED

    def test_rejects_non_local_search_kernel_hint(self) -> None:
        problem = Problem(
            space=IntegerSpace(0, 5),
            objective=IntegerObjective(),
        )
        kernel = StructuredHillClimbKernel[int, int](max_steps=8)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=5, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
            proposal_kernel_hints=(DummyProposalKernelHint(),),
        )

        with pytest.raises(TypeError, match="ProposalLocalSearchContext hints"):
            _ = kernel.run(query, evaluate_query_directly)

    def test_hill_climber_rejects_dynamic_topology_space(self) -> None:
        problem = Problem(
            space=ConditionalDiscretePairSpace(
                head_space=IntegerSpace(0, 3),
                tail_space=IntegerSpace(0, 3),
            ),
            objective=ConditionalDiscreteObjective(),
        )
        kernel = StructuredHillClimbKernel[
            ConditionalDiscreteCandidate,
            ConditionalDiscreteCandidate,
        ](max_steps=4)
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        with pytest.raises(TypeError, match="static topology"):
            _ = kernel.run(query, evaluate_query_directly)

    def test_rejects_real_valued_leaves(self) -> None:
        problem = Problem(
            space=RecordSpace(
                level=IntegerSpace(0, 5),
                weight=RealSpace(0.0, 5.0),
            ),
            objective=UnsupportedRecordObjective(),
        )
        kernel = StructuredHillClimbKernel[
            Mapping[str, SpaceBoundaryValue] | RecordCandidate,
            RecordCandidate,
        ](max_steps=4)
        candidate = problem.space.normalize({"level": 1, "weight": 2.0})
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=candidate, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        with pytest.raises(TypeError, match="IntegerSpace or CategoricalSpace"):
            _ = kernel.run(query, evaluate_query_directly)


class StructuredScheduledLocalSearchKernelTests:
    """Regression tests for staged structured local-search kernels."""

    def make_execution_resources(self) -> ExecutionResources:
        """Return canonical evaluator-owned execution resources."""
        return ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
            owner_worker_count=1,
            owner_backend="sequential",
        )

    def test_pair_move_stage_can_escape_single_leaf_trap(self) -> None:
        problem = Problem(
            space=ArraySpace(IntegerSpace(0, 1), length=2),
            objective=PairMoveObjective(),
        )
        kernel = StructuredScheduledLocalSearchKernel[
            Sequence[int],
            tuple[int, ...],
        ](
            max_steps=1,
            pair_move_leaf_limit=2,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert outcome.observation.candidate == (1, 1)
        assert outcome.observation.value == 0.0
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.method == "scheduled_single_then_pair"
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED

    def test_prioritized_leaf_paths_can_steer_pair_move_stage(self) -> None:
        problem = Problem(
            space=ArraySpace(IntegerSpace(0, 1), length=3),
            objective=PrioritizedPairMoveObjective(),
        )
        kernel = StructuredScheduledLocalSearchKernel[
            Sequence[int],
            tuple[int, ...],
        ](
            max_steps=1,
            pair_move_leaf_limit=2,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
            proposal_kernel_hints=(
                ProposalLocalSearchContext(
                    prioritized_leaf_paths=((1,), (2,)),
                ),
            ),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert outcome.observation.candidate == (0, 1, 1)
        assert outcome.observation.value == 0.0
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED


class StructuredStochasticNeighborhoodKernelTests:
    """Regression tests for bounded stochastic structured local-search."""

    def make_execution_resources(self) -> ExecutionResources:
        """Return canonical evaluator-owned execution resources."""
        return ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
            owner_worker_count=1,
            owner_backend="sequential",
        )

    def test_stochastic_kernel_stops_after_sampled_neighborhood_without_improvement(
        self,
    ) -> None:
        problem = Problem(
            space=CategoricalSpace(tuple(range(6))),
            objective=FlatCategoricalObjective(),
        )
        kernel = StructuredStochasticNeighborhoodKernel[int, int](
            max_steps=4,
            max_neighbors_per_step=2,
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=0, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert outcome.observation.candidate == 0
        assert outcome.observation.value == 0.0
        assert outcome.evaluation_count == 3
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.method == "sampled_leafwise_first_improvement"
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert outcome.kernel_diagnostics.message == "no improving move found in the sampled discrete neighborhood"

    def test_stochastic_kernel_converges_when_sampling_covers_full_neighborhood(
        self,
    ) -> None:
        problem = Problem(
            space=CategoricalSpace(tuple(range(3))),
            objective=FlatCategoricalObjective(),
        )
        kernel = StructuredStochasticNeighborhoodKernel[int, int](
            max_steps=4,
            max_neighbors_per_step=4,
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=0, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert outcome.evaluation_count == 3
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.CONVERGED
        assert outcome.kernel_diagnostics.message == "no improving move found in the full discrete neighborhood"

    def test_stochastic_kernel_can_cap_categorical_neighbors_per_leaf(self) -> None:
        problem = Problem(
            space=CategoricalSpace(tuple(range(8))),
            objective=FlatCategoricalObjective(),
        )
        kernel = StructuredStochasticNeighborhoodKernel[int, int](
            max_steps=4,
            max_neighbors_per_step=16,
            max_categorical_neighbors_per_leaf=3,
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=0, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert outcome.evaluation_count == 4
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert outcome.kernel_diagnostics.message == "no improving move found in the sampled discrete neighborhood"

    def test_stochastic_kernel_can_take_one_sampled_improving_move(self) -> None:
        problem = Problem(
            space=CategoricalSpace(tuple(range(6))),
            objective=ImproveAwayFromZeroObjective(),
        )
        kernel = StructuredStochasticNeighborhoodKernel[int, int](
            max_steps=1,
            max_neighbors_per_step=1,
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=0, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert outcome.observation.candidate != 0
        assert outcome.observation.value == 0.0
        assert outcome.evaluation_count == 2
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert outcome.kernel_diagnostics.message == "max_steps reached before stochastic local-search termination"

    def test_stochastic_kernel_rejects_dynamic_topology_space(self) -> None:
        problem = Problem(
            space=ConditionalDiscretePairSpace(
                head_space=IntegerSpace(0, 3),
                tail_space=IntegerSpace(0, 3),
            ),
            objective=ConditionalDiscreteObjective(),
        )
        kernel = StructuredStochasticNeighborhoodKernel[
            ConditionalDiscreteCandidate,
            ConditionalDiscreteCandidate,
        ](
            max_steps=4,
            max_neighbors_per_step=2,
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        with pytest.raises(TypeError, match="static topology"):
            _ = kernel.run(query, evaluate_query_directly)

    def test_stochastic_kernel_rejects_invalid_sampling_budgets(self) -> None:
        with pytest.raises(ValueError, match="max_neighbors_per_step"):
            _ = StructuredStochasticNeighborhoodKernel[int, int](
                max_neighbors_per_step=0,
            )

        with pytest.raises(ValueError, match="max_categorical_neighbors_per_leaf"):
            _ = StructuredStochasticNeighborhoodKernel[int, int](
                max_categorical_neighbors_per_leaf=0,
            )


class StructuredVariableNeighborhoodKernelTests:
    """Regression tests for structured variable-neighborhood wrappers."""

    def make_execution_resources(self) -> ExecutionResources:
        """Return canonical evaluator-owned execution resources."""
        return ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
            owner_worker_count=1,
            owner_backend="sequential",
        )

    def test_variable_neighborhood_kernel_resets_to_first_stage_after_improvement(
        self,
    ) -> None:
        problem = Problem(
            space=ArraySpace(IntegerSpace(0, 1), length=3),
            objective=VariableNeighborhoodResetObjective(),
        )
        kernel = StructuredVariableNeighborhoodKernel[
            Sequence[int],
            tuple[int, ...],
        ](
            max_steps=2,
            stages=(
                StructuredVariableNeighborhoodStage.leafwise_first_improvement(),
                StructuredVariableNeighborhoodStage.scheduled_single_then_pair(
                    pair_move_leaf_limit=2,
                ),
            ),
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert outcome.observation.candidate == (1, 1, 1)
        assert outcome.observation.value == 0.0
        assert outcome.evaluation_count == 11
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.method == "variable_neighborhood_search"
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert outcome.kernel_diagnostics.message == "max_steps reached before variable-neighborhood termination"

    def test_variable_neighborhood_kernel_uses_sampled_stage_terminal_status(
        self,
    ) -> None:
        problem = Problem(
            space=CategoricalSpace(tuple(range(6))),
            objective=FlatCategoricalObjective(),
        )
        kernel = StructuredVariableNeighborhoodKernel[int, int](
            max_steps=4,
            stages=(
                StructuredVariableNeighborhoodStage.sampled_leafwise_first_improvement(
                    max_neighbors_per_step=2,
                ),
            ),
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=0, proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert outcome.observation.candidate == 0
        assert outcome.evaluation_count == 3
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert outcome.kernel_diagnostics.message == (
                "no improving move found in the sampled variable neighborhood "
                "after exhausting the configured variable-neighborhood stages"
            )

    def test_variable_neighborhood_kernel_rejects_invalid_stage_metadata(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="stages must contain at least one"):
            _ = StructuredVariableNeighborhoodKernel[int, int](stages=())

        with pytest.raises(ValueError, match="must not set max_neighbors_per_step"):
            _ = StructuredVariableNeighborhoodStage(
                kind="leafwise_first_improvement",
                max_neighbors_per_step=1,
            )

        with pytest.raises(ValueError, match="pair_move_leaf_limit must be positive"):
            _ = StructuredVariableNeighborhoodStage.scheduled_single_then_pair(
                pair_move_leaf_limit=0,
            )


class StructuredIteratedLocalSearchKernelTests:
    """Regression tests for structured iterated local-search wrappers."""

    def make_execution_resources(self) -> ExecutionResources:
        """Return canonical evaluator-owned execution resources."""
        return ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
            owner_worker_count=1,
            owner_backend="sequential",
        )

    def test_iterated_local_search_kernel_can_escape_pair_move_trap_via_kick(
        self,
    ) -> None:
        problem = Problem(
            space=ArraySpace(IntegerSpace(0, 1), length=2),
            objective=PairMoveObjective(),
        )
        kernel = StructuredIteratedLocalSearchKernel[
            Sequence[int],
            tuple[int, ...],
        ](
            max_steps=4,
            max_kicks=1,
            kick_policy=StructuredKickPolicy(kick_leaf_count=2),
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert outcome.observation.candidate == (1, 1)
        assert outcome.observation.value == 0.0
        assert outcome.evaluation_count == 6
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.method == "iterated_local_search"
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert outcome.kernel_diagnostics.message == "max_kicks reached before iterated local-search termination"

    def test_iterated_local_search_kernel_requires_strict_improvement_to_accept(
        self,
    ) -> None:
        problem = Problem(
            space=ArraySpace(IntegerSpace(0, 1), length=3),
            objective=StrictKickAcceptanceObjective(),
        )
        kernel = StructuredIteratedLocalSearchKernel[
            Sequence[int],
            tuple[int, ...],
        ](
            max_steps=4,
            max_kicks=1,
            kick_policy=StructuredKickPolicy(kick_leaf_count=1),
            random_state=0,
        )
        query = ProposalBatchQuery(
            problem=problem,
            proposals=(Proposal(candidate=(0, 0, 0), proposal_id="p-1"),),
            execution_resources=self.make_execution_resources(),
        )

        outcomes = kernel.run(query, evaluate_query_directly)

        outcome = outcomes[0]
        assert outcome.observation.candidate == (0, 0, 0)
        assert outcome.observation.value == 1.0
        assert outcome.evaluation_count == 11
        assert outcome.kernel_diagnostics is not None
        assert outcome.kernel_diagnostics.status == KernelStatus.STOPPED
        assert outcome.kernel_diagnostics.message == "max_kicks reached before iterated local-search termination"

    def test_iterated_local_search_kernel_rejects_invalid_kick_metadata(self) -> None:
        with pytest.raises(ValueError, match="max_kicks must be positive"):
            _ = StructuredIteratedLocalSearchKernel[int, int](max_kicks=0)

        with pytest.raises(ValueError, match="kick_leaf_count must be positive"):
            _ = StructuredKickPolicy(kick_leaf_count=0)

        with pytest.raises(ValueError, match="max_categorical_alternatives_per_leaf must be positive"):
            _ = StructuredKickPolicy(max_categorical_alternatives_per_leaf=0)


def record_int(candidate: RecordCandidate, field_name: str) -> int:
    """Return one canonical integer-valued record field."""
    value = candidate[field_name]
    if type(value) is not int:
        msg = f"record field {field_name!r} must be a canonical integer"
        raise TypeError(msg)
    return value


def record_str(candidate: RecordCandidate, field_name: str) -> str:
    """Return one canonical string-valued record field."""
    value = candidate[field_name]
    if type(value) is not str:
        msg = f"record field {field_name!r} must be a canonical string"
        raise TypeError(msg)
    return value
