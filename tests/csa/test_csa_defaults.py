"""Tests for space-derived CSA default components."""

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from typing_extensions import override

from variopt import (
    IntegerSpace,
    Objective,
    PermutationSpace,
    Problem,
    Proposal,
    Study,
    TupleSpace,
)
from variopt.algorithms.population.csa import (
    BoundedMutation,
    CSAOptimizer,
    CSAPerturbationSchedule,
    CSAPerturbationSpec,
    CSAProfile,
    CSAProposalPolicy,
    UniformCrossover,
    derive_csa_defaults,
)
from variopt.algorithms.population.csa.engine.ask import (
    apply_variation_operator_from_validated_parents,
)
from variopt.algorithms.population.csa.generation.proposal.logic import (
    record_proposal_attribution,
    sample_mutation_family_indices,
)
from variopt.algorithms.population.csa.generation.proposal.state import (
    CSAProposalState,
    ProposalAttribution,
)
from variopt.algorithms.population.csa.generation.proposal.state.generation_evidence import (
    ProposalGenerationAdaptationEvidence,
    ProposalLeafAdaptationSummary,
)
from variopt.algorithms.population.permutation import (
    InversionMutation,
    OrderCrossover,
    SwapMutation,
)
from variopt.diversity import (
    DiversityMetric,
    StructuredSpaceDiversityMetric,
)
from variopt.evaluators import SequentialEvaluator
from variopt.operators import VariationOperator
from variopt.sampling import CandidateSampler, SearchSpaceSampler


class IntegerSquareObjective(Objective[int]):
    """Scalar quadratic objective with a minimum at zero."""

    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate * candidate)


class PermutationMismatchObjective(Objective[tuple[int, ...]]):
    """Objective that counts positional mismatches against the identity permutation."""

    @override
    def evaluate(self, candidate: tuple[int, ...]) -> float:
        mismatch_count = 0
        for index, value in enumerate(candidate):
            if index != value:
                mismatch_count += 1
        return float(mismatch_count)


class CSADefaultComponentTests:
    """Regression tests for CSA default derivation from space semantics."""

    def test_derive_csa_defaults_returns_metric_and_schedule(self) -> None:
        defaults = derive_csa_defaults(IntegerSpace(-10, 10))

        assert isinstance(defaults.sampler, SearchSpaceSampler)
        assert isinstance(defaults.diversity_metric, StructuredSpaceDiversityMetric)
        assert isinstance(defaults.perturbation_schedule, CSAPerturbationSchedule)
        assert len(defaults.perturbation_schedule.regular_family) == 1
        assert len(defaults.perturbation_schedule.initial_family) == 1
        assert len(defaults.perturbation_schedule.mutation_family) == 2
        assert defaults.perturbation_schedule.regular_family[0].count == 2
        assert defaults.perturbation_schedule.initial_family[0].count == 2
        assert defaults.perturbation_schedule.mutation_family[0].count == 2
        assert defaults.perturbation_schedule.mutation_family[1].count == 1

    def test_derive_csa_defaults_supports_joung_2018_style(self) -> None:
        defaults = derive_csa_defaults(
            IntegerSpace(-10, 10),
            style="joung_2018",
        )

        assert len(defaults.perturbation_schedule.regular_family) == 1
        assert len(defaults.perturbation_schedule.initial_family) == 1
        assert len(defaults.perturbation_schedule.mutation_family) == 1
        assert defaults.perturbation_schedule.regular_family[0].count == 10
        assert defaults.perturbation_schedule.initial_family[0].count == 10
        assert defaults.perturbation_schedule.mutation_family[0].count == 10

        regular_operator = defaults.perturbation_schedule.regular_family[0].operator
        initial_operator = defaults.perturbation_schedule.initial_family[0].operator
        mutation_operator = defaults.perturbation_schedule.mutation_family[0].operator

        assert isinstance(regular_operator, UniformCrossover)
        assert regular_operator.max_exchange_fraction == 0.5
        assert isinstance(initial_operator, UniformCrossover)
        assert initial_operator.max_exchange_fraction == 0.2
        assert isinstance(mutation_operator, BoundedMutation)
        assert mutation_operator.max_perturbation_fraction == 0.2

    def test_derive_csa_defaults_uses_permutation_safe_schedule(self) -> None:
        defaults = derive_csa_defaults(PermutationSpace(size=6))

        regular_operator = defaults.perturbation_schedule.regular_family[0].operator
        initial_operator = defaults.perturbation_schedule.initial_family[0].operator
        first_mutation_operator = defaults.perturbation_schedule.mutation_family[
            0
        ].operator
        second_mutation_operator = defaults.perturbation_schedule.mutation_family[
            1
        ].operator

        assert isinstance(regular_operator, OrderCrossover)
        assert isinstance(initial_operator, OrderCrossover)
        assert isinstance(first_mutation_operator, InversionMutation)
        assert isinstance(second_mutation_operator, SwapMutation)

    def test_derived_defaults_can_drive_csa_optimization(self) -> None:
        space = IntegerSpace(-10, 10)
        problem = Problem(
            space=space,
            objective=IntegerSquareObjective(),
        )
        optimizer = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=8,
            profile=CSAProfile(seed_count=3),
            random_state=1,
        )
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=SequentialEvaluator[int, int](),
        )

        result, _ = study.optimize(max_evaluations=40)

        assert result.best_observation is not None

    def test_from_space_defaults_uses_derived_components(self) -> None:
        space = IntegerSpace(-10, 10)
        defaults = derive_csa_defaults(space)

        optimizer = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=4,
            random_state=0,
        )

        assert isinstance(optimizer.sampler, SearchSpaceSampler)
        assert isinstance(optimizer.diversity_metric, StructuredSpaceDiversityMetric)
        assert (
            optimizer.resolved_profile.perturbation_schedule
            == defaults.perturbation_schedule
        )
        assert (
            optimizer.create_initial_state().proposal_state.policy
            == CSAProposalPolicy()
        )
        assert optimizer.resolved_profile.max_bank_capacity == 24
        assert (
            optimizer.resolved_profile.update_policy.far_update_mode == "crowding_aware"
        )

    def test_from_space_defaults_infers_joung_2018_default_schedule_from_profile(
        self,
    ) -> None:
        space = IntegerSpace(-10, 10)
        defaults = derive_csa_defaults(
            space,
            style="joung_2018",
        )

        optimizer = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=4,
            profile=CSAProfile(preset="joung_2018"),
            random_state=0,
        )

        assert optimizer.resolved_profile.seed_count == 6
        assert (
            optimizer.resolved_profile.perturbation_schedule
            == defaults.perturbation_schedule
        )

    def test_from_space_defaults_respects_explicit_overrides(self) -> None:
        space = IntegerSpace(-10, 10)
        custom_sampler = ConstantIntegerSampler(3)
        custom_metric = AbsoluteDistance()
        custom_schedule = CSAPerturbationSchedule(
            mutation_family=(CSAPerturbationSpec(IdentityMutation()),),
        )

        optimizer = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=4,
            profile=CSAProfile(
                preset="joung_2018",
                proposal_policy=CSAProposalPolicy(enabled=True),
            ),
            sampler=custom_sampler,
            diversity_metric=custom_metric,
            perturbation_schedule=custom_schedule,
            random_state=0,
        )

        assert optimizer.sampler is custom_sampler
        assert optimizer.diversity_metric is custom_metric
        assert optimizer.resolved_profile.perturbation_schedule is custom_schedule
        assert optimizer.create_initial_state().proposal_state.policy.enabled
        assert optimizer.resolved_profile.seed_count == 6

    def test_csa_optimizer_can_override_boundary_sampler(self) -> None:
        space = IntegerSpace(-10, 10)
        defaults = derive_csa_defaults(space)
        optimizer = CSAOptimizer(
            space=space,
            diversity_metric=defaults.diversity_metric,
            bank_capacity=4,
            profile=CSAProfile(
                perturbation_schedule=defaults.perturbation_schedule,
                seed_count=2,
            ),
            sampler=ConstantIntegerSampler(7),
            random_state=0,
        )

        proposals, _ = optimizer.ask(optimizer.create_initial_state(), batch_size=1)

        assert len(proposals) == 1
        assert proposals[0].candidate == 7

    def test_from_space_defaults_can_optimize_permutation_space(self) -> None:
        space = PermutationSpace(size=6)
        problem = Problem(
            space=space,
            objective=PermutationMismatchObjective(),
        )
        optimizer = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=8,
            profile=CSAProfile(seed_count=3),
            random_state=0,
        )
        study = Study(
            problem=problem,
            run_method=optimizer,
            evaluator=SequentialEvaluator[Sequence[int], tuple[int, ...]](),
        )

        result, _ = study.optimize(max_evaluations=40)

        assert result.best_observation is not None

    def test_permutation_mutation_cold_start_preserves_declared_children_and_rng(
        self,
    ) -> None:
        space = PermutationSpace(size=6)
        family = derive_csa_defaults(space).perturbation_schedule.mutation_family
        proposal_state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))
        adaptive_random_state = np.random.RandomState(37)
        fixed_random_state = np.random.RandomState(37)
        candidate = (0, 1, 2, 3, 4, 5)

        adaptive_indices = sample_mutation_family_indices(
            state=proposal_state,
            family=family,
            random_state=adaptive_random_state,
        )
        fixed_indices = tuple(
            index for index, spec in enumerate(family) for _ in range(spec.count)
        )
        adaptive_children = tuple(
            apply_variation_operator_from_validated_parents(
                operator=family[index].operator,
                parents=(candidate,),
                random_state=adaptive_random_state,
            )
            for index in adaptive_indices
        )
        fixed_children = tuple(
            apply_variation_operator_from_validated_parents(
                operator=family[index].operator,
                parents=(candidate,),
                random_state=fixed_random_state,
            )
            for index in fixed_indices
        )

        assert adaptive_indices == fixed_indices
        assert adaptive_children == fixed_children
        assert repr(adaptive_random_state.get_state()) == repr(
            fixed_random_state.get_state()
        )

    def test_csa_optimizer_can_emit_history_conditioned_local_search_contexts(
        self,
    ) -> None:
        space = TupleSpace(
            IntegerSpace(0, 5),
            IntegerSpace(0, 5),
        )
        optimizer = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=4,
            profile=CSAProfile(
                proposal_policy=CSAProposalPolicy(enabled=True),
            ),
            random_state=0,
        )
        proposal = Proposal(candidate=space.normalize((3, 1)), proposal_id="p-1")
        state = optimizer.create_initial_state()
        proposal_state = record_proposal_attribution(
            state.proposal_state,
            ProposalAttribution(
                proposal_id="p-1",
                mutated_leaf_paths=((1,),),
            ),
        ).record_generation_evidence(
            ProposalGenerationAdaptationEvidence(
                evidence_count=1,
                mutation_leaf_summaries=(
                    ProposalLeafAdaptationSummary(
                        path=(0,),
                        observation_count=1,
                        total_survival_efficiency=1.0,
                    ),
                ),
            ),
        )
        state = replace(state, proposal_state=proposal_state)

        contexts = optimizer.proposal_kernel_hints(state, (proposal,))

        assert contexts is not None
        assert len(contexts) == 1
        context = contexts[0]
        assert context is not None
        assert context.local_budget == 2
        assert context.prioritized_leaf_paths == ((1,), (0,))
        assert context.random_state_snapshot is not None
        assert optimizer.proposal_kernel_hints(state, (proposal,)) == contexts

    def test_disabled_csa_proposal_adaptation_emits_only_rng_hint(self) -> None:
        space = TupleSpace(
            IntegerSpace(0, 5),
            IntegerSpace(0, 5),
        )
        optimizer = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=4,
            profile=CSAProfile(
                proposal_policy=CSAProposalPolicy(enabled=False),
            ),
            random_state=0,
        )
        proposal = Proposal(candidate=space.normalize((3, 1)), proposal_id="p-1")
        state = optimizer.create_initial_state()

        contexts = optimizer.proposal_kernel_hints(state, (proposal,))

        assert contexts is not None
        assert len(contexts) == 1
        context = contexts[0]
        assert context is not None
        assert context.enabled
        assert context.local_budget is None
        assert context.prioritized_leaf_paths == ()
        assert context.random_state_snapshot is not None

    def test_csa_local_search_rng_hint_is_keyed_by_proposal_id(self) -> None:
        space = TupleSpace(
            IntegerSpace(0, 5),
            IntegerSpace(0, 5),
        )
        optimizer = CSAOptimizer.from_space_defaults(
            space=space,
            bank_capacity=4,
            random_state=0,
        )
        first_proposal = Proposal(candidate=space.normalize((3, 1)), proposal_id="p-1")
        second_proposal = Proposal(candidate=space.normalize((1, 3)), proposal_id="p-2")
        state = optimizer.create_initial_state()

        single_contexts = optimizer.proposal_kernel_hints(state, (first_proposal,))
        reordered_contexts = optimizer.proposal_kernel_hints(
            state,
            (second_proposal, first_proposal),
        )

        assert single_contexts is not None
        assert reordered_contexts is not None
        single_context = single_contexts[0]
        reordered_first_context = reordered_contexts[1]
        reordered_second_context = reordered_contexts[0]
        assert single_context is not None
        assert reordered_first_context is not None
        assert reordered_second_context is not None
        assert (
            reordered_first_context.random_state_snapshot
            == single_context.random_state_snapshot
        )
        assert (
            reordered_second_context.random_state_snapshot
            != single_context.random_state_snapshot
        )


class ConstantIntegerSampler(CandidateSampler[int]):
    """Test sampler that always returns one declared integer candidate."""

    value: int

    def __init__(self, value: int) -> None:
        self.value = value

    @override
    def sample(self, random_state: np.random.RandomState) -> int:
        _ = random_state
        return self.value


class AbsoluteDistance(DiversityMetric[int]):
    """Simple integer diversity metric for override tests."""

    @override
    def distance(self, left: int, right: int) -> float:
        return float(abs(left - right))


class IdentityMutation(VariationOperator[int]):
    """Unary test operator used to override the derived perturbation schedule."""

    @property
    @override
    def arity(self) -> int:
        return 1

    @override
    def apply(
        self,
        parents: Sequence[int],
        random_state: np.random.RandomState,
    ) -> int:
        _ = random_state
        return parents[0]
