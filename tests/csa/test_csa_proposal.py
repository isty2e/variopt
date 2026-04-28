"""Tests for CSA proposal-adaptation ontology and reducer state."""

from collections.abc import Sequence

import numpy as np
import pytest
from typing_extensions import override

from tests.numeric_support import approx_equal
from variopt import (
    IntegerSpace,
    Observation,
    Proposal,
    TupleSpace,
)
from variopt.algorithms.population.csa import CSAProposalPolicy
from variopt.algorithms.population.csa.generation.perturbation import (
    CSAPerturbationSpec,
)
from variopt.algorithms.population.csa.generation.proposal.logic import (
    infer_structured_local_displacement_leaf_paths,
    mutation_family_key,
    mutation_family_weights,
    mutation_leaf_weights,
    plan_mutated_leaf_paths,
    planned_mutation_attribution,
    proposal_local_search_context,
    record_proposal_attribution,
    sample_mutation_family_indices,
    update_proposal_state,
)
from variopt.algorithms.population.csa.generation.proposal.state import (
    CSAProposalState,
    PlannedProposalAttribution,
    ProposalAttribution,
    ProposalFamilyStat,
    ProposalLeafStat,
)
from variopt.kernel import ProposalLocalSearchContext
from variopt.operators import VariationOperator


class CSAProposalStateTests:
    """Regression tests for proposal-side adaptive-memory ontology."""

    def test_disabled_policy_keeps_registered_attributions_out_of_state(self) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=False))

        next_state = record_proposal_attribution(
            state,
            ProposalAttribution(
                proposal_id="p-1",
                source_score=10.0,
                mutated_leaf_paths=(("x",),),
            ),
        )

        assert next_state.pending_attributions == ()

    def test_update_proposal_state_consumes_matching_attribution(self) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))
        state = record_proposal_attribution(
            state,
            ProposalAttribution(
                proposal_id="p-1",
                source_score=10.0,
                proposal_family_key="mutation:0",
                mutated_leaf_paths=(("x",), ("y",)),
            ),
        )
        observation = Observation(
            proposal=Proposal(candidate=3, proposal_id="p-1"),
            candidate=3,
            value=3.0,
            score=3.0,
        )

        next_state = update_proposal_state(state, (observation,))

        assert next_state.pending_attributions == ()
        assert len(next_state.family_stats) == 1
        assert next_state.family_stats[0].family_key == "mutation:0"
        assert next_state.family_stats[0].observation_count == 1
        assert next_state.family_stats[0].discounted_score_credit == 7.0
        assert len(next_state.leaf_stats) == 2
        assert next_state.leaf_stats[0].path == ("x",)
        assert next_state.leaf_stats[1].path == ("y",)
        assert next_state.leaf_stats[0].observation_count == 1
        assert next_state.leaf_stats[1].observation_count == 1
        assert next_state.leaf_stats[0].discounted_score_credit == 7.0
        assert next_state.leaf_stats[1].discounted_score_credit == 7.0
        assert next_state.local_displacement_leaf_stats == ()

    def test_update_proposal_state_ignores_observations_without_registered_attribution(self) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))
        observation = Observation(
            proposal=Proposal(candidate=4, proposal_id="p-2"),
            candidate=4,
            value=4.0,
            score=4.0,
        )

        next_state = update_proposal_state(state, (observation,))

        assert next_state == state

    def test_update_proposal_state_records_local_displacement_stats_separately(self) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))
        state = record_proposal_attribution(
            state,
            ProposalAttribution(
                proposal_id="p-1",
                source_score=10.0,
                proposal_family_key="mutation:0",
                mutated_leaf_paths=(("x",),),
            ),
        )
        proposal: Proposal[tuple[str, str]] = Proposal(
            candidate=("before-x", "before-y"),
            proposal_id="p-1",
        )
        observation: Observation[tuple[str, str]] = Observation(
            proposal=proposal,
            candidate=("after-x", "after-y"),
            value=3.0,
            score=3.0,
        )

        next_state = update_proposal_state(
            state,
            (observation,),
            infer_local_displacement_leaf_paths=lambda _before, _after: (("y",),),
        )

        assert len(next_state.leaf_stats) == 1
        assert next_state.leaf_stats[0].path == ("x",)
        assert len(next_state.local_displacement_leaf_stats) == 1
        assert next_state.local_displacement_leaf_stats[0].path == ("y",)
        assert next_state.local_displacement_leaf_stats[0].discounted_score_credit == 7.0

    def test_plan_mutated_leaf_paths_prefers_recently_successful_leaf(self) -> None:
        policy = CSAProposalPolicy(
            enabled=True,
            leaf_bias_strength=10.0,
            score_decay=1.0,
        )
        state = CSAProposalState(
            policy=policy,
            leaf_stats=(
                ProposalLeafStat(
                    path=("x",),
                    observation_count=2,
                    discounted_score_credit=5.0,
                ),
                ProposalLeafStat(
                    path=("y",),
                    observation_count=2,
                    discounted_score_credit=0.0,
                ),
            ),
        )

        weights = mutation_leaf_weights(
            state=state,
            leaf_paths=(("x",), ("y",)),
        )
        selected_paths = plan_mutated_leaf_paths(
            state=state,
            leaf_paths=(("x",), ("y",)),
            exchange_count=1,
            random_state=np.random.RandomState(0),
        )

        assert weights[0] > weights[1]
        assert selected_paths == (("x",),)

    def test_plan_mutated_leaf_paths_can_prefer_local_displacement_signal(self) -> None:
        policy = CSAProposalPolicy(
            enabled=True,
            leaf_bias_strength=0.0,
            local_displacement_leaf_bias_strength=10.0,
            score_decay=1.0,
        )
        state = CSAProposalState(
            policy=policy,
            local_displacement_leaf_stats=(
                ProposalLeafStat(
                    path=("x",),
                    observation_count=2,
                    discounted_score_credit=5.0,
                ),
                ProposalLeafStat(
                    path=("y",),
                    observation_count=2,
                    discounted_score_credit=0.0,
                ),
            ),
        )

        weights = mutation_leaf_weights(
            state=state,
            leaf_paths=(("x",), ("y",)),
        )
        selected_paths = plan_mutated_leaf_paths(
            state=state,
            leaf_paths=(("x",), ("y",)),
            exchange_count=1,
            random_state=np.random.RandomState(0),
        )

        assert weights[0] > weights[1]
        assert selected_paths == (("x",),)

    def test_proposal_local_search_context_returns_none_without_policy_signal(self) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))

        context = proposal_local_search_context(
            state=state,
            leaf_paths=(("x",), ("y",)),
        )

        assert context is None

    def test_proposal_local_search_context_prioritizes_mutated_then_successful_paths(
        self,
    ) -> None:
        state = CSAProposalState(
            policy=CSAProposalPolicy(
                enabled=True,
                leaf_bias_strength=10.0,
                score_decay=1.0,
            ),
            leaf_stats=(
                ProposalLeafStat(
                    path=("y",),
                    observation_count=2,
                    discounted_score_credit=5.0,
                ),
                ProposalLeafStat(
                    path=("x",),
                    observation_count=2,
                    discounted_score_credit=0.0,
                ),
            ),
        )

        context = proposal_local_search_context(
            state=state,
            leaf_paths=(("x",), ("y",), ("z",)),
            attribution=ProposalAttribution(
                proposal_id="p-1",
                source_score=10.0,
                mutated_leaf_paths=(("z",),),
            ),
        )

        assert context == ProposalLocalSearchContext(
                local_budget=2,
                prioritized_leaf_paths=(("z",), ("y",), ("x",))
            )

    def test_proposal_local_search_context_can_disable_repeatedly_failing_paths(
        self,
    ) -> None:
        state = CSAProposalState(
            policy=CSAProposalPolicy(
                enabled=True,
                local_search_disable_failure_streak=2,
            ),
            leaf_stats=(
                ProposalLeafStat(
                    path=("x",),
                    observation_count=3,
                    discounted_score_credit=0.0,
                    recent_failure_streak=2,
                ),
            ),
        )

        context = proposal_local_search_context(
            state=state,
            leaf_paths=(("x",), ("y",)),
            attribution=ProposalAttribution(
                proposal_id="p-1",
                source_score=10.0,
                mutated_leaf_paths=(("x",),),
            ),
        )

        assert context == ProposalLocalSearchContext(
                enabled=False,
                prioritized_leaf_paths=(("x",), ("y",)),
            )

    def test_proposal_local_search_context_shapes_budget_from_supportive_paths(
        self,
    ) -> None:
        state = CSAProposalState(
            policy=CSAProposalPolicy(
                enabled=True,
                local_search_base_budget=2,
                local_search_max_budget=8,
                score_decay=1.0,
            ),
            leaf_stats=(
                ProposalLeafStat(
                    path=("x",),
                    observation_count=2,
                    discounted_score_credit=3.0,
                ),
                ProposalLeafStat(
                    path=("y",),
                    observation_count=2,
                    discounted_score_credit=1.0,
                ),
            ),
        )

        context = proposal_local_search_context(
            state=state,
            leaf_paths=(("x",), ("y",), ("z",)),
            attribution=ProposalAttribution(
                proposal_id="p-1",
                source_score=10.0,
                mutated_leaf_paths=(("x",), ("y",)),
            ),
        )

        assert context is not None
        assert context.enabled
        assert context.local_budget == 5
        assert context.prioritized_leaf_paths[:2] == (("x",), ("y",))

    def test_proposal_local_search_context_demotes_recently_failed_paths_in_cooldown(
        self,
    ) -> None:
        state = CSAProposalState(
            policy=CSAProposalPolicy(
                enabled=True,
                leaf_bias_strength=10.0,
                score_decay=1.0,
                local_search_base_budget=2,
                local_search_max_budget=8,
                local_search_failure_cooldown_updates=3,
            ),
            leaf_stats=(
                ProposalLeafStat(
                    path=("x",),
                    observation_count=3,
                    discounted_score_credit=5.0,
                    last_update_index=10,
                    recent_failure_streak=1,
                ),
                ProposalLeafStat(
                    path=("y",),
                    observation_count=3,
                    discounted_score_credit=3.0,
                    last_update_index=6,
                    recent_failure_streak=1,
                ),
                ProposalLeafStat(
                    path=("z",),
                    observation_count=2,
                    discounted_score_credit=1.0,
                    last_update_index=2,
                ),
            ),
            update_index=10,
        )

        context = proposal_local_search_context(
            state=state,
            leaf_paths=(("x",), ("y",), ("z",)),
            attribution=ProposalAttribution(
                proposal_id="p-1",
                source_score=10.0,
                mutated_leaf_paths=(("x",), ("y",)),
            ),
        )

        assert context == ProposalLocalSearchContext(
                enabled=True,
                local_budget=3,
                prioritized_leaf_paths=(("y",), ("x",), ("z",)),
            )

    def test_proposal_local_search_context_can_gate_mutation_during_cooldown(
        self,
    ) -> None:
        state = CSAProposalState(
            policy=CSAProposalPolicy(
                enabled=True,
                local_search_disable_failure_streak=3,
                local_search_failure_cooldown_updates=2,
            ),
            leaf_stats=(
                ProposalLeafStat(
                    path=("x",),
                    observation_count=2,
                    discounted_score_credit=0.0,
                    last_update_index=4,
                    recent_failure_streak=1,
                ),
            ),
            update_index=5,
        )

        context = proposal_local_search_context(
            state=state,
            leaf_paths=(("x",), ("y",)),
            attribution=ProposalAttribution(
                proposal_id="p-1",
                source_score=10.0,
                mutated_leaf_paths=(("x",),),
            ),
        )

        assert context == ProposalLocalSearchContext(
                enabled=False,
                prioritized_leaf_paths=(("x",), ("y",)),
            )

    def test_infer_structured_local_displacement_leaf_paths_returns_changed_paths(self) -> None:
        space = TupleSpace(
            IntegerSpace(0, 9),
            IntegerSpace(0, 9),
        )

        changed_paths = infer_structured_local_displacement_leaf_paths(
            space=space,
            proposal_candidate=(1, 2),
            observed_candidate=(1, 5),
        )

        assert changed_paths == ((1,),)

    def test_record_leaf_score_improvement_applies_lazy_decay(self) -> None:
        state = CSAProposalState(
            policy=CSAProposalPolicy(enabled=True, score_decay=0.5),
            leaf_stats=(
                ProposalLeafStat(
                    path=("x",),
                    observation_count=1,
                    discounted_score_credit=8.0,
                    last_update_index=0,
                ),
            ),
            update_index=2,
        )

        next_state = state.record_score_improvement(
            family_key=None,
            leaf_paths=(("x",),),
            score_improvement=2.0,
        )

        assert next_state.update_index == 3
        assert next_state.leaf_stats[0].observation_count == 2
        assert approx_equal(next_state.leaf_stats[0].discounted_score_credit, 3.0)
        assert next_state.leaf_stats[0].recent_failure_streak == 0

    def test_record_leaf_score_improvement_tracks_recent_failure_streak(self) -> None:
        state = CSAProposalState(
            policy=CSAProposalPolicy(enabled=True, score_decay=0.5),
            leaf_stats=(
                ProposalLeafStat(
                    path=("x",),
                    observation_count=1,
                    discounted_score_credit=8.0,
                    last_update_index=0,
                    recent_failure_streak=1,
                ),
            ),
            update_index=2,
        )

        next_state = state.record_score_improvement(
            family_key=None,
            leaf_paths=(("x",),),
            score_improvement=0.0,
        )

        assert next_state.update_index == 3
        assert next_state.leaf_stats[0].observation_count == 2
        assert approx_equal(next_state.leaf_stats[0].discounted_score_credit, 1.0)
        assert next_state.leaf_stats[0].recent_failure_streak == 2

    def test_proposal_attribution_can_bind_planned_record_to_proposal_id(self) -> None:
        planned_attribution = PlannedProposalAttribution(
            source_score=12.0,
            proposal_family_key="mutation:0",
            mutated_leaf_paths=(("x",),),
        )

        attribution = ProposalAttribution.from_planned(
            proposal_id="p-1",
            attribution=planned_attribution,
        )

        assert attribution.proposal_id == "p-1"
        assert attribution.source_score == 12.0
        assert attribution.proposal_family_key == "mutation:0"
        assert attribution.mutated_leaf_paths == (("x",),)

    def test_planned_mutation_attribution_normalizes_paths(self) -> None:
        attribution = planned_mutation_attribution(
            source_score=5.0,
            mutated_leaf_paths=[("x",)],
        )

        assert attribution == PlannedProposalAttribution(
                source_score=5.0,
                mutated_leaf_paths=(("x",),),
            )

    def test_mutation_family_weights_prefer_successful_family(self) -> None:
        family = (
            CSAPerturbationSpec(IdentityMutation(), count=1),
            CSAPerturbationSpec(IdentityMutation(), count=1),
        )
        state = CSAProposalState(
            policy=CSAProposalPolicy(
                enabled=True,
                family_bias_strength=5.0,
                score_decay=1.0,
            ),
            family_stats=(
                ProposalFamilyStat(
                    family_key="mutation:0",
                    observation_count=1,
                    discounted_score_credit=3.0,
                ),
            ),
        )

        weights = mutation_family_weights(state=state, family=family)

        assert weights[0] > weights[1]

    def test_sample_mutation_family_indices_prefers_successful_family(self) -> None:
        family = (
            CSAPerturbationSpec(IdentityMutation(), count=1),
            CSAPerturbationSpec(IdentityMutation(), count=1),
        )
        state = CSAProposalState(
            policy=CSAProposalPolicy(
                enabled=True,
                family_bias_strength=10.0,
                score_decay=1.0,
            ),
            family_stats=(
                ProposalFamilyStat(
                    family_key="mutation:0",
                    observation_count=1,
                    discounted_score_credit=4.0,
                ),
            ),
        )

        sampled_indices = sample_mutation_family_indices(
            state=state,
            family=family,
            random_state=np.random.RandomState(0),
        )

        assert sampled_indices == (0, 0)

    def test_mutation_family_key_rejects_negative_index(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _ = mutation_family_key(-1)

    def test_mutation_family_key_uses_canonical_mutation_prefix(self) -> None:
        assert mutation_family_key(2) == "mutation:2"

    def test_proposal_policy_rejects_negative_local_displacement_leaf_bias_strength(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _ = CSAProposalPolicy(local_displacement_leaf_bias_strength=-1.0)

    def test_proposal_policy_rejects_negative_numeric_covariance_strength(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _ = CSAProposalPolicy(numeric_covariance_strength=-1.0)

    def test_proposal_policy_rejects_non_positive_numeric_covariance_min_observations(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="positive"):
            _ = CSAProposalPolicy(numeric_covariance_min_observations=0)

    def test_proposal_policy_rejects_negative_numeric_covariance_ridge(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _ = CSAProposalPolicy(numeric_covariance_ridge=-1.0)

    def test_proposal_policy_rejects_non_positive_local_search_budget(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            _ = CSAProposalPolicy(local_search_base_budget=0)

    def test_proposal_policy_rejects_local_search_budget_inversion(self) -> None:
        with pytest.raises(ValueError, match="at least"):
            _ = CSAProposalPolicy(
                local_search_base_budget=3,
                local_search_max_budget=2,
            )

    def test_proposal_policy_rejects_negative_failure_cooldown_updates(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            _ = CSAProposalPolicy(local_search_failure_cooldown_updates=-1)


class IdentityMutation(VariationOperator[int]):
    """Unary test operator used to build valid mutation-family schedules."""

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
