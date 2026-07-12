"""Tests for CSA proposal-adaptation ontology and reducer state."""

from collections.abc import Sequence
from typing import TypeVar, overload

import numpy as np
import pytest
from typing_extensions import override

from tests.numeric_support import approx_equal
from variopt import (
    CandidateRefinement,
    EvaluationOutcome,
    IntegerSpace,
    Observation,
    Proposal,
    TupleSpace,
)
from variopt.algorithms.population.csa import CSAProposalPolicy
from variopt.algorithms.population.csa.banking.update.transition import (
    CSABankTransition,
    CSABankTransitionDisposition,
    CSABankTransitionRoute,
)
from variopt.algorithms.population.csa.generation.perturbation import (
    CSAPerturbationSpec,
)
from variopt.algorithms.population.csa.generation.proposal.evidence import (
    CSAProposalCredit,
    CSAProposalEvaluation,
    CSAProposalLeafCredit,
    CSAProposalOutcomeEvidence,
    derive_proposal_credits,
)
from variopt.algorithms.population.csa.generation.proposal.logic import (
    collect_proposal_outcome_evidence,
    consume_refresh_proposal_provenance,
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
    NonAdaptiveProposalAttribution,
    PlannedProposalAttribution,
    ProposalAttribution,
    ProposalFamilyStat,
    ProposalLeafStat,
)
from variopt.kernel import ProposalLocalSearchContext
from variopt.operators import VariationOperator
from variopt.spaces import LeafPath

CandidateT = TypeVar("CandidateT")


def proposal_outcome_evidence(
    state: CSAProposalState,
    observation: Observation[CandidateT],
    *,
    refinement_changed_leaf_paths: tuple[LeafPath, ...] | None = None,
    evaluation_count: int = 1,
) -> tuple[tuple[CSAProposalOutcomeEvidence[CandidateT], ...], CSAProposalState]:
    """Join one test evaluation to a conclusive local bank replacement."""
    return collect_proposal_outcome_evidence(
        state,
        (
            CSAProposalEvaluation(
                observation=observation,
                evaluation_count=evaluation_count,
                refinement_changed_leaf_paths=refinement_changed_leaf_paths,
            ),
        ),
        (
            CSABankTransition(
                proposal_id=observation.proposal.proposal_id or "",
                route="local",
                disposition="replaced",
                target_index=0,
                survived_batch=True,
            ),
        ),
    )


class ExplodingOutcomeEvidenceSequence(
    Sequence[CSAProposalOutcomeEvidence[int]],
):
    """Sequence that fails if disabled proposal code inspects evidence."""

    @overload
    def __getitem__(self, index: int) -> CSAProposalOutcomeEvidence[int]: ...

    @overload
    def __getitem__(
        self,
        index: slice,
    ) -> Sequence[CSAProposalOutcomeEvidence[int]]: ...

    @override
    def __getitem__(
        self,
        index: int | slice,
    ) -> CSAProposalOutcomeEvidence[int] | Sequence[CSAProposalOutcomeEvidence[int]]:
        _ = index
        raise AssertionError("outcome evidence should not be materialized")

    @override
    def __len__(self) -> int:
        raise AssertionError("outcome evidence should not be measured")


class ExplodingProposalEvaluationSequence(Sequence[CSAProposalEvaluation[int]]):
    """Sequence that fails if disabled collection inspects evaluations."""

    @overload
    def __getitem__(self, index: int) -> CSAProposalEvaluation[int]: ...

    @overload
    def __getitem__(
        self,
        index: slice,
    ) -> Sequence[CSAProposalEvaluation[int]]: ...

    @override
    def __getitem__(
        self,
        index: int | slice,
    ) -> CSAProposalEvaluation[int] | Sequence[CSAProposalEvaluation[int]]:
        _ = index
        raise AssertionError("proposal evaluations should not be materialized")

    @override
    def __len__(self) -> int:
        raise AssertionError("proposal evaluations should not be measured")


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

    @pytest.mark.parametrize(
        ("evaluation_count", "expected_error"),
        [(True, TypeError), (-1, ValueError)],
    )
    def test_proposal_evaluation_rejects_invalid_logical_cost(
        self,
        evaluation_count: int,
        expected_error: type[Exception],
    ) -> None:
        observation = Observation(
            proposal=Proposal(candidate=3, proposal_id="p-1"),
            candidate=3,
            value=3.0,
            score=3.0,
        )

        with pytest.raises(expected_error, match="evaluation_count"):
            _ = CSAProposalEvaluation(
                observation=observation,
                evaluation_count=evaluation_count,
            )

    @pytest.mark.parametrize(
        ("survived_batch", "evaluation_count", "expected_credit"),
        [
            (False, 1, 0.0),
            (True, 0, 1.0),
            (True, 1, 1.0),
            (True, 4, 0.25),
        ],
    )
    def test_proposal_credit_uses_final_survival_and_logical_cost(
        self,
        survived_batch: bool,
        evaluation_count: int,
        expected_credit: float,
    ) -> None:
        evidence = self._outcome_evidence(
            proposal_id="p-1",
            source_score=10.0,
            observed_score=3.0,
            evaluation_count=evaluation_count,
            survived_batch=survived_batch,
        )

        credit = CSAProposalCredit(outcome_evidence=evidence)

        assert approx_equal(credit.pipeline_credit, expected_credit)

    def test_proposal_credit_rejects_admission_without_final_survival(self) -> None:
        evidence = self._outcome_evidence(
            proposal_id="p-1",
            source_score=10.0,
            observed_score=3.0,
            survived_batch=False,
            disposition="replaced",
        )

        assert CSAProposalCredit(outcome_evidence=evidence).pipeline_credit == 0.0

    def test_proposal_credit_accepts_surviving_growth_append(self) -> None:
        evidence = self._outcome_evidence(
            proposal_id="p-1",
            source_score=10.0,
            observed_score=3.0,
            route="growth",
            disposition="appended",
        )

        assert CSAProposalCredit(outcome_evidence=evidence).pipeline_credit == 1.0

    def test_proposal_credit_remains_bounded_for_large_logical_cost(self) -> None:
        evidence = self._outcome_evidence(
            proposal_id="p-1",
            source_score=10.0,
            observed_score=3.0,
            evaluation_count=10**12,
        )

        credit = CSAProposalCredit(outcome_evidence=evidence).pipeline_credit

        assert 0.0 < credit <= 1.0
        assert approx_equal(credit, 1e-12)

    def test_proposal_credit_does_not_overflow_for_arbitrary_size_cost(self) -> None:
        evidence = self._outcome_evidence(
            proposal_id="p-1",
            source_score=10.0,
            observed_score=3.0,
            evaluation_count=10**400,
        )

        credit = CSAProposalCredit(outcome_evidence=evidence).pipeline_credit

        assert credit == 0.0

    def test_proposal_credit_is_positive_affine_invariant(self) -> None:
        original = self._outcome_evidence(
            proposal_id="p-1",
            source_score=10.0,
            observed_score=3.0,
            evaluation_count=2,
        )
        transformed = self._outcome_evidence(
            proposal_id="p-1",
            source_score=1007.0,
            observed_score=307.0,
            evaluation_count=2,
        )

        assert (
            CSAProposalCredit(outcome_evidence=original).pipeline_credit
            == CSAProposalCredit(outcome_evidence=transformed).pipeline_credit
        )

    def test_proposal_credit_conserves_multi_stage_leaf_credit(self) -> None:
        evidence = self._outcome_evidence(
            proposal_id="p-1",
            source_score=10.0,
            observed_score=3.0,
            mutated_leaf_paths=(("x",), ("x",), ("y",)),
        )
        credit = CSAProposalCredit(outcome_evidence=evidence)

        leaf_credits = credit.leaf_association_credits(
            local_displacement_leaf_paths=(("y",), ("z",), ("z",)),
        )

        assert tuple((item.source, item.path) for item in leaf_credits) == (
            ("mutation", ("x",)),
            ("mutation", ("y",)),
            ("local_displacement", ("y",)),
            ("local_displacement", ("z",)),
        )
        assert all(approx_equal(item.credit, 0.25) for item in leaf_credits)
        assert approx_equal(
            sum(item.credit for item in leaf_credits),
            credit.pipeline_credit,
        )

    def test_proposal_credit_without_leaf_associations_emits_none(self) -> None:
        evidence = self._outcome_evidence(
            proposal_id="p-1",
            source_score=10.0,
            observed_score=3.0,
        )

        leaf_credits = CSAProposalCredit(
            outcome_evidence=evidence,
        ).leaf_association_credits()

        assert leaf_credits == ()

    def test_rejected_proposal_emits_zero_leaf_association_credit(self) -> None:
        evidence = self._outcome_evidence(
            proposal_id="p-1",
            source_score=10.0,
            observed_score=3.0,
            survived_batch=False,
            mutated_leaf_paths=(("x",),),
        )

        leaf_credits = CSAProposalCredit(
            outcome_evidence=evidence,
        ).leaf_association_credits()

        assert len(leaf_credits) == 1
        assert leaf_credits[0].credit == 0.0

    @pytest.mark.parametrize("invalid_credit", [True, float("nan"), float("inf")])
    def test_leaf_credit_rejects_noncanonical_values(
        self,
        invalid_credit: float,
    ) -> None:
        with pytest.raises((TypeError, ValueError), match="credit"):
            _ = CSAProposalLeafCredit(
                source="mutation",
                path=("x",),
                credit=invalid_credit,
            )

    def test_derive_proposal_credits_uses_canonical_proposal_order(self) -> None:
        later = self._outcome_evidence(
            proposal_id="p-2",
            source_score=10.0,
            observed_score=3.0,
        )
        earlier = self._outcome_evidence(
            proposal_id="p-1",
            source_score=10.0,
            observed_score=3.0,
        )

        credits = derive_proposal_credits((later, earlier))

        assert tuple(credit.proposal_id for credit in credits) == ("p-1", "p-2")

    def test_derive_proposal_credits_rejects_duplicate_proposal_ids(self) -> None:
        evidence = self._outcome_evidence(
            proposal_id="p-1",
            source_score=10.0,
            observed_score=3.0,
        )

        with pytest.raises(ValueError, match="distinct proposal ids"):
            _ = derive_proposal_credits((evidence, evidence))

    @pytest.mark.parametrize(
        (
            "route",
            "disposition",
            "target_index",
            "survived_batch",
            "expected_error",
        ),
        [
            ("local", "rejected", 0, False, ValueError),
            ("local", "rejected", None, True, ValueError),
            ("local", "replaced", None, True, TypeError),
            ("local", "replaced", -1, True, ValueError),
            ("initial", "replaced", 0, True, ValueError),
            ("local", "appended", 0, True, ValueError),
        ],
    )
    def test_transition_evidence_rejects_invalid_structural_combinations(
        self,
        route: CSABankTransitionRoute,
        disposition: CSABankTransitionDisposition,
        target_index: int | None,
        survived_batch: bool,
        expected_error: type[Exception],
    ) -> None:
        with pytest.raises(expected_error):
            _ = CSABankTransition(
                proposal_id="p-1",
                route=route,
                disposition=disposition,
                target_index=target_index,
                survived_batch=survived_batch,
            )

    @staticmethod
    def _outcome_evidence(
        *,
        proposal_id: str,
        source_score: float,
        observed_score: float,
        evaluation_count: int = 1,
        survived_batch: bool = True,
        mutated_leaf_paths: tuple[LeafPath, ...] = (),
        route: CSABankTransitionRoute = "local",
        disposition: CSABankTransitionDisposition | None = None,
    ) -> CSAProposalOutcomeEvidence[int]:
        canonical_disposition = disposition
        if canonical_disposition is None:
            canonical_disposition = "replaced" if survived_batch else "rejected"
        return CSAProposalOutcomeEvidence(
            attribution=ProposalAttribution(
                proposal_id=proposal_id,
                source_score=source_score,
                mutated_leaf_paths=mutated_leaf_paths,
            ),
            evaluation=CSAProposalEvaluation(
                observation=Observation(
                    proposal=Proposal(candidate=1, proposal_id=proposal_id),
                    candidate=1,
                    value=observed_score,
                    score=observed_score,
                ),
                evaluation_count=evaluation_count,
            ),
            bank_transition=CSABankTransition(
                proposal_id=proposal_id,
                route=route,
                disposition=canonical_disposition,
                target_index=(0 if canonical_disposition != "rejected" else None),
                survived_batch=survived_batch,
            ),
        )

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

        evidence, state = proposal_outcome_evidence(
            state,
            observation,
            evaluation_count=7,
        )
        assert evidence[0].evaluation.evaluation_count == 7
        assert evidence[0].bank_transition.disposition == "replaced"
        assert evidence[0].bank_transition.survived_batch
        next_state = update_proposal_state(state, evidence)

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

    def test_outcome_join_consumes_explicit_non_adaptive_provenance(self) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))
        state = record_proposal_attribution(
            state,
            NonAdaptiveProposalAttribution(
                proposal_id="p-1",
                reason="regular",
            ),
        )
        observation = Observation(
            proposal=Proposal(candidate=3, proposal_id="p-1"),
            candidate=3,
            value=3.0,
            score=3.0,
        )

        evidence, next_state = proposal_outcome_evidence(state, observation)

        assert evidence == ()
        assert next_state.pending_attributions == ()

    def test_outcome_join_rejects_duplicate_provenance_consumption(self) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))
        state = record_proposal_attribution(
            state,
            ProposalAttribution(proposal_id="p-1", source_score=10.0),
        )
        observation = Observation(
            proposal=Proposal(candidate=3, proposal_id="p-1"),
            candidate=3,
            value=3.0,
            score=3.0,
        )
        _, consumed_state = proposal_outcome_evidence(state, observation)

        with pytest.raises(ValueError, match="no pending adaptation provenance"):
            _ = proposal_outcome_evidence(consumed_state, observation)

    def test_outcome_evidence_rejects_transition_proposal_mismatch(self) -> None:
        attribution = ProposalAttribution(proposal_id="p-1", source_score=10.0)
        observation = Observation(
            proposal=Proposal(candidate=3, proposal_id="p-1"),
            candidate=3,
            value=3.0,
            score=3.0,
        )

        with pytest.raises(ValueError, match="bank transition must align"):
            _ = CSAProposalOutcomeEvidence(
                attribution=attribution,
                evaluation=CSAProposalEvaluation.from_observation(observation),
                bank_transition=CSABankTransition(
                    proposal_id="p-2",
                    route="local",
                    disposition="replaced",
                    target_index=0,
                    survived_batch=True,
                ),
            )

    def test_outcome_join_rejects_reordered_batch_transitions_atomically(self) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))
        for proposal_id in ("p-1", "p-2"):
            state = record_proposal_attribution(
                state,
                ProposalAttribution(proposal_id=proposal_id, source_score=10.0),
            )
        evaluations = tuple(
            CSAProposalEvaluation.from_observation(
                Observation(
                    proposal=Proposal(candidate=index, proposal_id=proposal_id),
                    candidate=index,
                    value=float(index),
                    score=float(index),
                ),
            )
            for index, proposal_id in enumerate(("p-1", "p-2"), start=1)
        )
        transitions = tuple(
            CSABankTransition(
                proposal_id=proposal_id,
                route="local",
                disposition="replaced",
                target_index=index,
                survived_batch=True,
            )
            for index, proposal_id in enumerate(("p-2", "p-1"))
        )

        with pytest.raises(ValueError, match="bank transition must align"):
            _ = collect_proposal_outcome_evidence(state, evaluations, transitions)

        assert tuple(
            provenance.proposal_id for provenance in state.pending_attributions
        ) == ("p-1", "p-2")

    def test_outcome_join_rejects_duplicate_success_in_one_batch(self) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))
        state = record_proposal_attribution(
            state,
            ProposalAttribution(proposal_id="p-1", source_score=10.0),
        )
        evaluation = CSAProposalEvaluation.from_observation(
            Observation(
                proposal=Proposal(candidate=1, proposal_id="p-1"),
                candidate=1,
                value=1.0,
                score=1.0,
            ),
        )
        transition = CSABankTransition(
            proposal_id="p-1",
            route="local",
            disposition="replaced",
            target_index=0,
            survived_batch=True,
        )

        with pytest.raises(ValueError, match="proposal ids must be distinct"):
            _ = collect_proposal_outcome_evidence(
                state,
                (evaluation, evaluation),
                (transition, transition),
            )

    def test_outcome_join_handles_mixed_provenance_without_false_credit(self) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))
        state = record_proposal_attribution(
            state,
            NonAdaptiveProposalAttribution(proposal_id="p-1", reason="regular"),
        )
        state = record_proposal_attribution(
            state,
            ProposalAttribution(proposal_id="p-2", source_score=10.0),
        )
        evaluations = tuple(
            CSAProposalEvaluation.from_observation(
                Observation(
                    proposal=Proposal(candidate=index, proposal_id=proposal_id),
                    candidate=index,
                    value=float(index),
                    score=float(index),
                ),
            )
            for index, proposal_id in enumerate(("p-1", "p-2"), start=1)
        )
        transitions = tuple(
            CSABankTransition(
                proposal_id=proposal_id,
                route="local",
                disposition="replaced",
                target_index=index,
                survived_batch=True,
            )
            for index, proposal_id in enumerate(("p-1", "p-2"))
        )

        evidence, next_state = collect_proposal_outcome_evidence(
            state,
            evaluations,
            transitions,
        )

        assert tuple(item.attribution.proposal_id for item in evidence) == ("p-2",)
        assert next_state.pending_attributions == ()

    def test_outcome_join_preserves_zero_logical_evaluation_cost(self) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))
        state = record_proposal_attribution(
            state,
            ProposalAttribution(proposal_id="p-1", source_score=10.0),
        )
        observation = Observation(
            proposal=Proposal(candidate=1, proposal_id="p-1"),
            candidate=1,
            value=1.0,
            score=1.0,
        )

        evidence, _ = proposal_outcome_evidence(
            state,
            observation,
            evaluation_count=0,
        )

        assert evidence[0].evaluation.evaluation_count == 0

    def test_disabled_outcome_join_does_not_materialize_evaluations(self) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=False))

        evidence, next_state = collect_proposal_outcome_evidence(
            state,
            ExplodingProposalEvaluationSequence(),
            (),
        )

        assert evidence == ()
        assert next_state is state

    def test_refresh_rejects_non_refresh_provenance(self) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))
        state = record_proposal_attribution(
            state,
            NonAdaptiveProposalAttribution(proposal_id="p-1", reason="regular"),
        )
        evaluation = CSAProposalEvaluation.from_observation(
            Observation(
                proposal=Proposal(candidate=1, proposal_id="p-1"),
                candidate=1,
                value=1.0,
                score=1.0,
            ),
        )

        with pytest.raises(ValueError, match="refresh_sample provenance"):
            _ = consume_refresh_proposal_provenance(state, (evaluation,))

    def test_outcome_join_rejects_observation_without_registered_provenance(
        self,
    ) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))
        observation = Observation(
            proposal=Proposal(candidate=4, proposal_id="p-2"),
            candidate=4,
            value=4.0,
            score=4.0,
        )

        with pytest.raises(ValueError, match="no pending adaptation provenance"):
            _ = proposal_outcome_evidence(state, observation)

    def test_update_proposal_state_records_local_displacement_stats_separately(
        self,
    ) -> None:
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
        source_candidate: tuple[str, str] = ("before-x", "before-y")
        refined_candidate: tuple[str, str] = ("after-x", "after-y")
        proposal: Proposal[tuple[str, str]] = Proposal(
            candidate=source_candidate,
            proposal_id="p-1",
        )
        observation: Observation[tuple[str, str]] = Observation(
            proposal=proposal,
            candidate=refined_candidate,
            value=3.0,
            score=3.0,
        )

        evidence, state = proposal_outcome_evidence(state, observation)
        next_state = update_proposal_state(
            state,
            evidence,
            infer_local_displacement_leaf_paths=lambda _before, _after: (("y",),),
        )

        assert len(next_state.leaf_stats) == 1
        assert next_state.leaf_stats[0].path == ("x",)
        assert len(next_state.local_displacement_leaf_stats) == 1
        assert next_state.local_displacement_leaf_stats[0].path == ("y",)
        assert (
            next_state.local_displacement_leaf_stats[0].discounted_score_credit == 7.0
        )

    def test_update_proposal_state_prefers_explicit_refinement_paths(self) -> None:
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

        evidence, state = proposal_outcome_evidence(
            state,
            observation,
            refinement_changed_leaf_paths=(("x",),),
        )
        next_state = update_proposal_state(
            state,
            evidence,
            infer_local_displacement_leaf_paths=lambda _before, _after: (("y",),),
        )

        assert len(next_state.local_displacement_leaf_stats) == 1
        assert next_state.local_displacement_leaf_stats[0].path == ("x",)

    def test_update_proposal_state_records_explicit_numeric_refinement_paths(
        self,
    ) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))
        state = record_proposal_attribution(
            state,
            ProposalAttribution(
                proposal_id="p-1",
                source_score=10.0,
                proposal_family_key="mutation:0",
                mutated_leaf_paths=((0,),),
            ),
        )
        source_candidate: tuple[int, int] = (1, 2)
        refined_candidate: tuple[int, int] = (3, 2)
        observation: Observation[tuple[int, int]] = Observation(
            proposal=Proposal(candidate=source_candidate, proposal_id="p-1"),
            candidate=refined_candidate,
            value=3.0,
            score=3.0,
        )

        def raise_if_called(
            _before: tuple[int, int],
            _after: tuple[int, int],
        ) -> tuple[LeafPath, ...]:
            raise AssertionError("inference should not run")

        evidence, state = proposal_outcome_evidence(
            state,
            observation,
            refinement_changed_leaf_paths=((0,),),
        )
        next_state = update_proposal_state(
            state,
            evidence,
            infer_local_displacement_leaf_paths=raise_if_called,
        )

        assert len(next_state.local_displacement_leaf_stats) == 1
        assert next_state.local_displacement_leaf_stats[0].path == (0,)

    def test_update_proposal_state_records_explicit_mixed_refinement_paths(
        self,
    ) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))
        state = record_proposal_attribution(
            state,
            ProposalAttribution(
                proposal_id="p-1",
                source_score=10.0,
                proposal_family_key="mutation:0",
                mutated_leaf_paths=((0,), ("color",)),
            ),
        )
        source_candidate: tuple[int, str, float] = (1, "red", 0.5)
        refined_candidate: tuple[int, str, float] = (2, "blue", 0.5)
        observation: Observation[tuple[int, str, float]] = Observation(
            proposal=Proposal(candidate=source_candidate, proposal_id="p-1"),
            candidate=refined_candidate,
            value=3.0,
            score=3.0,
        )

        def raise_if_called(
            _before: tuple[int, str, float],
            _after: tuple[int, str, float],
        ) -> tuple[LeafPath, ...]:
            raise AssertionError("inference should not run")

        evidence, state = proposal_outcome_evidence(
            state,
            observation,
            refinement_changed_leaf_paths=((0,), ("color",)),
        )
        next_state = update_proposal_state(
            state,
            evidence,
            infer_local_displacement_leaf_paths=raise_if_called,
        )

        assert tuple(
            stat.path for stat in next_state.local_displacement_leaf_stats
        ) == ((0,), ("color",))

    def test_update_proposal_state_falls_back_when_refinement_paths_are_absent(
        self,
    ) -> None:
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
        source_candidate: tuple[str, str] = ("before-x", "before-y")
        refined_candidate: tuple[str, str] = ("after-x", "after-y")
        observation: Observation[tuple[str, str]] = Observation(
            proposal=Proposal(candidate=source_candidate, proposal_id="p-1"),
            candidate=refined_candidate,
            value=3.0,
            score=3.0,
        )

        evidence, state = proposal_outcome_evidence(state, observation)
        next_state = update_proposal_state(
            state,
            evidence,
            infer_local_displacement_leaf_paths=lambda _before, _after: (("y",),),
        )

        assert len(next_state.local_displacement_leaf_stats) == 1
        assert next_state.local_displacement_leaf_stats[0].path == ("y",)

    def test_update_proposal_state_treats_empty_explicit_refinement_paths_as_noop(
        self,
    ) -> None:
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
        observation = Observation(
            proposal=Proposal(candidate=("same-x", "same-y"), proposal_id="p-1"),
            candidate=("same-x", "same-y"),
            value=3.0,
            score=3.0,
        )

        evidence, state = proposal_outcome_evidence(
            state,
            observation,
            refinement_changed_leaf_paths=(),
        )
        next_state = update_proposal_state(
            state,
            evidence,
            infer_local_displacement_leaf_paths=self._raise_if_called,
        )

        assert next_state.local_displacement_leaf_stats == ()

    def test_update_proposal_state_empty_explicit_paths_do_not_fall_back_after_change(
        self,
    ) -> None:
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
        source_candidate: tuple[str, ...] = ("before",)
        refined_candidate: tuple[str, ...] = ("after",)
        proposal: Proposal[tuple[str, ...]] = Proposal(
            candidate=source_candidate,
            proposal_id="p-1",
        )
        observation: Observation[tuple[str, ...]] = Observation(
            proposal=proposal,
            candidate=refined_candidate,
            value=3.0,
            score=3.0,
        )

        def raise_if_called(
            _before: tuple[str, ...],
            _after: tuple[str, ...],
        ) -> tuple[LeafPath, ...]:
            raise AssertionError("inference should not run")

        evidence, state = proposal_outcome_evidence(
            state,
            observation,
            refinement_changed_leaf_paths=(),
        )
        next_state = update_proposal_state(
            state,
            evidence,
            infer_local_displacement_leaf_paths=raise_if_called,
        )

        assert next_state.local_displacement_leaf_stats == ()
        assert next_state.pending_attributions == ()

    def test_outcome_join_rejects_misaligned_bank_transitions(
        self,
    ) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))
        observation = Observation(
            proposal=Proposal(candidate=3, proposal_id="p-1"),
            candidate=3,
            value=3.0,
            score=3.0,
        )

        with pytest.raises(ValueError, match="align one-to-one"):
            _ = collect_proposal_outcome_evidence(
                state,
                (CSAProposalEvaluation.from_observation(observation),),
                (),
            )

    def test_update_proposal_state_does_not_materialize_disabled_evidence(
        self,
    ) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=False))

        next_state = update_proposal_state(
            state,
            ExplodingOutcomeEvidenceSequence(),
            infer_local_displacement_leaf_paths=self._raise_if_called,
        )

        assert next_state == state

    def test_update_proposal_state_accepts_empty_evidence_batch(
        self,
    ) -> None:
        state = CSAProposalState.from_policy(CSAProposalPolicy(enabled=True))

        next_state = update_proposal_state(
            state,
            (),
            infer_local_displacement_leaf_paths=self._raise_if_called,
        )

        assert next_state == state

    def test_update_proposal_state_accepts_paths_extracted_from_refinement(
        self,
    ) -> None:
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
        source_candidate: tuple[str, str] = ("before-x", "before-y")
        refined_candidate: tuple[str, str] = ("after-x", "after-y")
        observation: Observation[tuple[str, str]] = Observation(
            proposal=Proposal(candidate=source_candidate, proposal_id="p-1"),
            candidate=refined_candidate,
            value=3.0,
            score=3.0,
        )
        refinement: CandidateRefinement[tuple[str, str]] = CandidateRefinement(
            source_candidate=source_candidate,
            refined_candidate=refined_candidate,
            changed_leaf_paths=(("y",),),
        )

        outcome = EvaluationOutcome(
            observation=observation,
            refinement=refinement,
        )
        outcome_refinement = outcome.refinement
        assert outcome_refinement is not None

        evidence, state = proposal_outcome_evidence(
            state,
            observation,
            refinement_changed_leaf_paths=outcome_refinement.changed_leaf_paths,
        )
        next_state = update_proposal_state(state, evidence)

        assert len(next_state.local_displacement_leaf_stats) == 1
        assert next_state.local_displacement_leaf_stats[0].path == ("y",)

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

    def test_proposal_local_search_context_returns_none_without_policy_signal(
        self,
    ) -> None:
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
            local_budget=2, prioritized_leaf_paths=(("z",), ("y",), ("x",))
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

    def test_infer_structured_local_displacement_leaf_paths_returns_changed_paths(
        self,
    ) -> None:
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
        assert attribution.generator_kind == "mutation"

    def test_proposal_attribution_preserves_passthrough_generator_kind(self) -> None:
        attribution = ProposalAttribution.from_planned(
            proposal_id="p-1",
            attribution=PlannedProposalAttribution(
                source_score=12.0,
                generator_kind="passthrough",
            ),
        )

        assert attribution.generator_kind == "passthrough"

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

    @staticmethod
    def _raise_if_called(
        _before: tuple[str, str] | int,
        _after: tuple[str, str] | int,
    ) -> tuple[tuple[str, ...], ...]:
        raise AssertionError("inference should not run")


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
