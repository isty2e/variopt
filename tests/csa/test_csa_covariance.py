"""Tests for CSA-private covariance-aware numeric subspace adaptation."""

from typing import cast

import numpy as np

from tests.numeric_support import approx_equal
from variopt import IntegerSpace, Observation, Proposal, RealSpace, TupleSpace
from variopt.algorithms.population.csa import CSAProposalPolicy
from variopt.algorithms.population.csa.banking.update.transition import (
    CSABankTransition,
)
from variopt.algorithms.population.csa.generation.proposal.covariance import (
    sample_covariance_guided_candidate,
)
from variopt.algorithms.population.csa.generation.proposal.evidence import (
    CSAProposalEvaluation,
)
from variopt.algorithms.population.csa.generation.proposal.logic import (
    collect_proposal_outcome_evidence,
    record_proposal_attribution,
    update_proposal_state,
)
from variopt.algorithms.population.csa.generation.proposal.state import (
    CSAProposalState,
    NumericSubspaceAttribution,
    NumericSubspaceDisplacement,
    ProposalAttribution,
    ProposalNumericSubspaceCovarianceStat,
)
from variopt.spaces.projections import compile_homogeneous_numeric_subspace


class NumericSubspaceDescriptorTests:
    """Regression tests for homogeneous numeric subspace descriptors."""

    def test_compile_descriptor_roundtrips_homogeneous_real_tuple(self) -> None:
        space = TupleSpace(
            RealSpace(0.0, 10.0),
            RealSpace(0.0, 10.0),
        )
        descriptor = compile_homogeneous_numeric_subspace(space)
        assert descriptor is not None
        candidate = space.normalize((1.0, 2.0))

        coordinates = descriptor.coordinates_from_candidate(candidate)
        projected_candidate = descriptor.candidate_from_coordinates(
            candidate,
            (3.0, 4.0),
        )

        assert coordinates == (1.0, 2.0)
        assert projected_candidate == (3.0, 4.0)

    def test_compile_descriptor_rejects_mixed_numeric_leaf_types(self) -> None:
        space = TupleSpace(
            RealSpace(0.0, 10.0),
            IntegerSpace(0, 10),
        )

        descriptor = compile_homogeneous_numeric_subspace(space)

        assert descriptor is None

    def test_numeric_subspace_attribution_rejects_duplicate_paths(self) -> None:
        with np.testing.assert_raises_regex(ValueError, "distinct leaf paths"):
            _ = NumericSubspaceAttribution(
                leaf_paths=((0,), (0,)),
                source_coordinates=(0.0, 1.0),
            )

    def test_numeric_subspace_attribution_rejects_non_finite_coordinates(
        self,
    ) -> None:
        with np.testing.assert_raises_regex(ValueError, "coordinates must be finite"):
            _ = NumericSubspaceAttribution(
                leaf_paths=((0,),),
                source_coordinates=(float("inf"),),
            )

    def test_numeric_subspace_attribution_rejects_bool_coordinates(self) -> None:
        with np.testing.assert_raises_regex(TypeError, "coordinates must be numeric"):
            _ = NumericSubspaceAttribution(
                leaf_paths=((0,),),
                source_coordinates=(True,),
            )

    def test_numeric_subspace_displacement_rejects_duplicate_paths(self) -> None:
        with np.testing.assert_raises_regex(ValueError, "distinct leaf paths"):
            _ = NumericSubspaceDisplacement(
                leaf_paths=((0,), (0,)),
                displacement_coordinates=(0.0, 1.0),
            )

    def test_numeric_subspace_displacement_rejects_non_finite_coordinates(
        self,
    ) -> None:
        with np.testing.assert_raises_regex(ValueError, "coordinates must be finite"):
            _ = NumericSubspaceDisplacement(
                leaf_paths=((0,),),
                displacement_coordinates=(float("nan"),),
            )

    def test_numeric_subspace_displacement_rejects_bool_coordinates(self) -> None:
        with np.testing.assert_raises_regex(TypeError, "coordinates must be numeric"):
            _ = NumericSubspaceDisplacement(
                leaf_paths=((0,),),
                displacement_coordinates=(False,),
            )

    def test_numeric_subspace_displacement_rejects_outer_product_overflow(
        self,
    ) -> None:
        with np.testing.assert_raises_regex(ValueError, "finite outer products"):
            _ = NumericSubspaceDisplacement(
                leaf_paths=((0,),),
                displacement_coordinates=(1e155,),
            )


class CSAProposalCovarianceTests:
    """Regression tests for CSA-private covariance proposal adaptation."""

    def test_covariance_stat_rejects_noncanonical_integer_fields(self) -> None:
        with np.testing.assert_raises_regex(TypeError, "observation_count"):
            _ = ProposalNumericSubspaceCovarianceStat(
                leaf_paths=((0,),),
                observation_count=True,
            )
        with np.testing.assert_raises_regex(TypeError, "last_update_index"):
            _ = ProposalNumericSubspaceCovarianceStat(
                leaf_paths=((0,),),
                last_update_index=False,
            )

    def test_covariance_stat_rejects_weight_over_observation_count(self) -> None:
        with np.testing.assert_raises_regex(ValueError, "bounded by observation_count"):
            _ = ProposalNumericSubspaceCovarianceStat(
                leaf_paths=((0,),),
                observation_count=1,
                discounted_weight=1.1,
            )

    def test_covariance_stat_rejects_observations_without_weight(self) -> None:
        with np.testing.assert_raises_regex(ValueError, "must have positive"):
            _ = ProposalNumericSubspaceCovarianceStat(
                leaf_paths=((0,),),
                observation_count=1,
            )

    def test_covariance_stat_rejects_weight_without_complete_moments(self) -> None:
        with np.testing.assert_raises_regex(ValueError, "complete moment"):
            _ = ProposalNumericSubspaceCovarianceStat(
                leaf_paths=((0,),),
                observation_count=1,
                discounted_weight=1.0,
            )

    def test_covariance_stat_rejects_zero_weight_nonzero_moments(self) -> None:
        with np.testing.assert_raises_regex(ValueError, "zero moment"):
            _ = ProposalNumericSubspaceCovarianceStat(
                leaf_paths=((0,),),
                discounted_displacement_sum=(1.0,),
                discounted_outer_product_sum=((1.0,),),
            )

    def test_covariance_stat_zero_weight_has_zero_moments(self) -> None:
        covariance_stat = ProposalNumericSubspaceCovarianceStat(
            leaf_paths=((0,), (1,)),
        )

        assert covariance_stat.effective_mean(
            current_update_index=3,
            adaptation_decay=0.5,
        ) == (0.0, 0.0)
        assert covariance_stat.effective_covariance(
            current_update_index=3,
            adaptation_decay=0.5,
        ) == ((0.0, 0.0), (0.0, 0.0))

    def test_covariance_lazy_decay_precedes_later_observation(self) -> None:
        covariance_stat = ProposalNumericSubspaceCovarianceStat(
            leaf_paths=((0,),),
            discounted_displacement_sum=(0.0,),
            discounted_outer_product_sum=((0.0,),),
        ).record_successful_displacement(
            NumericSubspaceDisplacement(
                leaf_paths=((0,),),
                displacement_coordinates=(1.0,),
            ),
            survival_efficiency=1.0,
            current_update_index=1,
            adaptation_decay=0.5,
        )

        covariance_stat = covariance_stat.record_successful_displacement(
            NumericSubspaceDisplacement(
                leaf_paths=((0,),),
                displacement_coordinates=(3.0,),
            ),
            survival_efficiency=1.0,
            current_update_index=3,
            adaptation_decay=0.5,
        )

        assert approx_equal(covariance_stat.discounted_weight, 1.25)
        assert approx_equal(
            covariance_stat.effective_mean(
                current_update_index=3,
                adaptation_decay=0.5,
            )[0],
            2.6,
        )
        assert approx_equal(
            covariance_stat.effective_covariance(
                current_update_index=3,
                adaptation_decay=0.5,
            )[0][0],
            0.64,
        )

    def test_covariance_is_invariant_to_uniform_survival_efficiency_scale(self) -> None:
        covariance_stats: list[ProposalNumericSubspaceCovarianceStat] = []
        for survival_efficiency in (1.0, 0.1):
            covariance_stat = ProposalNumericSubspaceCovarianceStat(
                leaf_paths=((0,),),
                discounted_displacement_sum=(0.0,),
                discounted_outer_product_sum=((0.0,),),
            )
            for update_index, coordinate in enumerate(
                (1.0, 2.0, 3.0, 4.0),
                start=1,
            ):
                covariance_stat = covariance_stat.record_successful_displacement(
                    NumericSubspaceDisplacement(
                        leaf_paths=((0,),),
                        displacement_coordinates=(coordinate,),
                    ),
                    survival_efficiency=survival_efficiency,
                    current_update_index=update_index,
                    adaptation_decay=1.0,
                )
            covariance_stats.append(covariance_stat)

        covariances = tuple(
            covariance_stat.effective_covariance(
                current_update_index=4,
                adaptation_decay=1.0,
            )
            for covariance_stat in covariance_stats
        )
        assert approx_equal(covariances[0][0][0], covariances[1][0][0])

    def test_covariance_rejects_outer_product_accumulation_overflow(self) -> None:
        safe_coordinate = np.sqrt(np.finfo(np.float64).max)
        displacement = NumericSubspaceDisplacement(
            leaf_paths=((0,),),
            displacement_coordinates=(safe_coordinate,),
        )
        covariance_stat = ProposalNumericSubspaceCovarianceStat(
            leaf_paths=((0,),),
            discounted_displacement_sum=(0.0,),
            discounted_outer_product_sum=((0.0,),),
        ).record_successful_displacement(
            displacement,
            survival_efficiency=1.0,
            current_update_index=1,
            adaptation_decay=1.0,
        )

        with np.testing.assert_raises_regex(ValueError, "moment accumulation"):
            _ = covariance_stat.record_successful_displacement(
                displacement,
                survival_efficiency=1.0,
                current_update_index=2,
                adaptation_decay=1.0,
            )

    def test_update_proposal_state_records_numeric_covariance_displacement(
        self,
    ) -> None:
        policy = CSAProposalPolicy(
            enabled=True,
            adaptation_decay=1.0,
            numeric_covariance_strength=1.0,
            numeric_covariance_min_observations=1,
        )
        state = CSAProposalState.from_policy(policy)
        state = record_proposal_attribution(
            state,
            ProposalAttribution(
                proposal_id="p-1",
                numeric_subspace_attribution=NumericSubspaceAttribution(
                    leaf_paths=((0,), (1,)),
                    source_coordinates=(0.0, 0.0),
                ),
            ),
        )
        observation = Observation(
            proposal=Proposal(candidate=(0.0, 0.0), proposal_id="p-1"),
            candidate=(1.0, -1.0),
            value=3.0,
            score=3.0,
        )

        evidence, state = collect_proposal_outcome_evidence(
            state,
            (CSAProposalEvaluation(observation=observation, evaluation_count=4),),
            (
                CSABankTransition(
                    proposal_id="p-1",
                    route="local",
                    disposition="replaced",
                    target_index=0,
                    survived_batch=True,
                ),
            ),
        )
        next_state = update_proposal_state(
            state,
            evidence,
            infer_numeric_subspace_displacement=lambda attribution, observed_candidate: (
                NumericSubspaceDisplacement(
                    leaf_paths=attribution.numeric_subspace_attribution.leaf_paths,
                    displacement_coordinates=observed_candidate,
                )
                if attribution.numeric_subspace_attribution is not None
                else None
            ),
        )

        assert len(next_state.numeric_covariance_stats) == 1
        covariance_stat = next_state.numeric_covariance_stats[0]
        assert covariance_stat.observation_count == 1
        assert covariance_stat.discounted_weight == 0.25
        assert covariance_stat.effective_mean(
            current_update_index=next_state.update_index,
            adaptation_decay=1.0,
        ) == (1.0, -1.0)

    def test_covariance_moments_weight_displacements_by_logical_cost(self) -> None:
        policy = CSAProposalPolicy(
            enabled=True,
            adaptation_decay=1.0,
            numeric_covariance_strength=1.0,
            numeric_covariance_min_observations=1,
        )
        state = CSAProposalState.from_policy(policy)
        for proposal_id in ("p-1", "p-2"):
            state = record_proposal_attribution(
                state,
                ProposalAttribution(
                    proposal_id=proposal_id,
                    numeric_subspace_attribution=NumericSubspaceAttribution(
                        leaf_paths=((0,),),
                        source_coordinates=(0.0,),
                    ),
                ),
            )
        observations = tuple(
            Observation(
                proposal=Proposal(candidate=(0.0,), proposal_id=proposal_id),
                candidate=(displacement,),
                value=displacement,
                score=displacement,
            )
            for proposal_id, displacement in (("p-1", 1.0), ("p-2", 3.0))
        )
        evaluations = tuple(
            CSAProposalEvaluation(
                observation=observation,
                evaluation_count=evaluation_count,
            )
            for observation, evaluation_count in zip(
                observations,
                (1, 3),
                strict=True,
            )
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
        evidence, state = collect_proposal_outcome_evidence(
            state,
            evaluations,
            transitions,
        )

        next_state = update_proposal_state(
            state,
            evidence,
            infer_numeric_subspace_displacement=(
                lambda attribution, observed_candidate: (
                    NumericSubspaceDisplacement(
                        leaf_paths=attribution.numeric_subspace_attribution.leaf_paths,
                        displacement_coordinates=observed_candidate,
                    )
                    if attribution.numeric_subspace_attribution is not None
                    else None
                )
            ),
        )

        covariance_stat = next_state.numeric_covariance_stats[0]
        assert approx_equal(covariance_stat.discounted_weight, 4.0 / 3.0)
        assert approx_equal(
            covariance_stat.effective_mean(
                current_update_index=next_state.update_index,
                adaptation_decay=1.0,
            )[0],
            1.5,
        )
        assert approx_equal(
            covariance_stat.effective_covariance(
                current_update_index=next_state.update_index,
                adaptation_decay=1.0,
            )[0][0],
            0.75,
        )

    def test_rejected_outcome_does_not_infer_numeric_displacement(self) -> None:
        state = CSAProposalState.from_policy(
            CSAProposalPolicy(
                enabled=True,
                numeric_covariance_strength=1.0,
            ),
        )
        state = record_proposal_attribution(
            state,
            ProposalAttribution(
                proposal_id="p-1",
                numeric_subspace_attribution=NumericSubspaceAttribution(
                    leaf_paths=((0,),),
                    source_coordinates=(0.0,),
                ),
            ),
        )
        observation = Observation(
            proposal=Proposal(candidate=(0.0,), proposal_id="p-1"),
            candidate=(1.0,),
            value=1.0,
            score=1.0,
        )
        evidence, state = collect_proposal_outcome_evidence(
            state,
            (CSAProposalEvaluation.from_observation(observation),),
            (
                CSABankTransition(
                    proposal_id="p-1",
                    route="local",
                    disposition="rejected",
                    target_index=None,
                    survived_batch=False,
                ),
            ),
        )

        def raise_if_called(
            _attribution: ProposalAttribution,
            _candidate: tuple[float, ...],
        ) -> NumericSubspaceDisplacement | None:
            raise AssertionError("rejected outcomes must not infer displacement")

        next_state = update_proposal_state(
            state,
            evidence,
            infer_numeric_subspace_displacement=raise_if_called,
        )

        assert next_state.numeric_covariance_stats == ()

    def test_covariance_guided_candidate_masks_unselected_paths(self) -> None:
        space = TupleSpace(
            RealSpace(0.0, 10.0),
            RealSpace(0.0, 10.0),
        )
        descriptor = compile_homogeneous_numeric_subspace(space)
        assert descriptor is not None
        source_candidate = space.normalize((5.0, 5.0))
        proposal_state = CSAProposalState(
            policy=CSAProposalPolicy(
                enabled=True,
                adaptation_decay=1.0,
                numeric_covariance_strength=1.0,
                numeric_covariance_min_observations=1,
            ),
            numeric_covariance_stats=(
                ProposalNumericSubspaceCovarianceStat(
                    leaf_paths=((0,), (1,)),
                    observation_count=1,
                    discounted_weight=1.0,
                    discounted_displacement_sum=(1.0, 1.0),
                    discounted_outer_product_sum=((1.0, 1.0), (1.0, 1.0)),
                ),
            ),
        )

        sampled = sample_covariance_guided_candidate(
            descriptor=descriptor,
            source_candidate=source_candidate,
            selected_paths=((0,),),
            proposal_state=proposal_state,
            max_coordinate_fraction=0.2,
            random_state=np.random.RandomState(0),
        )

        assert sampled is not None
        candidate, changed_paths = sampled
        typed_candidate = cast(tuple[float, float], candidate)
        assert changed_paths == ((0,),)
        assert typed_candidate[0] != source_candidate[0]
        assert typed_candidate[1] == source_candidate[1]
