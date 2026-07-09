"""Tests for CSA-private covariance-aware numeric subspace adaptation."""

from typing import cast

import numpy as np

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


class CSAProposalCovarianceTests:
    """Regression tests for CSA-private covariance proposal adaptation."""

    def test_update_proposal_state_records_numeric_covariance_displacement(
        self,
    ) -> None:
        policy = CSAProposalPolicy(
            enabled=True,
            score_decay=1.0,
            numeric_covariance_strength=1.0,
            numeric_covariance_min_observations=1,
        )
        state = CSAProposalState.from_policy(policy)
        state = record_proposal_attribution(
            state,
            ProposalAttribution(
                proposal_id="p-1",
                source_score=10.0,
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
            (CSAProposalEvaluation.from_observation(observation),),
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
        assert covariance_stat.effective_mean(
            current_update_index=next_state.update_index,
            score_decay=1.0,
        ) == (1.0, -1.0)

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
                score_decay=1.0,
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
