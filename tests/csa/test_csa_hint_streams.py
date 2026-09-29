"""Proposal alignment for batch-derived local-search streams."""

from dataclasses import replace

import pytest

from variopt import IntegerSpace, Proposal
from variopt.algorithms.population.csa import (
    CSAOptimizer,
    CSAProfile,
    CSAProposalPolicy,
)
from variopt.randomness import derive_random_state_snapshot


@pytest.mark.parametrize("adaptation_enabled", [False, True])
def test_hint_streams_keep_missing_duplicate_and_reordered_proposal_ids(
    adaptation_enabled: bool,
) -> None:
    optimizer = CSAOptimizer.from_space_defaults(
        space=IntegerSpace(0, 9),
        bank_capacity=4,
        profile=CSAProfile(
            proposal_policy=CSAProposalPolicy(enabled=adaptation_enabled)
        ),
        random_state=7,
    )
    state = optimizer.create_initial_state()
    proposals = (
        Proposal(candidate=1),
        Proposal(candidate=2, proposal_id="p-2"),
        Proposal(candidate=3),
        Proposal(candidate=4, proposal_id="p-1"),
        Proposal(candidate=5, proposal_id="p-2"),
        Proposal(candidate=6),
    )

    hints = optimizer.proposal_kernel_hints(state, proposals)

    assert hints is not None
    assert len(hints) == len(proposals)
    for proposal, hint in zip(proposals, hints, strict=True):
        one_hint = optimizer.proposal_kernel_hints(state, (proposal,))
        assert hint == (None if one_hint is None else one_hint[0])
        if proposal.proposal_id is not None:
            assert hint is not None
            assert hint.random_state_snapshot == derive_random_state_snapshot(
                state.random_state,
                namespace="variopt.csa.local_search",
                keys=(proposal.proposal_id,),
            )
        elif hint is not None:
            assert hint.random_state_snapshot is None
    assert optimizer.proposal_kernel_hints(state, tuple(reversed(proposals))) == tuple(
        reversed(hints)
    )
    assert optimizer.proposal_kernel_hints(state, ()) is None
    assert state == optimizer.create_initial_state()

    restored = optimizer.state_from_dict(optimizer.state_to_dict(state))
    assert optimizer.proposal_kernel_hints(restored, proposals) == hints
    advanced = replace(state, random_state=replace(state.random_state, position=0))
    assert optimizer.proposal_kernel_hints(advanced, proposals) != hints
