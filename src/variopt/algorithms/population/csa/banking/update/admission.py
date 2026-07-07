"""CSA bank-admission rules."""

from ......artifacts import Observation
from ......diversity import DiversityMetric
from ......typevars import CandidateT
from ..bank import Bank, BankEntry
from ..queries import (
    crowded_indices,
    crowding_aware_scores,
    nearest_entry,
    worst_index,
)
from .policy import CSABankUpdatePolicy


def admit_observation(
    *,
    policy: CSABankUpdatePolicy,
    bank: Bank[CandidateT],
    observation: Observation[CandidateT],
    diversity_metric: DiversityMetric[CandidateT],
    distance_cutoff: float,
) -> Bank[CandidateT]:
    """Apply the configured CSA bank-admission rule to one observation.

    Parameters
    ----------
    policy : CSABankUpdatePolicy
        Configured near/far admission policy.
    bank : Bank[CandidateT]
        Current bank snapshot.
    observation : Observation[CandidateT]
        Observation being considered for admission.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used for near/far decisions and crowding-aware logic.
    distance_cutoff : float
        Distance threshold separating local and far updates.

    Returns
    -------
    Bank[CandidateT]
        Updated bank after applying the configured rule.

    Raises
    ------
    ValueError
        If ``distance_cutoff`` is negative.
    """
    if distance_cutoff < 0.0:
        msg = "distance_cutoff must be non-negative"
        raise ValueError(msg)

    new_entry = BankEntry(
        candidate=observation.candidate,
        value=observation.score,
        proposal_id=observation.proposal.proposal_id,
    )
    if not bank.is_full:
        return Bank(
            capacity=bank.capacity,
            entries=bank.entries + (new_entry,),
        )

    nearest_index, nearest_distance = nearest_entry(
        entries=bank.entries,
        candidate=new_entry.candidate,
        diversity_metric=diversity_metric,
    )
    assert nearest_distance is not None

    if nearest_distance < distance_cutoff:
        if policy.local_update_mode == "disabled":
            return bank

        closest_entry = bank.entries[nearest_index]
        if new_entry.value < closest_entry.value:
            return replace_bank_entry(
                bank=bank,
                index=nearest_index,
                new_entry=new_entry,
            )

        return bank

    far_replacement_candidates: tuple[int, ...] = ()
    if policy.far_update_mode == "crowded_worst":
        far_replacement_candidates = tuple(
            crowded_indices(
                entries=bank.entries,
                diversity_metric=diversity_metric,
                distance_cutoff=distance_cutoff,
            )
        )
        far_worst_index = worst_index(
            bank.entries,
            candidate_indices=(
                far_replacement_candidates if far_replacement_candidates else None
            ),
        )
    elif policy.far_update_mode == "crowding_aware":
        removal_scores = crowding_aware_scores(
            base_scores=tuple(entry.value for entry in bank.entries),
            entries=bank.entries,
            diversity_metric=diversity_metric,
            distance_cutoff=distance_cutoff,
            penalty_ratio=policy.crowding_penalty_ratio,
            niche_quality_policy=policy.niche_quality_policy,
        )
        far_worst_index = max(
            range(len(removal_scores)), key=removal_scores.__getitem__
        )
    else:
        far_worst_index = worst_index(bank.entries)
    if new_entry.value < bank.entries[far_worst_index].value:
        return replace_bank_entry(
            bank=bank,
            index=far_worst_index,
            new_entry=new_entry,
        )

    return bank


def replace_bank_entry(
    *,
    bank: Bank[CandidateT],
    index: int,
    new_entry: BankEntry[CandidateT],
) -> Bank[CandidateT]:
    """Return a bank with one entry replaced in place.

    Parameters
    ----------
    bank : Bank[CandidateT]
        Bank snapshot to update.
    index : int
        Entry index to replace.
    new_entry : BankEntry[CandidateT]
        Replacement bank entry.

    Returns
    -------
    Bank[CandidateT]
        Updated bank containing ``new_entry`` at ``index``.
    """
    replaced_entries = list(bank.entries)
    replaced_entries[index] = new_entry
    return Bank(
        capacity=bank.capacity,
        entries=tuple(replaced_entries),
    )
