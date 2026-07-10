"""Scale-invariant CSA significant-update contracts."""

from collections.abc import Sequence
from math import inf, nan

import pytest

from variopt.algorithms.population.csa.banking.bank import Bank, BankEntry
from variopt.algorithms.population.csa.banking.update.policy import (
    CSABankUpdatePolicy,
)
from variopt.algorithms.population.csa.banking.update.result import (
    significant_update_indices,
)


def bank(
    scores: Sequence[float],
    *,
    candidates: Sequence[int] | None = None,
) -> Bank[int]:
    """Return an index-aligned bank fixture with explicit score values."""
    candidate_values = tuple(range(len(scores))) if candidates is None else candidates
    return Bank(
        capacity=max(1, len(scores)),
        entries=tuple(
            BankEntry(candidate=candidate, value=score, proposal_id=f"p-{index}")
            for index, (candidate, score) in enumerate(
                zip(candidate_values, scores, strict=True),
            )
        ),
    )


def transformed_scores(
    scores: Sequence[float],
    *,
    scale: float,
    offset: float,
) -> tuple[float, ...]:
    """Return one positive affine transform of the supplied scores."""
    return tuple(scale * score + offset for score in scores)


@pytest.mark.parametrize(
    ("scale", "offset"),
    [
        (1.0, 0.0),
        (100.0, 7.0),
        (1.0, 1_000_000.0),
        (1e200, 0.0),
        (1e-200, 0.0),
    ],
)
def test_significant_membership_is_positive_affine_invariant(
    scale: float,
    offset: float,
) -> None:
    previous_scores = (0.0, 10.0, 20.0)
    next_scores = (1.0, 8.0, 20.0)

    updated_indices = significant_update_indices(
        previous_bank=bank(
            transformed_scores(previous_scores, scale=scale, offset=offset),
        ),
        next_bank=bank(
            transformed_scores(next_scores, scale=scale, offset=offset),
            candidates=(3, 4, 2),
        ),
        minimum_significant_score_gap_ratio=0.075,
    )

    assert updated_indices == frozenset({1})


def test_threshold_equality_is_not_significant() -> None:
    updated_indices = significant_update_indices(
        previous_bank=bank((0.0, 10.0)),
        next_bank=bank((2.0, 10.0), candidates=(2, 1)),
        minimum_significant_score_gap_ratio=0.2,
    )

    assert updated_indices == frozenset()


def test_zero_ratio_marks_every_nonzero_score_change_as_significant() -> None:
    updated_indices = significant_update_indices(
        previous_bank=bank((0.0, 10.0)),
        next_bank=bank((1.0, 10.0), candidates=(2, 1)),
        minimum_significant_score_gap_ratio=0.0,
    )

    assert updated_indices == frozenset({0})


@pytest.mark.parametrize(
    ("previous_scores", "next_scores"),
    [
        ((5.0,), (4.0,)),
        ((5.0, 5.0), (4.0, 4.0)),
    ],
)
def test_zero_spread_nonzero_changes_are_significant(
    previous_scores: tuple[float, ...],
    next_scores: tuple[float, ...],
) -> None:
    updated_indices = significant_update_indices(
        previous_bank=bank(previous_scores),
        next_bank=bank(
            next_scores,
            candidates=tuple(range(10, 10 + len(next_scores))),
        ),
        minimum_significant_score_gap_ratio=100.0,
    )

    assert updated_indices == frozenset(range(len(next_scores)))


def test_zero_spread_identity_only_change_is_not_significant() -> None:
    updated_indices = significant_update_indices(
        previous_bank=bank((5.0,)),
        next_bank=bank((5.0,), candidates=(10,)),
        minimum_significant_score_gap_ratio=0.0,
    )

    assert updated_indices == frozenset()


@pytest.mark.parametrize(
    ("previous_scores", "next_scores"),
    [
        ((0.0, 1e-323), (5e-324, 1e-323)),
        ((-1e308, 1e308), (0.0, 1e308)),
    ],
)
def test_extreme_finite_scales_do_not_underflow_or_overflow(
    previous_scores: tuple[float, float],
    next_scores: tuple[float, float],
) -> None:
    updated_indices = significant_update_indices(
        previous_bank=bank(previous_scores),
        next_bank=bank(next_scores, candidates=(2, 1)),
        minimum_significant_score_gap_ratio=0.4,
    )

    assert updated_indices == frozenset({0})


def test_additions_are_significant_and_removals_have_no_next_index() -> None:
    appended_indices = significant_update_indices(
        previous_bank=bank((1.0, 2.0)),
        next_bank=bank((1.0, 2.0, 3.0)),
        minimum_significant_score_gap_ratio=10.0,
    )
    removed_indices = significant_update_indices(
        previous_bank=bank((1.0, 2.0, 3.0)),
        next_bank=bank((1.0, 2.0)),
        minimum_significant_score_gap_ratio=0.0,
    )

    assert appended_indices == frozenset({2})
    assert removed_indices == frozenset()


def test_reordered_entries_use_aligned_score_changes() -> None:
    updated_indices = significant_update_indices(
        previous_bank=bank((0.0, 10.0), candidates=(0, 1)),
        next_bank=bank((10.0, 0.0), candidates=(1, 0)),
        minimum_significant_score_gap_ratio=0.5,
    )

    assert updated_indices == frozenset({0, 1})


@pytest.mark.parametrize(
    ("ratio", "expected_error"),
    [
        (True, TypeError),
        (-0.1, ValueError),
        (inf, ValueError),
        (-inf, ValueError),
        (nan, ValueError),
    ],
)
def test_policy_rejects_invalid_significance_ratios(
    ratio: float,
    expected_error: type[Exception],
) -> None:
    with pytest.raises(expected_error, match="minimum_significant_score_gap_ratio"):
        _ = CSABankUpdatePolicy(minimum_significant_score_gap_ratio=ratio)
