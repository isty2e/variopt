"""Tests for execution-budget ownership primitives."""

import pickle

import pytest

from variopt.execution import (
    EvaluationBudget,
    EvaluationBudgetExhausted,
    EvaluationReservationBatch,
)


def test_evaluation_reservation_settles_and_refunds_unused_capacity() -> None:
    budget = EvaluationBudget(20)
    reservation = EvaluationReservationBatch(budget=budget, limits=(2, 5, 3))

    assert budget.remaining == 10
    assert reservation.reserved_evaluation_count == 10
    assert reservation.is_finalized is False

    assert reservation.settle((2, 1, 0)) == 3

    assert budget.remaining == 17
    assert reservation.is_finalized is True


def test_evaluation_reservation_forfeits_all_reserved_capacity() -> None:
    budget = EvaluationBudget(10)
    reservation = EvaluationReservationBatch(budget=budget, limits=(2, 3))

    reservation.forfeit()

    assert budget.remaining == 5
    assert reservation.is_finalized is True


@pytest.mark.parametrize(
    ("limits", "exception_type"),
    [
        ((), ValueError),
        ((0,), ValueError),
        ((-1,), ValueError),
        ((True,), TypeError),
        ((1.0,), TypeError),
    ],
)
def test_invalid_evaluation_reservation_does_not_consume_budget(
    limits: tuple[int, ...],
    exception_type: type[Exception],
) -> None:
    budget = EvaluationBudget(10)

    with pytest.raises(exception_type):
        _ = EvaluationReservationBatch(budget=budget, limits=limits)

    assert budget.remaining == 10


def test_oversized_evaluation_reservation_is_atomic() -> None:
    budget = EvaluationBudget(4)

    with pytest.raises(EvaluationBudgetExhausted, match="budget exhausted"):
        _ = EvaluationReservationBatch(budget=budget, limits=(2, 3))

    assert budget.remaining == 4


@pytest.mark.parametrize(
    ("consumed_counts", "exception_type"),
    [
        ((1,), ValueError),
        ((1, 2, 3), ValueError),
        ((-1, 0), ValueError),
        ((True, 0), TypeError),
        ((1.0, 0), TypeError),
        ((3, 0), ValueError),
    ],
)
def test_malformed_evaluation_settlement_fails_closed(
    consumed_counts: tuple[int, ...],
    exception_type: type[Exception],
) -> None:
    budget = EvaluationBudget(10)
    reservation = EvaluationReservationBatch(budget=budget, limits=(2, 2))

    with pytest.raises(exception_type):
        _ = reservation.settle(consumed_counts)

    assert budget.remaining == 6
    assert reservation.is_finalized is True
    with pytest.raises(RuntimeError, match="already finalized"):
        reservation.forfeit()


def test_evaluation_reservation_rejects_duplicate_finalization() -> None:
    budget = EvaluationBudget(10)
    settled = EvaluationReservationBatch(budget=budget, limits=(2,))
    forfeited = EvaluationReservationBatch(budget=budget, limits=(2,))

    assert settled.settle((1,)) == 1
    forfeited.forfeit()

    with pytest.raises(RuntimeError, match="already finalized"):
        _ = settled.settle((1,))
    with pytest.raises(RuntimeError, match="already finalized"):
        forfeited.forfeit()

    assert budget.remaining == 7


def test_evaluation_reservation_is_not_picklable() -> None:
    reservation = EvaluationReservationBatch(
        budget=EvaluationBudget(2),
        limits=(1,),
    )

    with pytest.raises(
        TypeError,
        match="coordinator-local and cannot be pickled",
    ):
        _ = pickle.dumps(reservation)
