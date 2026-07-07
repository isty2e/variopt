"""Weighted operator-family wrapper for CSA perturbation schedules."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

import numpy as np
from typing_extensions import override

from variopt.generic_runtime import (
    FrozenGenericSlotsCompat,
    install_frozen_generic_slots_pickle,
)

from .....operators import VariationOperator
from .....typevars import CandidateT
from .validation import require_parent_count, sample_weighted_index


@dataclass(frozen=True, slots=True)
class MixtureVariation(
    FrozenGenericSlotsCompat,
    VariationOperator[CandidateT],
    Generic[CandidateT],
):
    """Choose one operator from a weighted family and apply it.

    Parameters
    ----------
    operators : Sequence[VariationOperator[CandidateT]]
        Operator family to sample from.
    weights : Sequence[float] | None, optional
        Optional non-negative weights aligned with ``operators``. When omitted,
        the family is sampled uniformly.
    """

    operators: tuple[VariationOperator[CandidateT], ...]
    weights: tuple[float, ...]

    def __init__(
        self,
        operators: Sequence[VariationOperator[CandidateT]],
        weights: Sequence[float] | None = None,
    ) -> None:
        if len(operators) == 0:
            msg = "operators must not be empty"
            raise ValueError(msg)

        operator_tuple = tuple(operators)
        if weights is None:
            weight_tuple = tuple(1.0 for _ in operator_tuple)
        else:
            weight_tuple = tuple(float(weight) for weight in weights)

        if len(weight_tuple) != len(operator_tuple):
            msg = "weights must match operators in length"
            raise ValueError(msg)

        if any(weight < 0.0 for weight in weight_tuple):
            msg = "weights must be non-negative"
            raise ValueError(msg)

        if all(weight == 0.0 for weight in weight_tuple):
            msg = "weights must contain at least one positive value"
            raise ValueError(msg)

        object.__setattr__(self, "operators", operator_tuple)
        object.__setattr__(self, "weights", weight_tuple)

    @property
    @override
    def arity(self) -> int:
        """Return the maximum arity required by the operator family."""
        return max(operator.arity for operator in self.operators)

    @override
    def apply(
        self,
        parents: Sequence[CandidateT],
        random_state: np.random.RandomState,
    ) -> CandidateT:
        """Sample one operator and produce its child.

        Parameters
        ----------
        parents : Sequence[CandidateT]
            Parent tuple large enough for the highest-arity operator in the
            family.
        random_state : np.random.RandomState
            Random state used for weighted operator selection.

        Returns
        -------
        CandidateT
            Child candidate produced by the sampled operator.
        """
        require_parent_count(parents, arity=self.arity)
        operator_index = sample_weighted_index(self.weights, random_state)
        operator = self.operators[operator_index]
        return operator.apply(parents[: operator.arity], random_state)


install_frozen_generic_slots_pickle(MixtureVariation)
