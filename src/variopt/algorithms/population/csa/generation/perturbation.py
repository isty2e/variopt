"""Public CSA perturbation-family policy."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....operators import VariationOperator
from .....typevars import CandidateT


@dataclass(frozen=True, slots=True)
class CSAPerturbationSpec(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """One operator/count member of a CSA child-generation family.

    Parameters
    ----------
    operator : VariationOperator[CandidateT]
        Variation operator to apply.
    count : int, default=1
        Number of children to emit from the operator within the family.
    """

    operator: VariationOperator[CandidateT]
    count: int = 1

    def __post_init__(self) -> None:
        """Reject invalid perturbation-family members."""
        if self.count <= 0:
            msg = "count must be positive"
            raise ValueError(msg)

        if self.operator.arity <= 0:
            msg = "operator arity must be positive"
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class CSAPerturbationSchedule(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """CSA-specific child-generation families over one active seed batch.

    Parameters
    ----------
    regular_family : tuple[CSAPerturbationSpec[CandidateT], ...], default=()
        Family used for the regular proposal stage.
    initial_family : tuple[CSAPerturbationSpec[CandidateT], ...], default=()
        Family used for initial seeding stages.
    mutation_family : tuple[CSAPerturbationSpec[CandidateT], ...], default=()
        Unary mutation family used for mutation-only stages.
    shuffle_children : bool, default=True
        Whether emitted children are shuffled before evaluation.
    """

    regular_family: tuple[CSAPerturbationSpec[CandidateT], ...] = ()
    initial_family: tuple[CSAPerturbationSpec[CandidateT], ...] = ()
    mutation_family: tuple[CSAPerturbationSpec[CandidateT], ...] = ()
    shuffle_children: bool = True

    def __post_init__(self) -> None:
        """Reject invalid perturbation schedules."""
        regular_family = tuple(self.regular_family)
        initial_family = tuple(self.initial_family)
        mutation_family = tuple(self.mutation_family)
        object.__setattr__(self, "regular_family", regular_family)
        object.__setattr__(self, "initial_family", initial_family)
        object.__setattr__(self, "mutation_family", mutation_family)

        if not (
            len(regular_family) > 0
            or len(initial_family) > 0
            or len(mutation_family) > 0
        ):
            msg = "at least one perturbation family must be configured"
            raise ValueError(msg)

        self._require_initial_family_arities(initial_family)
        self._require_mutation_family_arities(mutation_family)

    @staticmethod
    def _require_initial_family_arities(
        family: Sequence[CSAPerturbationSpec[CandidateT]],
    ) -> None:
        for spec in family:
            if spec.operator.arity < 2:
                msg = "initial_family operators must have arity at least 2"
                raise ValueError(msg)

    @staticmethod
    def _require_mutation_family_arities(
        family: Sequence[CSAPerturbationSpec[CandidateT]],
    ) -> None:
        for spec in family:
            if spec.operator.arity != 1:
                msg = "mutation_family operators must have arity exactly 1"
                raise ValueError(msg)
