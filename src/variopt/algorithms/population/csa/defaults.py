"""Space-derived default CSA components."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar, overload

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....diversity import StructuredSpaceDiversityMetric
from ....sampling import CandidateSampler, SearchSpaceSampler
from ....spaces import PermutationSpace, StructuredSearchSpace
from ....spaces.types import SpaceCandidateValue
from ..permutation.operators import InversionMutation, OrderCrossover, SwapMutation
from .generation.perturbation import CSAPerturbationSchedule, CSAPerturbationSpec
from .operators import BoundedMutation, RandomResetMutation, UniformCrossover

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT", bound=SpaceCandidateValue)


@dataclass(frozen=True, slots=True)
class CSADefaultComponents(FrozenGenericSlotsCompat, Generic[BoundaryT, CandidateT]):
    """Space-derived default CSA components with explicit override points.

    Parameters
    ----------
    sampler : CandidateSampler[CandidateT]
        Default sampler inferred from the structured search space.
    diversity_metric : StructuredSpaceDiversityMetric[BoundaryT, CandidateT]
        Default diversity metric inferred from the structured search space.
    perturbation_schedule : CSAPerturbationSchedule[CandidateT]
        Default perturbation schedule inferred from the structured search
        space.
    """

    sampler: CandidateSampler[CandidateT]
    diversity_metric: StructuredSpaceDiversityMetric[BoundaryT, CandidateT]
    perturbation_schedule: CSAPerturbationSchedule[CandidateT]


def derive_permutation_csa_schedule(
    *,
    space: PermutationSpace,
    style: Literal["variopt", "joung_2018"],
) -> CSAPerturbationSchedule[tuple[int, ...]]:
    """Derive a permutation-safe CSA perturbation schedule.

    Parameters
    ----------
    space : PermutationSpace
        Permutation space whose semantics constrain the perturbation family.
    style : {"variopt", "joung_2018"}
        Default-style variant used to choose operator counts and amplitudes.

    Returns
    -------
    CSAPerturbationSchedule[tuple[int, ...]]
        Perturbation schedule specialized to permutation candidates.
    """
    if style == "joung_2018":
        return CSAPerturbationSchedule(
            regular_family=(
                CSAPerturbationSpec(
                    OrderCrossover(
                        space=space,
                        max_segment_fraction=0.5,
                    ),
                    count=10,
                ),
            ),
            initial_family=(
                CSAPerturbationSpec(
                    OrderCrossover(
                        space=space,
                        max_segment_fraction=0.2,
                    ),
                    count=10,
                ),
            ),
            mutation_family=(
                CSAPerturbationSpec(
                    InversionMutation(
                        space=space,
                        max_inversion_fraction=0.2,
                    ),
                    count=10,
                ),
            ),
        )

    return CSAPerturbationSchedule(
        regular_family=(
            CSAPerturbationSpec(
                OrderCrossover(space=space),
                count=2,
            ),
        ),
        initial_family=(
            CSAPerturbationSpec(
                OrderCrossover(space=space),
                count=2,
            ),
        ),
        mutation_family=(
            CSAPerturbationSpec(
                InversionMutation(space=space),
                count=2,
            ),
            CSAPerturbationSpec(
                SwapMutation(space=space),
                count=1,
            ),
        ),
    )


def derive_permutation_csa_defaults(
    space: PermutationSpace,
    *,
    style: Literal["variopt", "joung_2018"],
) -> CSADefaultComponents[Sequence[int], tuple[int, ...]]:
    """Derive permutation-safe CSA defaults.

    Parameters
    ----------
    space : PermutationSpace
        Permutation space whose semantics determine the defaults.
    style : {"variopt", "joung_2018"}
        Default-style variant used to choose operator counts and amplitudes.

    Returns
    -------
    CSADefaultComponents[Sequence[int], tuple[int, ...]]
        Default sampler, diversity metric, and perturbation schedule for the
        permutation space.
    """
    return CSADefaultComponents(
        sampler=SearchSpaceSampler(space=space),
        diversity_metric=StructuredSpaceDiversityMetric(space=space),
        perturbation_schedule=derive_permutation_csa_schedule(
            space=space,
            style=style,
        ),
    )


@overload
def derive_csa_defaults(
    space: PermutationSpace,
    *,
    style: Literal["variopt", "joung_2018"] = "variopt",
) -> CSADefaultComponents[Sequence[int], tuple[int, ...]]:
    """Derive CSA defaults for a permutation space.

    Parameters
    ----------
    space : PermutationSpace
        Permutation space whose semantics determine the defaults.
    style : {"variopt", "joung_2018"}, default="variopt"
        Default-style variant used to choose operator counts and amplitudes.

    Returns
    -------
    CSADefaultComponents[Sequence[int], tuple[int, ...]]
        Default sampler, diversity metric, and perturbation schedule for the
        permutation space.
    """
    ...


@overload
def derive_csa_defaults(
    space: StructuredSearchSpace[BoundaryT, CandidateT],
    *,
    style: Literal["variopt", "joung_2018"] = "variopt",
) -> CSADefaultComponents[BoundaryT, CandidateT]:
    """Derive CSA defaults for a generic structured search space.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, CandidateT]
        Structured space whose semantics determine the defaults.
    style : {"variopt", "joung_2018"}, default="variopt"
        Default-style variant used to choose operator counts and amplitudes.

    Returns
    -------
    CSADefaultComponents[BoundaryT, CandidateT]
        Default sampler, diversity metric, and perturbation schedule for the
        structured space.
    """
    ...


def derive_csa_defaults(
    space: StructuredSearchSpace[BoundaryT, CandidateT],
    *,
    style: Literal["variopt", "joung_2018"] = "variopt",
) -> (
    CSADefaultComponents[BoundaryT, CandidateT]
    | CSADefaultComponents[Sequence[int], tuple[int, ...]]
):
    """Derive default CSA components from a structured search space.

    Parameters
    ----------
    space : StructuredSearchSpace[BoundaryT, CandidateT]
        Structured search space whose semantics determine the defaults.
    style : {"variopt", "joung_2018"}, default="variopt"
        Default-style variant used to choose operator counts and amplitudes.

    Returns
    -------
    CSADefaultComponents[BoundaryT, CandidateT] | CSADefaultComponents[Sequence[int], tuple[int, ...]]
        Default sampler, diversity metric, and perturbation schedule for the
        supplied structured space.
    """
    if isinstance(space, PermutationSpace):
        return derive_permutation_csa_defaults(
            space,
            style=style,
        )

    perturbation_schedule = CSAPerturbationSchedule(
        regular_family=(
            CSAPerturbationSpec(
                UniformCrossover(space=space),
                count=2,
            ),
        ),
        initial_family=(
            CSAPerturbationSpec(
                UniformCrossover(space=space),
                count=2,
            ),
        ),
        mutation_family=(
            CSAPerturbationSpec(
                BoundedMutation(space=space),
                count=2,
            ),
            CSAPerturbationSpec(
                RandomResetMutation(space=space),
                count=1,
            ),
        ),
    )

    if style == "joung_2018":
        perturbation_schedule = CSAPerturbationSchedule(
            regular_family=(
                CSAPerturbationSpec(
                    UniformCrossover(
                        space=space,
                        max_exchange_fraction=0.5,
                    ),
                    count=10,
                ),
            ),
            initial_family=(
                CSAPerturbationSpec(
                    UniformCrossover(
                        space=space,
                        max_exchange_fraction=0.2,
                    ),
                    count=10,
                ),
            ),
            mutation_family=(
                CSAPerturbationSpec(
                    BoundedMutation(
                        space=space,
                        max_perturbation_fraction=0.2,
                    ),
                    count=10,
                ),
            ),
        )

    return CSADefaultComponents(
        sampler=SearchSpaceSampler(space=space),
        diversity_metric=StructuredSpaceDiversityMetric(space=space),
        perturbation_schedule=perturbation_schedule,
    )
