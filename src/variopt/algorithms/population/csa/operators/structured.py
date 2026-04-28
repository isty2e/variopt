"""Structured CSA variation-operator wrappers."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeGuard, TypeVar, cast

import numpy as np
from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....operators import VariationOperator
from .....spaces import LeafPath, SearchSpace, StructuredSearchSpace
from .....spaces.structured import require_space_candidate_value
from .....spaces.types import SpaceCandidateValue
from .crossover import uniform_crossover as _uniform_crossover
from .mutation import (
    bounded_mutation as _bounded_mutation,
)
from .mutation import (
    bounded_mutation_on_paths as _bounded_mutation_on_paths,
)
from .mutation import (
    random_reset_mutation as _random_reset_mutation,
)
from .mutation import (
    random_reset_mutation_on_paths as _random_reset_mutation_on_paths,
)
from .validation import (
    require_parent_count,
    require_structured_space,
    validate_fraction,
)

BoundaryT = TypeVar("BoundaryT")
StructuredCandidateT = TypeVar("StructuredCandidateT", bound=SpaceCandidateValue)


class StructuredPathMutationOperator(Protocol):
    """Capability protocol for structured unary operators over explicit leaf paths.

    Notes
    -----
    CSA ask-side planning only needs the operator's structured candidate-space
    view, its path-selection fraction cap, and the ability to apply one
    mutation on an explicit set of leaf paths. It does not need the concrete
    operator class or the exact structured candidate subtype.
    """

    @property
    def structured_candidate_space(
        self,
    ) -> StructuredSearchSpace[object, SpaceCandidateValue]:
        """Return the type-erased structured candidate space."""
        ...

    @property
    def max_selected_path_fraction(self) -> float:
        """Return the maximum fraction of active leaf paths that may be selected."""
        ...

    def apply_space_candidate_on_paths(
        self,
        candidate: SpaceCandidateValue,
        selected_paths: Sequence[LeafPath],
        random_state: np.random.RandomState,
    ) -> SpaceCandidateValue:
        """Return a child by editing explicit paths on a canonical candidate value."""
        ...


class CovarianceGuidedStructuredMutationOperator(
    StructuredPathMutationOperator,
    Protocol,
):
    """Capability protocol for structured path mutation with numeric covariance guidance."""

    @property
    def max_coordinate_fraction(self) -> float:
        """Return the maximum coordinate-space fraction used by covariance guidance."""
        ...


def is_structured_path_mutation_operator(
    operator: object,
) -> TypeGuard[StructuredPathMutationOperator]:
    """Return whether one operator supports explicit structured leaf-path editing."""
    return isinstance(operator, (RandomResetMutation, BoundedMutation))


def is_covariance_guided_structured_mutation_operator(
    operator: object,
) -> TypeGuard[CovarianceGuidedStructuredMutationOperator]:
    """Return whether one operator supports covariance-guided structured mutation."""
    return isinstance(operator, BoundedMutation)


@dataclass(frozen=True, slots=True)
class UniformCrossover(FrozenGenericSlotsCompat,
    VariationOperator[StructuredCandidateT],
    Generic[BoundaryT, StructuredCandidateT],
):
    """Copy a random subset of partner leaves into the primary parent.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, StructuredCandidateT]
        Search space used to validate parents and resulting children.
    max_exchange_fraction : float, default=0.5
        Maximum fraction of active leaves copied from the partner.
    """

    space: SearchSpace[BoundaryT, StructuredCandidateT]
    structured_space: StructuredSearchSpace[BoundaryT, StructuredCandidateT] = field(
        init=False,
        repr=False,
    )
    max_exchange_fraction: float = 0.5

    def __post_init__(self) -> None:
        """Validate the crossover configuration."""
        validate_fraction(
            self.max_exchange_fraction,
            name="max_exchange_fraction",
        )
        object.__setattr__(
            self,
            "structured_space",
            require_structured_space(self.space),
        )

    @property
    @override
    def arity(self) -> int:
        """Return the required number of parent candidates."""
        return 2

    @override
    def apply(
        self,
        parents: Sequence[StructuredCandidateT],
        random_state: np.random.RandomState,
    ) -> StructuredCandidateT:
        """Return a crossover child built from two parent candidates.

        Parameters
        ----------
        parents : Sequence[StructuredCandidateT]
            Parent candidates in ``(primary, partner)`` order.
        random_state : np.random.RandomState
            Random state used for leaf-selection decisions.

        Returns
        -------
        StructuredCandidateT
            Child candidate after structured uniform crossover.

        Raises
        ------
        ValueError
            If ``parents`` does not match the required arity.
        TypeError
            If the configured search space is not structured.
        """
        require_parent_count(parents, arity=self.arity)
        child = _uniform_crossover(
            space=self.structured_space,
            primary_parent=parents[0],
            partner_parent=parents[1],
            max_exchange_fraction=self.max_exchange_fraction,
            random_state=random_state,
        )
        self.space.validate(child)
        return child


@dataclass(frozen=True, slots=True)
class RandomResetMutation(FrozenGenericSlotsCompat,
    VariationOperator[StructuredCandidateT],
    Generic[BoundaryT, StructuredCandidateT],
):
    """Resample a random subset of candidate leaves from the declared space.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, StructuredCandidateT]
        Search space used to validate parents and resulting children.
    max_exchange_fraction : float, default=0.2
        Maximum fraction of active leaves eligible for random reset.
    """

    space: SearchSpace[BoundaryT, StructuredCandidateT]
    structured_space: StructuredSearchSpace[BoundaryT, StructuredCandidateT] = field(
        init=False,
        repr=False,
    )
    max_exchange_fraction: float = 0.2

    def __post_init__(self) -> None:
        """Validate the mutation configuration."""
        validate_fraction(
            self.max_exchange_fraction,
            name="max_exchange_fraction",
        )
        object.__setattr__(
            self,
            "structured_space",
            require_structured_space(self.space),
        )

    @property
    @override
    def arity(self) -> int:
        """Return the required number of parent candidates."""
        return 1

    @override
    def apply(
        self,
        parents: Sequence[StructuredCandidateT],
        random_state: np.random.RandomState,
    ) -> StructuredCandidateT:
        """Return a child formed by leaf-wise resampling.

        Parameters
        ----------
        parents : Sequence[StructuredCandidateT]
            Parent candidates. Exactly one parent is required.
        random_state : np.random.RandomState
            Random state used for path selection and resampling.

        Returns
        -------
        StructuredCandidateT
            Mutation child after random-reset resampling.

        Raises
        ------
        ValueError
            If ``parents`` does not match the required arity.
        TypeError
            If the configured search space is not structured.
        """
        require_parent_count(parents, arity=self.arity)
        child = _random_reset_mutation(
            space=self.structured_space,
            candidate=parents[0],
            max_exchange_fraction=self.max_exchange_fraction,
            random_state=random_state,
        )
        self.space.validate(child)
        return child

    def apply_on_paths(
        self,
        candidate: StructuredCandidateT,
        selected_paths: Sequence[LeafPath],
        random_state: np.random.RandomState,
    ) -> StructuredCandidateT:
        """Return a mutation child over one explicit set of structured leaves.

        Parameters
        ----------
        candidate : StructuredCandidateT
            Candidate to mutate.
        selected_paths : Sequence[LeafPath]
            Explicit leaf paths to resample.
        random_state : np.random.RandomState
            Random state used for leaf-value resampling.

        Returns
        -------
        StructuredCandidateT
            Mutation child restricted to ``selected_paths``.
        """
        child = _random_reset_mutation_on_paths(
            space=self.structured_space,
            candidate=candidate,
            selected_paths=selected_paths,
            random_state=random_state,
        )
        self.space.validate(child)
        return child

    @property
    def structured_candidate_space(
        self,
    ) -> StructuredSearchSpace[object, SpaceCandidateValue]:
        """Return the structured candidate-space view used by ask-side planning."""
        return cast(
            StructuredSearchSpace[object, SpaceCandidateValue],
            self.structured_space,
        )

    @property
    def max_selected_path_fraction(self) -> float:
        """Return the maximum fraction of active leaf paths eligible for reset."""
        return self.max_exchange_fraction

    def apply_space_candidate_on_paths(
        self,
        candidate: SpaceCandidateValue,
        selected_paths: Sequence[LeafPath],
        random_state: np.random.RandomState,
    ) -> SpaceCandidateValue:
        """Return one reset child from a type-erased structured candidate value."""
        candidate_value = require_space_candidate_value(
            candidate,
            operation="RandomResetMutation path editing",
        )
        structured_space = self.structured_candidate_space
        structured_space.validate(candidate_value)
        child = _random_reset_mutation_on_paths(
            space=structured_space,
            candidate=candidate_value,
            selected_paths=selected_paths,
            random_state=random_state,
        )
        structured_space.validate(child)
        return child


@dataclass(frozen=True, slots=True)
class BoundedMutation(FrozenGenericSlotsCompat,
    VariationOperator[StructuredCandidateT],
    Generic[BoundaryT, StructuredCandidateT],
):
    """Apply bounded perturbations to a random subset of candidate leaves.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, StructuredCandidateT]
        Search space used to validate parents and resulting children.
    max_perturbation_fraction : float, default=0.2
        Maximum fraction of each numeric leaf range used during perturbation.
    """

    space: SearchSpace[BoundaryT, StructuredCandidateT]
    structured_space: StructuredSearchSpace[BoundaryT, StructuredCandidateT] = field(
        init=False,
        repr=False,
    )
    max_perturbation_fraction: float = 0.2

    def __post_init__(self) -> None:
        """Validate the mutation configuration."""
        validate_fraction(
            self.max_perturbation_fraction,
            name="max_perturbation_fraction",
        )
        object.__setattr__(
            self,
            "structured_space",
            require_structured_space(self.space),
        )

    @property
    @override
    def arity(self) -> int:
        """Return the required number of parent candidates."""
        return 1

    @override
    def apply(
        self,
        parents: Sequence[StructuredCandidateT],
        random_state: np.random.RandomState,
    ) -> StructuredCandidateT:
        """Return a child formed by bounded leaf-wise perturbations.

        Parameters
        ----------
        parents : Sequence[StructuredCandidateT]
            Parent candidates. Exactly one parent is required.
        random_state : np.random.RandomState
            Random state used for path selection and perturbation sampling.

        Returns
        -------
        StructuredCandidateT
            Mutation child after bounded perturbation.

        Raises
        ------
        ValueError
            If ``parents`` does not match the required arity.
        TypeError
            If the configured search space is not structured.
        """
        require_parent_count(parents, arity=self.arity)
        child = _bounded_mutation(
            space=self.structured_space,
            candidate=parents[0],
            max_perturbation_fraction=self.max_perturbation_fraction,
            random_state=random_state,
        )
        self.space.validate(child)
        return child

    def apply_on_paths(
        self,
        candidate: StructuredCandidateT,
        selected_paths: Sequence[LeafPath],
        random_state: np.random.RandomState,
    ) -> StructuredCandidateT:
        """Return a mutation child over one explicit set of structured leaves.

        Parameters
        ----------
        candidate : StructuredCandidateT
            Candidate to mutate.
        selected_paths : Sequence[LeafPath]
            Explicit leaf paths to perturb.
        random_state : np.random.RandomState
            Random state used for perturbation sampling.

        Returns
        -------
        StructuredCandidateT
            Mutation child restricted to ``selected_paths``.
        """
        child = _bounded_mutation_on_paths(
            space=self.structured_space,
            candidate=candidate,
            selected_paths=selected_paths,
            max_perturbation_fraction=self.max_perturbation_fraction,
            random_state=random_state,
        )
        self.space.validate(child)
        return child

    @property
    def structured_candidate_space(
        self,
    ) -> StructuredSearchSpace[object, SpaceCandidateValue]:
        """Return the structured candidate-space view used by ask-side planning."""
        return cast(
            StructuredSearchSpace[object, SpaceCandidateValue],
            self.structured_space,
        )

    @property
    def max_selected_path_fraction(self) -> float:
        """Return the maximum fraction of active leaf paths eligible for perturbation."""
        return self.max_perturbation_fraction

    @property
    def max_coordinate_fraction(self) -> float:
        """Return the maximum coordinate-space fraction used for covariance guidance."""
        return self.max_perturbation_fraction

    def apply_space_candidate_on_paths(
        self,
        candidate: SpaceCandidateValue,
        selected_paths: Sequence[LeafPath],
        random_state: np.random.RandomState,
    ) -> SpaceCandidateValue:
        """Return one bounded-mutation child from a type-erased structured candidate value."""
        candidate_value = require_space_candidate_value(
            candidate,
            operation="BoundedMutation path editing",
        )
        structured_space = self.structured_candidate_space
        structured_space.validate(candidate_value)
        child = _bounded_mutation_on_paths(
            space=structured_space,
            candidate=candidate_value,
            selected_paths=selected_paths,
            max_perturbation_fraction=self.max_perturbation_fraction,
            random_state=random_state,
        )
        structured_space.validate(child)
        return child
