"""Permutation-safe variation operators and kernels."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from typing_extensions import override

from ....operators import VariationOperator
from ....randomness import (
    random_state_choice_indices_without_replacement,
    random_state_randint,
)
from ....spaces import PermutationSpace


def validate_fraction(*, value: float, name: str) -> None:
    """Reject one out-of-range fraction value.

    Parameters
    ----------
    value : float
        Fraction value to validate.
    name : str
        Parameter name used in the validation error.

    Raises
    ------
    ValueError
        If ``value`` is not in ``(0.0, 1.0]``.
    """
    if not 0.0 < value <= 1.0:
        msg = f"{name} must be in (0.0, 1.0]"
        raise ValueError(msg)


def require_parent_count(
    parents: Sequence[tuple[int, ...]],
    *,
    arity: int,
) -> None:
    """Reject one invalid parent tuple length.

    Parameters
    ----------
    parents : Sequence[tuple[int, ...]]
        Parent candidate sequence supplied to an operator.
    arity : int
        Required parent count for the operator.

    Raises
    ------
    ValueError
        If ``parents`` does not contain exactly ``arity`` candidates.
    """
    if len(parents) != arity:
        msg = f"operator requires exactly {arity} parent candidates"
        raise ValueError(msg)


def sample_exchange_count(
    *,
    leaf_count: int,
    max_exchange_fraction: float,
    random_state: np.random.RandomState,
) -> int:
    """Return one bounded position-count sample.

    Parameters
    ----------
    leaf_count : int
        Number of permutation positions available for exchange.
    max_exchange_fraction : float
        Fractional cap on the number of exchanged positions.
    random_state : numpy.random.RandomState
        Random generator used for deterministic sampling.

    Returns
    -------
    int
        Number of positions to exchange.

    Raises
    ------
    ValueError
        If ``leaf_count`` is not positive.
    """
    if leaf_count <= 0:
        msg = "leaf_count must be positive"
        raise ValueError(msg)

    if leaf_count == 1:
        return 1

    max_exchange_count = min(
        leaf_count,
        max(1, int(leaf_count * max_exchange_fraction)),
    )
    return random_state_randint(random_state, 1, max_exchange_count + 1)


def sample_segment_bounds(
    *,
    size: int,
    max_segment_fraction: float,
    random_state: np.random.RandomState,
) -> tuple[int, int]:
    """Return one contiguous segment in ``[start, end)``.

    Parameters
    ----------
    size : int
        Permutation length.
    max_segment_fraction : float
        Fractional cap on the segment length.
    random_state : numpy.random.RandomState
        Random generator used for deterministic sampling.

    Returns
    -------
    tuple[int, int]
        Inclusive-exclusive segment bounds.

    Raises
    ------
    ValueError
        If ``size`` is not positive.
    """
    if size <= 0:
        msg = "size must be positive"
        raise ValueError(msg)

    if size == 1:
        return 0, 1

    segment_length = sample_exchange_count(
        leaf_count=size,
        max_exchange_fraction=max_segment_fraction,
        random_state=random_state,
    )
    segment_length = max(1, min(size, segment_length))
    if segment_length == size:
        return 0, size

    start_index = random_state_randint(
        random_state,
        size - segment_length + 1,
    )
    return start_index, start_index + segment_length


def sample_swap_count(
    *,
    size: int,
    max_swap_fraction: float,
    random_state: np.random.RandomState,
) -> int:
    """Return the number of disjoint swaps to apply.

    Parameters
    ----------
    size : int
        Permutation length.
    max_swap_fraction : float
        Fractional cap on the number of touched positions.
    random_state : numpy.random.RandomState
        Random generator used for deterministic sampling.

    Returns
    -------
    int
        Number of disjoint swaps to perform.
    """
    if size < 2:
        return 0

    max_touched_positions = max(2, int(size * max_swap_fraction))
    max_swap_count = min(size // 2, max_touched_positions // 2)
    if max_swap_count <= 1:
        return 1

    return random_state_randint(random_state, 1, max_swap_count + 1)


def order_crossover(
    *,
    space: PermutationSpace,
    primary_parent: tuple[int, ...],
    partner_parent: tuple[int, ...],
    max_segment_fraction: float,
    random_state: np.random.RandomState,
) -> tuple[int, ...]:
    """Return one order-crossover child.

    Parameters
    ----------
    space : PermutationSpace
        Permutation space that defines candidate validity.
    primary_parent : tuple[int, ...]
        Parent contributing the preserved contiguous segment.
    partner_parent : tuple[int, ...]
        Parent contributing the remaining ordering.
    max_segment_fraction : float
        Fractional cap on the preserved segment length.
    random_state : numpy.random.RandomState
        Random generator used for deterministic sampling.

    Returns
    -------
    tuple[int, ...]
        Canonical order-crossover child.

    Raises
    ------
    TypeError
        If either parent is not canonical for ``space``.
    ValueError
        If a parent violates the permutation domain.
    """
    space.validate(primary_parent)
    space.validate(partner_parent)

    start_index, end_index = sample_segment_bounds(
        size=space.size,
        max_segment_fraction=max_segment_fraction,
        random_state=random_state,
    )
    child: list[int | None] = [None] * space.size
    used = [False] * space.size
    for index in range(start_index, end_index):
        value = primary_parent[index]
        child[index] = value
        used[value] = True

    partner_values = tuple(value for value in partner_parent if not used[value])
    partner_index = 0
    for index in range(end_index, space.size):
        if child[index] is None:
            child[index] = partner_values[partner_index]
            partner_index += 1
    for index in range(0, start_index):
        if child[index] is None:
            child[index] = partner_values[partner_index]
            partner_index += 1

    normalized_child = tuple(value for value in child if value is not None)
    return space.normalize(normalized_child)


def swap_mutation(
    *,
    space: PermutationSpace,
    candidate: tuple[int, ...],
    max_swap_fraction: float,
    random_state: np.random.RandomState,
) -> tuple[int, ...]:
    """Return one permutation child formed by disjoint swaps.

    Parameters
    ----------
    space : PermutationSpace
        Permutation space that defines candidate validity.
    candidate : tuple[int, ...]
        Canonical permutation candidate to mutate.
    max_swap_fraction : float
        Fractional cap on the number of touched positions.
    random_state : numpy.random.RandomState
        Random generator used for deterministic sampling.

    Returns
    -------
    tuple[int, ...]
        Mutated permutation candidate.

    Raises
    ------
    TypeError
        If ``candidate`` is not canonical for ``space``.
    ValueError
        If ``candidate`` violates the permutation domain.
    """
    space.validate(candidate)
    if space.size < 2:
        return candidate

    swap_count = sample_swap_count(
        size=space.size,
        max_swap_fraction=max_swap_fraction,
        random_state=random_state,
    )
    selected_indices = random_state_choice_indices_without_replacement(
        random_state,
        population_size=space.size,
        count=2 * swap_count,
    )
    child = list(candidate)
    for index in range(0, len(selected_indices), 2):
        left_index = selected_indices[index]
        right_index = selected_indices[index + 1]
        child[left_index], child[right_index] = child[right_index], child[left_index]
    return tuple(child)


def inversion_mutation(
    *,
    space: PermutationSpace,
    candidate: tuple[int, ...],
    max_inversion_fraction: float,
    random_state: np.random.RandomState,
) -> tuple[int, ...]:
    """Return one permutation child formed by inverting one segment.

    Parameters
    ----------
    space : PermutationSpace
        Permutation space that defines candidate validity.
    candidate : tuple[int, ...]
        Canonical permutation candidate to mutate.
    max_inversion_fraction : float
        Fractional cap on the inverted segment length.
    random_state : numpy.random.RandomState
        Random generator used for deterministic sampling.

    Returns
    -------
    tuple[int, ...]
        Mutated permutation candidate.

    Raises
    ------
    TypeError
        If ``candidate`` is not canonical for ``space``.
    ValueError
        If ``candidate`` violates the permutation domain.
    """
    space.validate(candidate)
    if space.size < 2:
        return candidate

    start_index, end_index = sample_segment_bounds(
        size=space.size,
        max_segment_fraction=max_inversion_fraction,
        random_state=random_state,
    )
    if end_index - start_index < 2:
        if end_index == space.size:
            start_index -= 1
        else:
            end_index += 1

    child = list(candidate)
    child[start_index:end_index] = reversed(child[start_index:end_index])
    return tuple(child)


@dataclass(frozen=True, slots=True)
class OrderCrossover(VariationOperator[tuple[int, ...]]):
    """Permutation-safe order crossover over one contiguous segment.

    Parameters
    ----------
    space : PermutationSpace
        Permutation space that defines parent and child validity.
    max_segment_fraction : float, default=0.5
        Fractional cap on the segment copied directly from the primary parent.
    """

    space: PermutationSpace
    max_segment_fraction: float = 0.5

    def __post_init__(self) -> None:
        """Validate the crossover configuration."""
        validate_fraction(
            value=self.max_segment_fraction,
            name="max_segment_fraction",
        )

    @property
    @override
    def arity(self) -> int:
        """Return the required number of parent candidates."""
        return 2

    @override
    def apply(
        self,
        parents: Sequence[tuple[int, ...]],
        random_state: np.random.RandomState,
    ) -> tuple[int, ...]:
        """Return one order-crossover child.

        Parameters
        ----------
        parents : Sequence[tuple[int, ...]]
            Two parent permutations in primary/partner order.
        random_state : numpy.random.RandomState
            Random generator used for deterministic sampling.

        Returns
        -------
        tuple[int, ...]
            Canonical order-crossover child.

        Raises
        ------
        ValueError
            If ``parents`` does not contain exactly two candidates.
        """
        require_parent_count(parents, arity=self.arity)
        return order_crossover(
            space=self.space,
            primary_parent=parents[0],
            partner_parent=parents[1],
            max_segment_fraction=self.max_segment_fraction,
            random_state=random_state,
        )


@dataclass(frozen=True, slots=True)
class SwapMutation(VariationOperator[tuple[int, ...]]):
    """Permutation-safe mutation that swaps disjoint position pairs.

    Parameters
    ----------
    space : PermutationSpace
        Permutation space that defines candidate validity.
    max_swap_fraction : float, default=0.2
        Fractional cap on the number of touched positions.
    """

    space: PermutationSpace
    max_swap_fraction: float = 0.2

    def __post_init__(self) -> None:
        """Validate the mutation configuration."""
        validate_fraction(
            value=self.max_swap_fraction,
            name="max_swap_fraction",
        )

    @property
    @override
    def arity(self) -> int:
        """Return the required number of parent candidates."""
        return 1

    @override
    def apply(
        self,
        parents: Sequence[tuple[int, ...]],
        random_state: np.random.RandomState,
    ) -> tuple[int, ...]:
        """Return one swap-mutation child.

        Parameters
        ----------
        parents : Sequence[tuple[int, ...]]
            Single parent permutation to mutate.
        random_state : numpy.random.RandomState
            Random generator used for deterministic sampling.

        Returns
        -------
        tuple[int, ...]
            Canonical swap-mutation child.

        Raises
        ------
        ValueError
            If ``parents`` does not contain exactly one candidate.
        """
        require_parent_count(parents, arity=self.arity)
        return swap_mutation(
            space=self.space,
            candidate=parents[0],
            max_swap_fraction=self.max_swap_fraction,
            random_state=random_state,
        )


@dataclass(frozen=True, slots=True)
class InversionMutation(VariationOperator[tuple[int, ...]]):
    """Permutation-safe mutation that reverses one contiguous segment.

    Parameters
    ----------
    space : PermutationSpace
        Permutation space that defines candidate validity.
    max_inversion_fraction : float, default=0.2
        Fractional cap on the inverted segment length.
    """

    space: PermutationSpace
    max_inversion_fraction: float = 0.2

    def __post_init__(self) -> None:
        """Validate the mutation configuration."""
        validate_fraction(
            value=self.max_inversion_fraction,
            name="max_inversion_fraction",
        )

    @property
    @override
    def arity(self) -> int:
        """Return the required number of parent candidates."""
        return 1

    @override
    def apply(
        self,
        parents: Sequence[tuple[int, ...]],
        random_state: np.random.RandomState,
    ) -> tuple[int, ...]:
        """Return one inversion-mutation child.

        Parameters
        ----------
        parents : Sequence[tuple[int, ...]]
            Single parent permutation to mutate.
        random_state : numpy.random.RandomState
            Random generator used for deterministic sampling.

        Returns
        -------
        tuple[int, ...]
            Canonical inversion-mutation child.

        Raises
        ------
        ValueError
            If ``parents`` does not contain exactly one candidate.
        """
        require_parent_count(parents, arity=self.arity)
        return inversion_mutation(
            space=self.space,
            candidate=parents[0],
            max_inversion_fraction=self.max_inversion_fraction,
            random_state=random_state,
        )
