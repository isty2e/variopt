"""Internal space-derived permutation defaults."""

from collections.abc import Sequence
from dataclasses import dataclass

from ....diversity import StructuredSpaceDiversityMetric
from ....spaces import PermutationSpace
from .operators import InversionMutation, OrderCrossover


@dataclass(frozen=True, slots=True)
class PermutationVariationDefaults:
    """Cohesive permutation-default bundle for population optimizers.

    Parameters
    ----------
    crossover_operator : OrderCrossover
        Default permutation-safe crossover operator.
    mutation_operator : InversionMutation
        Default permutation-safe mutation operator.
    diversity_metric : StructuredSpaceDiversityMetric[Sequence[int], tuple[int, ...]]
        Default structured diversity metric for permutation candidates.
    """

    crossover_operator: OrderCrossover
    mutation_operator: InversionMutation
    diversity_metric: StructuredSpaceDiversityMetric[Sequence[int], tuple[int, ...]]


def derive_permutation_variation_defaults(
    space: PermutationSpace,
) -> PermutationVariationDefaults:
    """Derive default permutation-safe variation components.

    Parameters
    ----------
    space : PermutationSpace
        Permutation search space that defines the candidate domain.

    Returns
    -------
    PermutationVariationDefaults
        Default crossover, mutation, and diversity components for the space.
    """
    return PermutationVariationDefaults(
        crossover_operator=OrderCrossover(space=space),
        mutation_operator=InversionMutation(space=space),
        diversity_metric=StructuredSpaceDiversityMetric(space=space),
    )
