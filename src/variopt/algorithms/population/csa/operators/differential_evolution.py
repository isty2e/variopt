"""Differential-evolution-style CSA variation wrapper."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np
from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....operators import VariationOperator
from .....spaces import IntegerSpace, RealSpace, SearchSpace, StructuredSearchSpace
from .....spaces.types import SpaceCandidateValue
from ...de.variation import (
    differential_evolution_variation as _differential_evolution_variation,
)
from .validation import (
    require_parent_count,
    require_structured_space,
    validate_probability,
)

BoundaryT = TypeVar("BoundaryT")
StructuredCandidateT = TypeVar("StructuredCandidateT", bound=SpaceCandidateValue)


@dataclass(frozen=True, slots=True)
class DifferentialEvolutionVariation(
    FrozenGenericSlotsCompat,
    VariationOperator[StructuredCandidateT],
    Generic[BoundaryT, StructuredCandidateT],
):
    """Apply DE-style donor recombination over numeric structured leaves.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, StructuredCandidateT]
        Structured search space whose numeric leaves will be recombined.
    mutation_range : tuple[float, float], default=(0.5, 1.0)
        Inclusive range used to sample the DE mutation factor.
    recombination_probability : float, default=0.7
        Probability of taking donor coordinates during recombination.
    n_cross : int, default=1
        Minimum number of forced crossover positions.
    """

    space: SearchSpace[BoundaryT, StructuredCandidateT]
    structured_space: StructuredSearchSpace[BoundaryT, StructuredCandidateT] = field(
        init=False,
        repr=False,
    )
    mutation_range: tuple[float, float] = (0.5, 1.0)
    recombination_probability: float = 0.7
    n_cross: int = 1

    def __post_init__(self) -> None:
        """Validate the DE configuration."""
        structured_space = require_structured_space(self.space)
        object.__setattr__(self, "structured_space", structured_space)
        for path in structured_space.leaf_paths():
            if not isinstance(
                structured_space.leaf_space_at_path(path), (RealSpace, IntegerSpace)
            ):
                msg = (
                    "space must contain only numeric leaves for differential evolution"
                )
                raise TypeError(msg)

        mutation_low, mutation_high = self.mutation_range
        if mutation_low > mutation_high:
            msg = "mutation_range low must not exceed mutation_range high"
            raise ValueError(msg)

        if mutation_low < 0.0:
            msg = "mutation_range low must be non-negative"
            raise ValueError(msg)

        validate_probability(
            self.recombination_probability,
            name="recombination_probability",
        )

        if self.n_cross <= 0:
            msg = "n_cross must be positive"
            raise ValueError(msg)

    @property
    @override
    def arity(self) -> int:
        """Return the required number of parent candidates."""
        return 4

    @override
    def apply(
        self,
        parents: Sequence[StructuredCandidateT],
        random_state: np.random.RandomState,
    ) -> StructuredCandidateT:
        """Generate one DE-style child.

        Parameters
        ----------
        parents : Sequence[StructuredCandidateT]
            Parent tuple containing target, base, and two differential parents.
        random_state : np.random.RandomState
            Random state used for mutation-factor sampling and DE variation.

        Returns
        -------
        StructuredCandidateT
            Child candidate produced by DE-style donor recombination.
        """
        require_parent_count(parents, arity=self.arity)
        mutation_low, mutation_high = self.mutation_range
        mutation_factor = mutation_low
        if mutation_high > mutation_low:
            mutation_factor += (mutation_high - mutation_low) * float(
                random_state.random_sample()
            )

        child = _differential_evolution_variation(
            space=self.structured_space,
            target_parent=parents[0],
            base_parent=parents[1],
            differential_parent_a=parents[2],
            differential_parent_b=parents[3],
            mutation_factor=mutation_factor,
            recombination_probability=self.recombination_probability,
            n_cross=self.n_cross,
            random_state=random_state,
        )
        self.space.validate(child)
        return child
