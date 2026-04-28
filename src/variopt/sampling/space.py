"""Search-space-derived candidate samplers."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ..spaces import SearchSpace
from ..typevars import CandidateT
from .base import CandidateSampler

BoundaryT = TypeVar("BoundaryT")


@dataclass(frozen=True, slots=True)
class SearchSpaceSampler(FrozenGenericSlotsCompat,
    CandidateSampler[CandidateT],
    Generic[BoundaryT, CandidateT],
):
    """Candidate sampler that delegates to one search space's default prior.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, CandidateT]
        Search space whose default sampling prior should be used.
    """

    space: SearchSpace[BoundaryT, CandidateT]

    @override
    def sample(self, random_state: np.random.RandomState) -> CandidateT:
        """Return one sample from the declared search space.

        Parameters
        ----------
        random_state : np.random.RandomState
            Random state forwarded to ``space.sample``.

        Returns
        -------
        CandidateT
            Canonical sample drawn from ``space``.
        """
        return self.space.sample(random_state)
