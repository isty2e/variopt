"""Candidate sampler interface definitions."""

from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from ..typevars import CandidateT


class CandidateSampler(ABC, Generic[CandidateT]):
    """Candidate initialization component used at boundary sampling seams.

    Notes
    -----
    Samplers provide the stochastic prior over canonical candidates used by
    optimizers during initialization or restart paths.
    """

    @abstractmethod
    def sample(self, random_state: np.random.RandomState) -> CandidateT:
        """Return one canonical candidate sample.

        Parameters
        ----------
        random_state : np.random.RandomState
            Random state that owns all sampler stochasticity.

        Returns
        -------
        CandidateT
            Canonical sampled candidate.
        """
