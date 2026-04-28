"""Search-space interface definitions."""

from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from ..typevars import CandidateT, InputT


class SearchSpace(ABC, Generic[InputT, CandidateT]):
    """Domain specification for canonical candidates.

    Notes
    -----
    A search space owns three responsibilities:

    - normalization from boundary input into canonical candidate form,
    - validation of canonical candidates, and
    - explicit-randomness sampling.

    Concrete spaces may add richer traversal or geometry behavior, but these
    three operations define the minimal public contract.
    """

    @abstractmethod
    def normalize(self, raw_candidate: InputT) -> CandidateT:
        """Normalize boundary input into canonical candidate form.

        Parameters
        ----------
        raw_candidate : InputT
            Boundary-level candidate representation accepted by the space.

        Returns
        -------
        CandidateT
            Canonical candidate representation used internally by the library.
        """

    @abstractmethod
    def validate(self, candidate: CandidateT) -> None:
        """Validate a canonical candidate.

        Parameters
        ----------
        candidate : CandidateT
            Canonical candidate to validate.

        Raises
        ------
        TypeError
            If ``candidate`` is not in the canonical representation expected by
            the space.
        ValueError
            If ``candidate`` is canonical in shape but violates domain
            constraints.
        """

    @abstractmethod
    def sample(self, random_state: np.random.RandomState) -> CandidateT:
        """Sample a canonical candidate using explicit randomness.

        Parameters
        ----------
        random_state : numpy.random.RandomState
            Random-state object that owns all stochasticity for the sample.

        Returns
        -------
        CandidateT
            Canonical sampled candidate.
        """
