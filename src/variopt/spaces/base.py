"""Search-space interface definitions."""

from abc import ABC, abstractmethod
from typing import Generic

import numpy as np

from ..typevars import CandidateT, InputT
from .equality import scalar_candidate_equality


class SearchSpace(ABC, Generic[InputT, CandidateT]):
    """Domain specification for canonical candidates.

    Notes
    -----
    A search space owns four responsibilities:

    - normalization from boundary input into canonical candidate form,
    - validation of canonical candidates,
    - explicit-randomness sampling, and
    - equality semantics between canonical candidates.

    Concrete spaces may add richer traversal or geometry behavior, but these
    operations define the minimal public contract.
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

    def candidates_equal(
        self,
        left_candidate: CandidateT,
        right_candidate: CandidateT,
    ) -> bool:
        """Return whether two canonical candidates denote the same space point.

        Parameters
        ----------
        left_candidate : CandidateT
            Left canonical candidate to compare.
        right_candidate : CandidateT
            Right canonical candidate to compare.

        Returns
        -------
        bool
            Whether both candidates are equal under this space's candidate
            identity semantics.

        Raises
        ------
        TypeError
            If either candidate is not canonical for this space or if the
            default equality result is not scalar.
        ValueError
            If either candidate is canonical in shape but outside this space.
        """
        self.validate(left_candidate)
        self.validate(right_candidate)
        return scalar_candidate_equality(left_candidate, right_candidate)
