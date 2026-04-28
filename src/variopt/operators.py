"""Variation operator interface definitions."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic

import numpy as np

from .typevars import CandidateT


class VariationOperator(ABC, Generic[CandidateT]):
    """Candidate transformation component used by an optimizer.

    Notes
    -----
    Variation operators consume one or more parent candidates and produce a new
    child candidate in canonical representation.
    """

    @property
    @abstractmethod
    def arity(self) -> int:
        """Return the required number of parent candidates."""

    @abstractmethod
    def apply(
        self,
        parents: Sequence[CandidateT],
        random_state: np.random.RandomState,
    ) -> CandidateT:
        """Produce a child candidate from the supplied parents.

        Parameters
        ----------
        parents : Sequence[CandidateT]
            Parent candidates consumed by the operator.
        random_state : np.random.RandomState
            Random-state instance used for stochastic variation.

        Returns
        -------
        CandidateT
            Child candidate produced by the operator.
        """
