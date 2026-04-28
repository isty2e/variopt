"""Diversity metric interface definitions."""

from abc import ABC, abstractmethod
from typing import Generic

from ..typevars import CandidateT


class DiversityMetric(ABC, Generic[CandidateT]):
    """Distance-like metric used by diversity-aware optimizers.

    Notes
    -----
    Diversity metrics define the candidate-to-candidate distance law consumed
    by diversity-aware optimizers and bank-management logic.
    """

    @abstractmethod
    def distance(self, left: CandidateT, right: CandidateT) -> float:
        """Compute the diversity distance between two candidates.

        Parameters
        ----------
        left : CandidateT
            Left candidate in canonical search-space representation.
        right : CandidateT
            Right candidate in canonical search-space representation.

        Returns
        -------
        float
            Non-negative diversity distance between ``left`` and ``right``.
        """
