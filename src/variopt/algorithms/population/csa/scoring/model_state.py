"""CSA score-model state definitions."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import exp
from typing import Generic

from typing_extensions import Self

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .....artifacts import Observation
from .....distance import require_valid_distance
from .....diversity import DiversityMetric
from .....json_types import JSONDict, JSONValue
from .....typevars import CandidateT
from ..banking.bank import BankEntry
from .adaptive_potential import (
    AdaptiveBinIndex,
    AdaptivePotentialState,
    build_adaptive_potential_state,
)
from .model import CSAScoreModel


@dataclass(frozen=True, slots=True)
class ScoredBank(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """One bank with aligned real and shaped score views.

    Parameters
    ----------
    real_scores : tuple[float, ...]
        Objective scores for the current bank snapshot in bank order.
    shaped_scores : tuple[float, ...]
        Scores after applying the active biased and adaptive potential terms.
    biased_sigma2 : float | None
        Effective biased-potential variance used for the shaping pass, or
        ``None`` when biased potential is disabled.
    """

    real_scores: tuple[float, ...]
    shaped_scores: tuple[float, ...]
    biased_sigma2: float | None


@dataclass(frozen=True, slots=True)
class ScoredTrial(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """One scored trial observation under the active CSA score model.

    Parameters
    ----------
    candidate : CandidateT
        Trial candidate evaluated against the current bank.
    real_score : float
        Raw objective score returned by the evaluator.
    shaped_score : float
        Score after applying the active biased and adaptive potentials.
    adaptive_bin_index : AdaptiveBinIndex
        Adaptive-potential bin touched by the candidate, or ``None`` when the
        adaptive potential is disabled.
    """

    candidate: CandidateT
    real_score: float
    shaped_score: float
    adaptive_bin_index: AdaptiveBinIndex


@dataclass(frozen=True, slots=True)
class CSAScoreModelState(FrozenGenericSlotsCompat, Generic[CandidateT]):
    """Canonical runtime state for CSA score shaping.

    Parameters
    ----------
    score_model : CSAScoreModel[CandidateT]
        Score model controlling biased and adaptive potential behavior.
    biased_potential_max : float | None, default=None
        Resolved maximum biased-potential penalty. ``None`` means it has not
        been resolved yet or biased potential is disabled.
    adaptive_potential_state : AdaptivePotentialState[CandidateT] | None, default=None
        Runtime state for the adaptive potential component. When omitted, the
        state is derived from ``score_model.adaptive_potential``.
    """

    score_model: CSAScoreModel[CandidateT]
    biased_potential_max: float | None = None
    adaptive_potential_state: AdaptivePotentialState[CandidateT] | None = None

    def __post_init__(self) -> None:
        """Normalize the optional adaptive-potential state at ingress."""
        if self.adaptive_potential_state is not None:
            return

        object.__setattr__(
            self,
            "adaptive_potential_state",
            build_adaptive_potential_state(
                self.score_model.adaptive_potential,
            ),
        )

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the score-model state.

        Returns
        -------
        JSONDict
            JSON-safe score-model-state snapshot.
        """
        return {
            "biased_potential_max": self.biased_potential_max,
            "adaptive_potential_state": (
                None
                if self.adaptive_potential_state is None
                else self.adaptive_potential_state.to_dict()
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
        *,
        score_model: CSAScoreModel[CandidateT],
    ) -> Self:
        """Build a score-model state from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe score-model-state snapshot.
        score_model : CSAScoreModel[CandidateT]
            Score model that owns the reconstructed state.

        Returns
        -------
        Self
            Reconstructed score-model state.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        biased_potential_max = data.get("biased_potential_max")
        raw_adaptive_potential_state = data.get("adaptive_potential_state")
        if biased_potential_max is not None and not isinstance(biased_potential_max, (int, float)):
            msg = "score-model-state snapshot requires numeric biased_potential_max or null"
            raise TypeError(msg)
        if raw_adaptive_potential_state is not None and not isinstance(raw_adaptive_potential_state, dict):
            msg = "score-model-state snapshot requires adaptive_potential_state mapping or null"
            raise TypeError(msg)

        adaptive_potential_state = None
        if raw_adaptive_potential_state is not None:
            adaptive_model = score_model.adaptive_potential
            if adaptive_model is None:
                msg = "score-model-state snapshot cannot restore adaptive potential without an adaptive score model"
                raise ValueError(msg)
            adaptive_potential_state = AdaptivePotentialState[CandidateT].from_dict(
                raw_adaptive_potential_state,
                model=adaptive_model,
            )

        return cls(
            score_model=score_model,
            biased_potential_max=(
                None
                if biased_potential_max is None
                else float(biased_potential_max)
            ),
            adaptive_potential_state=adaptive_potential_state,
        )

    def score_bank(
        self,
        *,
        entries: Sequence[BankEntry[CandidateT]],
        diversity_metric: DiversityMetric[CandidateT],
        distance_cutoff: float,
        minimum_distance_cutoff: float | None,
        masked_entry_indices: frozenset[int],
    ) -> tuple[ScoredBank[CandidateT], Self]:
        """Return shaped bank scores and the updated score-model state.

        Parameters
        ----------
        entries : Sequence[BankEntry[CandidateT]]
            Bank entries to score in their current bank order.
        diversity_metric : DiversityMetric[CandidateT]
            Distance metric used by biased and adaptive potentials.
        distance_cutoff : float
            Active CSA distance cutoff.
        minimum_distance_cutoff : float | None
            Optional lower bound on the runtime cutoff scale.
        masked_entry_indices : frozenset[int]
            Entry indices excluded from biased-potential accumulation.

        Returns
        -------
        tuple[ScoredBank[CandidateT], Self]
            Scored bank snapshot and the resolved score-model state.

        Raises
        ------
        ValueError
            If the diversity metric produces an invalid distance.
        """
        real_scores = tuple(entry.value for entry in entries)
        resolved_bias_max = self._resolve_biased_potential_max(real_scores)
        next_state = self
        if resolved_bias_max != self.biased_potential_max:
            next_state = type(self)(
                score_model=self.score_model,
                biased_potential_max=resolved_bias_max,
                adaptive_potential_state=self.adaptive_potential_state,
            )

        shaped_scores = list(real_scores)
        biased_sigma2 = next_state._resolve_biased_sigma2(
            distance_cutoff=distance_cutoff,
            minimum_distance_cutoff=minimum_distance_cutoff,
        )
        if biased_sigma2 is not None and resolved_bias_max is not None:
            for left_index, left_entry in enumerate(entries):
                if left_index in masked_entry_indices:
                    continue

                bias_sum = 0.0
                for right_index, right_entry in enumerate(entries):
                    if left_index == right_index:
                        continue

                    if left_entry.value < right_entry.value:
                        continue

                    distance = require_valid_distance(
                        diversity_metric.distance(
                            left_entry.candidate,
                            right_entry.candidate,
                        )
                    )
                    bias_sum += exp(-(distance**2) / biased_sigma2)

                shaped_scores[left_index] += resolved_bias_max * bias_sum

        adaptive_state = next_state.adaptive_potential_state
        if adaptive_state is not None:
            for index, entry in enumerate(entries):
                energy, _ = adaptive_state.score_candidate(
                    candidate=entry.candidate,
                    diversity_metric=diversity_metric,
                )
                shaped_scores[index] += energy

        return ScoredBank(
            real_scores=real_scores,
            shaped_scores=tuple(shaped_scores),
            biased_sigma2=biased_sigma2,
        ), next_state

    def score_trial(
        self,
        *,
        observation: Observation[CandidateT],
        bank_real_scores: Sequence[float],
        entry_distances: Sequence[float],
        diversity_metric: DiversityMetric[CandidateT],
        distance_cutoff: float,
        minimum_distance_cutoff: float | None,
    ) -> ScoredTrial[CandidateT]:
        """Return the shaped trial score for one observation.

        Parameters
        ----------
        observation : Observation[CandidateT]
            Trial observation to score.
        bank_real_scores : Sequence[float]
            Raw bank-entry scores aligned with ``entry_distances``.
        entry_distances : Sequence[float]
            Distances from the trial candidate to each bank entry.
        diversity_metric : DiversityMetric[CandidateT]
            Distance metric used by the adaptive potential.
        distance_cutoff : float
            Active CSA distance cutoff.
        minimum_distance_cutoff : float | None
            Optional lower bound on the runtime cutoff scale.

        Returns
        -------
        ScoredTrial[CandidateT]
            Shaped trial record aligned with the active score model.
        """
        shaped_score = observation.score
        biased_sigma2 = self._resolve_biased_sigma2(
            distance_cutoff=distance_cutoff,
            minimum_distance_cutoff=minimum_distance_cutoff,
        )
        if biased_sigma2 is not None and self.biased_potential_max is not None:
            bias_sum = 0.0
            for entry_score, distance in zip(bank_real_scores, entry_distances, strict=False):
                if entry_score <= observation.score:
                    bias_sum += exp(-(distance**2) / biased_sigma2)
            shaped_score += self.biased_potential_max * bias_sum

        adaptive_bin_index: AdaptiveBinIndex = None
        adaptive_state = self.adaptive_potential_state
        if adaptive_state is not None:
            adaptive_energy, adaptive_bin_index = adaptive_state.score_candidate(
                candidate=observation.candidate,
                diversity_metric=diversity_metric,
            )
            shaped_score += adaptive_energy

        return ScoredTrial(
            candidate=observation.candidate,
            real_score=observation.score,
            shaped_score=shaped_score,
            adaptive_bin_index=adaptive_bin_index,
        )

    def trial_adjusted_bank_scores(
        self,
        *,
        scored_bank: ScoredBank[CandidateT],
        trial_real_score: float,
        entry_distances: Sequence[float],
    ) -> tuple[float, ...]:
        """Return bank scores after one trial-specific biased adjustment.

        Parameters
        ----------
        scored_bank : ScoredBank[CandidateT]
            Bank scores produced by :meth:`score_bank`.
        trial_real_score : float
            Raw score for the candidate being compared against the bank.
        entry_distances : Sequence[float]
            Distances from the trial candidate to each bank entry.

        Returns
        -------
        tuple[float, ...]
            Comparison-ready bank scores after injecting the trial-specific
            biased-potential term.
        """
        if (
            scored_bank.biased_sigma2 is None
            or self.biased_potential_max is None
        ):
            return scored_bank.shaped_scores

        adjusted_scores = list(scored_bank.shaped_scores)
        for index, (entry_score, distance) in enumerate(
            zip(scored_bank.real_scores, entry_distances, strict=False),
        ):
            if trial_real_score <= entry_score:
                adjusted_scores[index] += self.biased_potential_max * exp(
                    -(distance**2) / scored_bank.biased_sigma2
                )

        return tuple(adjusted_scores)

    def comparison_score_for_entry(
        self,
        *,
        base_score: float,
        entry_real_score: float,
        trial_real_score: float,
        entry_distance: float,
        biased_sigma2: float | None,
    ) -> float:
        """Return the effective comparison score for one bank entry.

        Parameters
        ----------
        base_score : float
            Shaped bank-entry score before trial-specific adjustment.
        entry_real_score : float
            Raw bank-entry objective score.
        trial_real_score : float
            Raw objective score for the trial candidate.
        entry_distance : float
            Distance from the trial candidate to the bank entry.
        biased_sigma2 : float | None
            Effective biased-potential variance, or ``None`` when biased
            potential is inactive.

        Returns
        -------
        float
            Score used for the bank-entry comparison step.
        """
        if (
            biased_sigma2 is None
            or self.biased_potential_max is None
            or trial_real_score <= entry_real_score
        ):
            return base_score

        return base_score + (
            self.biased_potential_max
            * exp(-(entry_distance**2) / biased_sigma2)
        )

    def bump_trial(self, trial: ScoredTrial[CandidateT]) -> Self:
        """Return a state with adaptive potential incremented at one trial bin.

        Parameters
        ----------
        trial : ScoredTrial[CandidateT]
            Scored trial carrying the adaptive bin to increment.

        Returns
        -------
        Self
            Updated score-model state. Returns ``self`` when adaptive
            potential is disabled.
        """
        adaptive_state = self.adaptive_potential_state
        if adaptive_state is None:
            return self

        return type(self)(
            score_model=self.score_model,
            biased_potential_max=self.biased_potential_max,
            adaptive_potential_state=adaptive_state.increment(trial.adaptive_bin_index),
        )

    def bump_candidate(
        self,
        *,
        candidate: CandidateT,
        diversity_metric: DiversityMetric[CandidateT],
    ) -> Self:
        """Return a state with adaptive potential incremented at one candidate bin.

        Parameters
        ----------
        candidate : CandidateT
            Candidate whose adaptive bin should be incremented.
        diversity_metric : DiversityMetric[CandidateT]
            Distance metric used to locate the adaptive bin.

        Returns
        -------
        Self
            Updated score-model state. Returns ``self`` when adaptive
            potential is disabled.
        """
        adaptive_state = self.adaptive_potential_state
        if adaptive_state is None:
            return self

        bin_index = adaptive_state.bin_index(
            candidate=candidate,
            diversity_metric=diversity_metric,
        )
        return type(self)(
            score_model=self.score_model,
            biased_potential_max=self.biased_potential_max,
            adaptive_potential_state=adaptive_state.increment(bin_index),
        )

    def _resolve_biased_potential_max(
        self,
        real_scores: Sequence[float],
    ) -> float | None:
        biased_potential = self.score_model.biased_potential
        if biased_potential is None:
            return None

        if self.biased_potential_max is not None:
            return self.biased_potential_max

        if biased_potential.maximum_bias is not None:
            return biased_potential.maximum_bias

        if len(real_scores) == 0:
            return 0.0

        return max(real_scores) - min(real_scores)

    def _resolve_biased_sigma2(
        self,
        *,
        distance_cutoff: float,
        minimum_distance_cutoff: float | None,
    ) -> float | None:
        biased_potential = self.score_model.biased_potential
        if biased_potential is None:
            return None

        sigma_distance: float
        if biased_potential.sigma_reference == "constant":
            sigma_distance = 1.0
        elif biased_potential.sigma_reference == "minimum_distance_cutoff":
            sigma_distance = (
                distance_cutoff
                if minimum_distance_cutoff is None
                else minimum_distance_cutoff
            )
        else:
            sigma_distance = distance_cutoff

        if sigma_distance <= 0.0:
            msg = "biased potential requires a positive runtime distance scale"
            raise ValueError(msg)

        return 2.0 * (biased_potential.sigma * sigma_distance) ** 2
