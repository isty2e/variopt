"""Request-local kernel episode execution contracts and reference helpers."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import (
    Generic,
    NoReturn,
    Protocol,
    SupportsIndex,
    TypeAlias,
    cast,
    runtime_checkable,
)

from typing_extensions import TypeVar

from ..artifacts import EvaluationAttemptBatch, EvaluationRequest
from ..artifacts.alignment import validate_aligned_attempts
from ..evaluation_pipeline import evaluate_request_attempt
from ..execution import EvaluationBudgetExhausted
from ..kernel import RequestLocalEpisode
from ..problem import Problem

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT")
PayloadT = TypeVar("PayloadT")
RequestLocalEpisodeAttemptResult: TypeAlias = tuple[
    int,
    EvaluationAttemptBatch[CandidateT, PayloadT],
]


@runtime_checkable
class RequestLocalEpisodeEvaluator(
    Protocol[BoundaryT, CandidateT, PayloadT],
):
    """Optional evaluator capability for ordered request-local episodes."""

    def evaluate_request_local_episodes(
        self,
        problem: Problem[BoundaryT, CandidateT, PayloadT],
        episodes: Sequence[RequestLocalEpisode[BoundaryT, CandidateT, PayloadT]],
    ) -> EvaluationAttemptBatch[CandidateT, PayloadT]:
        """Execute request-local episodes into one ordered attempt batch."""
        ...


@dataclass(slots=True)
class BoundedRequestLocalEvaluationRunner(
    Generic[BoundaryT, CandidateT, PayloadT],
):
    """Worker-local objective runner bounded to one episode's hard limit.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, PayloadT]
        Worker-local problem instance.
    evaluation_limit : int
        Positive maximum number of objective calls.
    """

    problem: Problem[BoundaryT, CandidateT, PayloadT]
    evaluation_limit: int
    _consumed_evaluations: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        """Validate the worker-local hard limit."""
        if type(self.evaluation_limit) is not int:
            msg = "evaluation_limit must be an exact integer"
            raise TypeError(msg)
        if self.evaluation_limit <= 0:
            msg = "evaluation_limit must be positive"
            raise ValueError(msg)

    @property
    def remaining_evaluations(self) -> int:
        """Return objective calls still available to this episode.

        Returns
        -------
        int
            Remaining objective-call capacity.
        """
        return self.evaluation_limit - self._consumed_evaluations

    @property
    def consumed_evaluations(self) -> int:
        """Return objective calls consumed by this episode.

        Returns
        -------
        int
            Number of calls admitted through :meth:`evaluate`.
        """
        return self._consumed_evaluations

    def evaluate(
        self,
        request: EvaluationRequest[CandidateT],
    ) -> EvaluationAttemptBatch[CandidateT, PayloadT]:
        """Evaluate one request through the canonical attempt boundary.

        Parameters
        ----------
        request : EvaluationRequest[CandidateT]
            Canonical request to evaluate.

        Returns
        -------
        EvaluationAttemptBatch[CandidateT, PayloadT]
            One-slot success or recorded objective failure.

        Raises
        ------
        EvaluationBudgetExhausted
            If the episode has no remaining objective-call capacity.
        RuntimeError
            If the canonical request pipeline violates its one-call accounting
            contract.
        """
        if self.remaining_evaluations == 0:
            msg = "request-local evaluation limit exhausted"
            raise EvaluationBudgetExhausted(msg)

        self._consumed_evaluations += 1
        attempt = evaluate_request_attempt(
            problem=self.problem,
            request=request,
        )
        if attempt.attempt_count != 1 or attempt.evaluation_count != 1:
            msg = "canonical request evaluation must report one consumed unit"
            raise RuntimeError(msg)
        return attempt

    def __reduce_ex__(self, protocol: SupportsIndex) -> NoReturn:
        """Reject serialization of worker-local mutable execution state."""
        del protocol
        msg = (
            "BoundedRequestLocalEvaluationRunner is worker-local and cannot be pickled"
        )
        raise TypeError(msg)


def ordered_request_local_episodes(
    episodes: Sequence[RequestLocalEpisode[BoundaryT, CandidateT, PayloadT]],
) -> tuple[RequestLocalEpisode[BoundaryT, CandidateT, PayloadT], ...]:
    """Validate and return episodes in contiguous request-index order.

    Parameters
    ----------
    episodes : Sequence[RequestLocalEpisode[BoundaryT, CandidateT, PayloadT]]
        Episode batch in arbitrary transport order.

    Returns
    -------
    tuple[RequestLocalEpisode[BoundaryT, CandidateT, PayloadT], ...]
        Episodes sorted by ``request_index``.

    Raises
    ------
    TypeError
        If the sequence contains a non-episode value.
    ValueError
        If request indices are duplicated or non-contiguous.
    """
    for episode in episodes:
        if type(episode) is not RequestLocalEpisode:
            msg = "episodes must contain RequestLocalEpisode values"
            raise TypeError(msg)

    ordered_episodes = tuple(
        sorted(
            episodes,
            key=lambda episode: episode.request_index,
        ),
    )
    for expected_index, episode in enumerate(ordered_episodes):
        if episode.request_index != expected_index:
            msg = "episode request indices must be unique and contiguous from zero"
            raise ValueError(msg)

    return ordered_episodes


def ordered_request_local_episode_attempts(
    results: Sequence[object],
    *,
    request_count: int,
) -> EvaluationAttemptBatch[CandidateT, PayloadT]:
    """Validate indexed worker results and materialize logical request order.

    Parameters
    ----------
    results : Sequence[object]
        Untrusted indexed one-slot attempt batches returned by workers.
    request_count : int
        Number of episodes submitted for the logical batch.

    Returns
    -------
    EvaluationAttemptBatch[CandidateT, PayloadT]
        Attempt slots ordered by contiguous request index.

    Raises
    ------
    TypeError
        If a worker returns a non-integer request index or a non-attempt batch.
    ValueError
        If result count or request-index alignment is invalid.
    """
    if type(request_count) is not int:
        msg = "request_count must be an exact integer"
        raise TypeError(msg)
    if request_count < 0:
        msg = "request_count must be non-negative"
        raise ValueError(msg)
    if len(results) != request_count:
        msg = "request-local episode results must match the submitted request count"
        raise ValueError(msg)

    validated_results: list[RequestLocalEpisodeAttemptResult[CandidateT, PayloadT]] = []
    for result in results:
        if type(result) is not tuple or len(result) != 2:
            msg = "request-local episode result must be an indexed attempt pair"
            raise TypeError(msg)
        request_index, attempt = result
        if type(request_index) is not int:
            msg = "request-local episode result index must be an exact integer"
            raise TypeError(msg)
        if type(attempt) is not EvaluationAttemptBatch:
            msg = "request-local episode result must contain an attempt batch"
            raise TypeError(msg)
        # Runtime validation recovers the nominal batch; generic arguments are
        # erased by the untyped Joblib transport boundary.
        validated_attempt = cast(
            EvaluationAttemptBatch[CandidateT, PayloadT],
            attempt,
        )
        validated_results.append((request_index, validated_attempt))

    ordered_results = tuple(
        sorted(validated_results, key=lambda result: result[0]),
    )
    for expected_index, (request_index, _) in enumerate(ordered_results):
        if request_index != expected_index:
            msg = "request-local episode results do not align with request indices"
            raise ValueError(msg)

    return EvaluationAttemptBatch[
        CandidateT,
        PayloadT,
    ].from_single_request_attempts(attempt for _, attempt in ordered_results)


def execute_request_local_episode(
    *,
    problem: Problem[BoundaryT, CandidateT, PayloadT],
    episode: RequestLocalEpisode[BoundaryT, CandidateT, PayloadT],
) -> EvaluationAttemptBatch[CandidateT, PayloadT]:
    """Execute and validate one request-local episode.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, PayloadT]
        Worker-local problem instance.
    episode : RequestLocalEpisode[BoundaryT, CandidateT, PayloadT]
        Immutable request-local work item.

    Returns
    -------
    EvaluationAttemptBatch[CandidateT, PayloadT]
        Exactly one top-level attempt whose cost matches worker-local
        consumption.

    Raises
    ------
    ValueError
        If the kernel returns other than one attempt, reports zero cost,
        exceeds its limit, or disagrees with runner consumption.

    Notes
    -----
    Exceptions raised by the episode kernel are intentionally not captured.
    They remain execution failures rather than objective-failure attempts.
    """
    runner = BoundedRequestLocalEvaluationRunner(
        problem=problem,
        evaluation_limit=episode.evaluation_limit,
    )
    attempt = episode.kernel.run_request_local_episode(
        problem=problem,
        episode=episode,
        runner=runner,
    )
    if attempt.attempt_count != 1:
        msg = "request-local episode must return exactly one top-level attempt"
        raise ValueError(msg)
    if attempt.evaluation_count <= 0:
        msg = "completed request-local episode must report positive evaluation cost"
        raise ValueError(msg)
    if attempt.evaluation_count > episode.evaluation_limit:
        msg = "request-local episode cost exceeds its evaluation limit"
        raise ValueError(msg)
    if attempt.evaluation_count != runner.consumed_evaluations:
        msg = "request-local episode cost disagrees with worker-local consumption"
        raise ValueError(msg)
    validate_aligned_attempts(
        (episode.request,),
        attempt,
        candidate_equal=problem.space.candidates_equal,
    )

    return attempt
