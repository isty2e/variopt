"""Evaluation interface definitions."""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Generic

from ..execution import ExecutionResources, NestedParallelismPolicy
from ..typevars import EvaluationRequestT, EvaluationT, ProblemT


class Evaluator(ABC, Generic[ProblemT, EvaluationRequestT, EvaluationT]):
    """Batch execution service for turning requests into evaluation outcomes.

    Implementations execute one complete request batch and must return exactly
    one outcome per input request in the same order as the input sequence.
    Evaluators own execution backend concerns such as request-level parallelism.

    Notes
    -----
    Evaluators are request-batch executors. They do not own optimizer state or
    cross-episode search memory.
    """

    @abstractmethod
    def evaluate(
        self,
        problem: ProblemT,
        requests: Sequence[EvaluationRequestT],
    ) -> Sequence[EvaluationT]:
        """Execute one full request batch.

        Parameters
        ----------
        problem : ProblemT
            Problem that defines evaluation semantics.
        requests : Sequence[EvaluationRequestT]
            Request batch to execute.

        Returns
        -------
        Sequence[EvaluationT]
            Outcomes aligned one-to-one with ``requests`` in the same logical
            order.
        """

    def execution_resources(self) -> ExecutionResources:
        """Return canonical execution ownership metadata.

        Returns
        -------
        ExecutionResources
            Resource contract describing which layer owns parallelism for this
            evaluator.
        """
        return ExecutionResources(
            parallel_owner="evaluator",
            nested_parallelism_policy=NestedParallelismPolicy.FORBID,
        )
