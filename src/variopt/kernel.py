"""Kernel contracts and runtime artifacts for one-episode execution."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic, Protocol, TypeGuard

from typing_extensions import TypeVar, override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .artifacts import (
    EvaluationAttemptBatch,
    EvaluationRequest,
    Observation,
    Proposal,
    ProposalEvaluationSpec,
)
from .execution import (
    EvaluationBudget,
    ExecutionResources,
    NestedParallelismPolicy,
)
from .problem import Problem
from .randomness import RandomStateSnapshot
from .spaces import LeafPath

BoundaryT = TypeVar("BoundaryT")
CandidateT = TypeVar("CandidateT")
QueryEvaluationPayloadT = TypeVar(
    "QueryEvaluationPayloadT",
    default=Observation[CandidateT],
)
KernelQueryT = TypeVar("KernelQueryT")
KernelReportT = TypeVar("KernelReportT")
RequestLocalPayloadT = TypeVar("RequestLocalPayloadT")
RequestLocalPayloadT_co = TypeVar(
    "RequestLocalPayloadT_co",
    covariant=True,
)


class ProposalKernelHint(ABC):
    """Marker base class for immutable per-proposal kernel hints.

    Notes
    -----
    Concrete kernel families can define richer hint records while the generic
    query surface stays free of family-specific nouns.
    """


@dataclass(frozen=True, slots=True)
class ProposalLocalSearchContext(ProposalKernelHint):
    """Per-proposal hint for episode-local local search.

    Parameters
    ----------
    enabled : bool, default=True
        Whether local search is enabled for the associated proposal.
    local_budget : int | None, optional
        Optional per-proposal evaluation or step budget reserved for local
        search.
    prioritized_leaf_paths : tuple[LeafPath, ...], default=()
        Optional ordered subset of leaf paths to prioritize during structured
        local search.
    random_state_snapshot : RandomStateSnapshot | None, optional
        Optional episode-local random-state snapshot. Checkpointable run methods
        can provide this to make stochastic local-search episodes reproducible
        from serialized run-method state.

    Notes
    -----
    The run method may derive this context from cross-episode state, but the
    context itself is immutable and scoped to a single kernel episode.
    """

    enabled: bool = True
    local_budget: int | None = None
    prioritized_leaf_paths: tuple[LeafPath, ...] = ()
    random_state_snapshot: RandomStateSnapshot | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the local-search context.

        Raises
        ------
        ValueError
            If ``local_budget`` is non-positive or if
            ``prioritized_leaf_paths`` contains duplicates.
        """
        if self.local_budget is not None and self.local_budget <= 0:
            msg = "local_budget must be positive when provided"
            raise ValueError(msg)

        if (
            self.random_state_snapshot is not None
            and type(self.random_state_snapshot) is not RandomStateSnapshot
        ):
            msg = "random_state_snapshot must be a RandomStateSnapshot when provided"
            raise TypeError(msg)

        normalized_leaf_paths = tuple(
            tuple(path) for path in self.prioritized_leaf_paths
        )
        if len(set(normalized_leaf_paths)) != len(normalized_leaf_paths):
            msg = "prioritized_leaf_paths must not contain duplicates"
            raise ValueError(msg)

        object.__setattr__(self, "prioritized_leaf_paths", normalized_leaf_paths)


@dataclass(frozen=True, slots=True)
class ProposalBatchQuery(
    FrozenGenericSlotsCompat, Generic[BoundaryT, CandidateT, QueryEvaluationPayloadT]
):
    """Canonical kernel query over a proposal batch.

    Parameters
    ----------
    problem : Problem[BoundaryT, CandidateT, QueryEvaluationPayloadT]
        Problem that owns the proposals and evaluation semantics.
    proposals : tuple[Proposal[CandidateT], ...]
        Proposals to evaluate or refine during this kernel episode.
    execution_resources : ExecutionResources
        Request-local execution ownership and worker-budget contract.
    proposal_evaluation_specs : tuple[ProposalEvaluationSpec | None, ...] | None, optional
        Optional request-local metadata aligned one-to-one with ``proposals``.
    proposal_kernel_hints : tuple[ProposalKernelHint | None, ...] | None, optional
        Optional per-proposal kernel hints aligned one-to-one with
        ``proposals``.
    evaluation_budget : EvaluationBudget | None, optional
        Shared runtime ledger for hard evaluation budgeting. Kernels may inspect
        or consume this ledger before issuing evaluator work.

    Notes
    -----
    Concrete hint semantics belong to specific kernel families. This query type
    only preserves alignment and ownership.
    """

    problem: Problem[BoundaryT, CandidateT, QueryEvaluationPayloadT]
    proposals: tuple[Proposal[CandidateT], ...]
    execution_resources: ExecutionResources
    proposal_evaluation_specs: tuple[ProposalEvaluationSpec | None, ...] | None = None
    proposal_kernel_hints: tuple[ProposalKernelHint | None, ...] | None = None
    evaluation_budget: EvaluationBudget | None = None

    def __post_init__(self) -> None:
        """Validate aligned per-proposal metadata.

        Raises
        ------
        ValueError
            If evaluation specs or kernel hints do not align one-to-one with
            ``proposals``.
        """
        if self.proposal_evaluation_specs is not None and (
            len(self.proposal_evaluation_specs) != len(self.proposals)
        ):
            msg = "proposal_evaluation_specs must align one-to-one with proposals"
            raise ValueError(msg)

        if self.proposal_kernel_hints is None:
            return

        if len(self.proposal_kernel_hints) != len(self.proposals):
            msg = "proposal_kernel_hints must align one-to-one with proposals"
            raise ValueError(msg)


class Kernel(ABC, Generic[KernelQueryT, KernelReportT]):
    """Run one bounded kernel episode.

    Notes
    -----
    Kernels may call the supplied runner multiple times inside a single
    episode, but they must not own cross-episode search memory. Persistent
    optimizer state belongs to the enclosing run method.
    """

    @abstractmethod
    def run(
        self,
        query: KernelQueryT,
        runner: Callable[[KernelQueryT], KernelReportT],
    ) -> KernelReportT:
        """Run one kernel episode.

        Parameters
        ----------
        query : KernelQueryT
            Canonical query for the episode.
        runner : Callable[[KernelQueryT], KernelReportT]
            Callback that evaluates the query at the kernel's chosen points.

        Returns
        -------
        KernelReportT
            Canonical report for the completed episode.
        """


class RequestLocalEvaluationRunner(
    Protocol[CandidateT, RequestLocalPayloadT_co],
):
    """Worker-local bounded objective runner for one request-local episode."""

    @property
    def remaining_evaluations(self) -> int:
        """Return objective calls still available to the current episode."""
        ...

    @property
    def consumed_evaluations(self) -> int:
        """Return objective calls consumed by the current episode."""
        ...

    def evaluate(
        self,
        request: EvaluationRequest[CandidateT],
    ) -> EvaluationAttemptBatch[CandidateT, RequestLocalPayloadT_co]:
        """Evaluate one canonical request within the episode's hard limit."""
        ...


class RequestLocalEpisodeKernel(
    Kernel[
        ProposalBatchQuery[BoundaryT, CandidateT, RequestLocalPayloadT],
        EvaluationAttemptBatch[CandidateT, RequestLocalPayloadT],
    ],
    Generic[BoundaryT, CandidateT, RequestLocalPayloadT],
):
    """Nominal capability for a serial episode scoped to one request.

    Notes
    -----
    Implementations must be referentially transparent with respect to kernel
    configuration. Randomness must come from the episode's authoritative
    snapshot rather than mutable kernel-owned state.
    """

    @abstractmethod
    def preferred_request_local_evaluation_limit(
        self,
        *,
        problem: Problem[BoundaryT, CandidateT, RequestLocalPayloadT],
        request: EvaluationRequest[CandidateT],
        proposal_kernel_hint: ProposalKernelHint | None,
    ) -> int | None:
        """Return the preferred objective-call limit for one request.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, RequestLocalPayloadT]
            Problem that owns candidate and evaluation semantics.
        request : EvaluationRequest[CandidateT]
            Canonical top-level request for the episode.
        proposal_kernel_hint : ProposalKernelHint | None
            Optional family-specific request-local hint.

        Returns
        -------
        int | None
            Positive finite objective-call limit preferred by this kernel, or
            ``None`` when the kernel cannot state a safe finite limit and must
            retain coordinator execution.
        """

    @abstractmethod
    def run_request_local_episode(
        self,
        *,
        problem: Problem[BoundaryT, CandidateT, RequestLocalPayloadT],
        episode: "RequestLocalEpisode[BoundaryT, CandidateT, RequestLocalPayloadT]",
        runner: RequestLocalEvaluationRunner[CandidateT, RequestLocalPayloadT],
    ) -> EvaluationAttemptBatch[CandidateT, RequestLocalPayloadT]:
        """Run one request-local episode through a bounded worker runner.

        Parameters
        ----------
        problem : Problem[BoundaryT, CandidateT, RequestLocalPayloadT]
            Worker-local problem instance.
        episode : RequestLocalEpisode[BoundaryT, CandidateT, RequestLocalPayloadT]
            Immutable work item for the request.
        runner : RequestLocalEvaluationRunner[CandidateT, RequestLocalPayloadT]
            Worker-local objective runner bounded by ``episode.evaluation_limit``.

        Returns
        -------
        EvaluationAttemptBatch[CandidateT, RequestLocalPayloadT]
            Exactly one top-level attempt for the completed episode.
        """


def is_explicit_request_local_episode_kernel(
    kernel: Kernel[
        ProposalBatchQuery[BoundaryT, CandidateT, RequestLocalPayloadT],
        EvaluationAttemptBatch[CandidateT, RequestLocalPayloadT],
    ],
) -> TypeGuard[RequestLocalEpisodeKernel[BoundaryT, CandidateT, RequestLocalPayloadT]]:
    """Return whether ``kernel`` explicitly implements the episode capability.

    A subclass that merely inherits a built-in implementation is deliberately
    excluded. Such subclasses may override observable ``run`` behavior and
    must retain coordinator execution until they explicitly implement both
    request-local methods.
    """
    if not isinstance(kernel, RequestLocalEpisodeKernel):
        return False

    implementation_namespace = type(kernel).__dict__
    return (
        "preferred_request_local_evaluation_limit" in implementation_namespace
        and "run_request_local_episode" in implementation_namespace
    )


@dataclass(frozen=True, slots=True)
class RequestLocalEpisode(
    FrozenGenericSlotsCompat,
    Generic[BoundaryT, CandidateT, RequestLocalPayloadT],
):
    """Immutable work item for one evaluator-owned kernel episode.

    Parameters
    ----------
    request_index : int
        Non-negative batch-local request slot.
    request : EvaluationRequest[CandidateT]
        Canonical top-level request.
    kernel : RequestLocalEpisodeKernel[BoundaryT, CandidateT, RequestLocalPayloadT]
        Explicitly eligible request-local kernel configuration.
    proposal_kernel_hint : ProposalKernelHint | None
        Optional family-specific request-local hint.
    random_state_snapshot : RandomStateSnapshot | None
        Authoritative random-state snapshot for the episode.
    evaluation_limit : int
        Positive hard objective-call limit.
    execution_resources : ExecutionResources
        Worker-local execution ownership metadata.
    """

    request_index: int
    request: EvaluationRequest[CandidateT]
    kernel: RequestLocalEpisodeKernel[BoundaryT, CandidateT, RequestLocalPayloadT]
    proposal_kernel_hint: ProposalKernelHint | None
    random_state_snapshot: RandomStateSnapshot | None
    evaluation_limit: int
    execution_resources: ExecutionResources

    def __post_init__(self) -> None:
        """Validate canonical request-local episode fields."""
        if type(self.request_index) is not int:
            msg = "request_index must be an exact integer"
            raise TypeError(msg)
        if self.request_index < 0:
            msg = "request_index must be non-negative"
            raise ValueError(msg)
        if type(self.request) is not EvaluationRequest:
            msg = "request must be an EvaluationRequest"
            raise TypeError(msg)
        if self.random_state_snapshot is not None and (
            type(self.random_state_snapshot) is not RandomStateSnapshot
        ):
            msg = "random_state_snapshot must be a RandomStateSnapshot when provided"
            raise TypeError(msg)
        if type(self.evaluation_limit) is not int:
            msg = "evaluation_limit must be an exact integer"
            raise TypeError(msg)
        if self.evaluation_limit <= 0:
            msg = "evaluation_limit must be positive"
            raise ValueError(msg)
        if type(self.execution_resources) is not ExecutionResources:
            msg = "execution_resources must be ExecutionResources"
            raise TypeError(msg)
        if self.execution_resources.parallel_owner != "evaluator":
            msg = "request-local episode execution must be evaluator-owned"
            raise ValueError(msg)
        if (
            self.execution_resources.nested_parallelism_policy
            is not NestedParallelismPolicy.FORBID
        ):
            msg = "request-local episodes must forbid nested parallelism"
            raise ValueError(msg)
        if (
            isinstance(self.proposal_kernel_hint, ProposalLocalSearchContext)
            and self.proposal_kernel_hint.random_state_snapshot is not None
        ):
            msg = (
                "request-local episode random state must be owned only by "
                "random_state_snapshot"
            )
            raise ValueError(msg)


class DirectKernel(Kernel[KernelQueryT, KernelReportT]):
    """Trivial kernel that delegates directly to the supplied runner.

    Notes
    -----
    This kernel is the baseline execution path when no proposal-local search or
    refinement should occur between ``ask`` and evaluation.
    """

    @override
    def run(
        self,
        query: KernelQueryT,
        runner: Callable[[KernelQueryT], KernelReportT],
    ) -> KernelReportT:
        """Return the direct runner result.

        Parameters
        ----------
        query : KernelQueryT
            Query to hand directly to ``runner``.
        runner : Callable[[KernelQueryT], KernelReportT]
            Callback that performs the actual work.

        Returns
        -------
        KernelReportT
            Result returned by ``runner(query)``.
        """
        return runner(query)
