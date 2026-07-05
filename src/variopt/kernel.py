"""Kernel contracts and runtime artifacts for one-episode execution."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Generic

from typing_extensions import TypeVar, override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from .artifacts import (
    Observation,
    Proposal,
    ProposalEvaluationSpec,
)
from .execution import EvaluationBudget, ExecutionResources
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
