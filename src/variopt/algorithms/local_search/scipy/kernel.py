"""SciPy-backed kernel implementation for continuous local search."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....artifacts import (
    CandidateRefinement,
    Observation,
    Proposal,
    ProposalEvaluationSpec,
)
from ....kernel import (
    Kernel,
    KernelDiagnostics,
    KernelStatus,
    ProposalBatchQuery,
    ProposalKernelHint,
    ProposalLocalSearchContext,
)
from ....outcomes import EvaluationOutcome
from ....spaces.projections import ContinuousStructuredSpaceCodec
from ....spaces.types import SpaceCandidateValue
from .contracts import ScipyMinimizeMethod
from .results import ScipyMinimizeResult
from .runner import run_scipy_minimize

BoundaryT = TypeVar("BoundaryT")
ContinuousCandidateT = TypeVar("ContinuousCandidateT", bound=SpaceCandidateValue)


def _candidate_refinement_from_codec(
    *,
    codec: ContinuousStructuredSpaceCodec[BoundaryT, ContinuousCandidateT],
    source_candidate: ContinuousCandidateT,
    refined_candidate: ContinuousCandidateT,
) -> CandidateRefinement[ContinuousCandidateT] | None:
    """Return candidate-refinement provenance from continuous codec topology.

    Parameters
    ----------
    codec : ContinuousStructuredSpaceCodec[BoundaryT, ContinuousCandidateT]
        Codec that owns the real-valued structured leaf paths.
    source_candidate : ContinuousCandidateT
        Candidate before SciPy-backed local optimization.
    refined_candidate : ContinuousCandidateT
        Candidate returned by SciPy-backed local optimization.

    Returns
    -------
    CandidateRefinement[ContinuousCandidateT] | None
        Refinement payload with changed leaf paths, or ``None`` when the final
        optimized candidate preserves every structured leaf value.
    """
    codec.space.validate(source_candidate)
    codec.space.validate(refined_candidate)
    changed_leaf_paths = tuple(
        path
        for path in codec.leaf_paths
        if codec.space.leaf_value_at_path(source_candidate, path)
        != codec.space.leaf_value_at_path(refined_candidate, path)
    )
    if len(changed_leaf_paths) == 0:
        return None

    return CandidateRefinement(
        source_candidate=source_candidate,
        refined_candidate=refined_candidate,
        changed_leaf_paths=changed_leaf_paths,
    )


def _as_local_search_context(
    hint: ProposalKernelHint | None,
) -> ProposalLocalSearchContext | None:
    """Return one local-search context after validating the kernel hint type."""
    if hint is None:
        return None

    if not isinstance(hint, ProposalLocalSearchContext):
        msg = "SciPy local-search kernel requires ProposalLocalSearchContext hints"
        raise TypeError(msg)

    return hint


@dataclass(frozen=True, slots=True)
class ScipyMinimizeKernel(FrozenGenericSlotsCompat,
    Kernel[
        ProposalBatchQuery[
            BoundaryT,
            ContinuousCandidateT,
            Observation[ContinuousCandidateT],
        ],
        tuple[
            EvaluationOutcome[
                ContinuousCandidateT,
                Observation[ContinuousCandidateT],
            ],
            ...,
        ],
    ],
    Generic[BoundaryT, ContinuousCandidateT],
):
    """SciPy ``minimize`` kernel for continuous structured search spaces.

    Parameters
    ----------
    method : ScipyMinimizeMethod, default="L-BFGS-B"
        SciPy minimization method used for each proposal-local optimization
        episode.
    tolerance : float | None, default=None
        Optional termination tolerance forwarded to SciPy.
    max_iterations : int | None, default=None
        Optional global iteration budget forwarded to SciPy, unless a proposal
        context overrides it.

    Notes
    -----
    This kernel currently supports only structured spaces whose leaves are all
    ``RealSpace`` instances. It evaluates the objective in the space's
    coordinate system so that log-scaled real variables are optimized in log
    coordinates rather than raw value space.
    """

    method: ScipyMinimizeMethod = "L-BFGS-B"
    tolerance: float | None = None
    max_iterations: int | None = None

    def __post_init__(self) -> None:
        """Validate SciPy adapter boundary settings.

        Raises
        ------
        ValueError
            Raised when the configured method, tolerance, or iteration limit is
            invalid.
        """
        if self.method not in {"L-BFGS-B", "Powell"}:
            msg = "method must be 'L-BFGS-B' or 'Powell'"
            raise ValueError(msg)

        if self.tolerance is not None and self.tolerance <= 0.0:
            msg = "tolerance must be positive when provided"
            raise ValueError(msg)

        if self.max_iterations is not None and self.max_iterations <= 0:
            msg = "max_iterations must be positive when provided"
            raise ValueError(msg)

    def scipy_options(self) -> dict[str, int]:
        """Return SciPy options derived from kernel settings.

        Returns
        -------
        dict[str, int]
            SciPy options dictionary containing the configured iteration cap
            when one is set.
        """
        if self.max_iterations is None:
            return {}
        return {"maxiter": self.max_iterations}

    def _proposal_context(
        self,
        *,
        query: ProposalBatchQuery[BoundaryT, ContinuousCandidateT],
        proposal_index: int,
    ) -> ProposalLocalSearchContext | None:
        """Return the canonical local-search context for one proposal index."""
        if query.proposal_kernel_hints is None:
            return None
        hint = query.proposal_kernel_hints[proposal_index]
        return _as_local_search_context(hint)

    def _evaluate_original_proposal(
        self,
        *,
        query: ProposalBatchQuery[BoundaryT, ContinuousCandidateT],
        proposal: Proposal[ContinuousCandidateT],
        proposal_evaluation_spec: ProposalEvaluationSpec | None,
        runner: Callable[
            [ProposalBatchQuery[BoundaryT, ContinuousCandidateT]],
            tuple[EvaluationOutcome[ContinuousCandidateT], ...],
        ],
    ) -> EvaluationOutcome[ContinuousCandidateT]:
        """Evaluate one original proposal once without local search."""
        local_outcomes = runner(
            ProposalBatchQuery(
                problem=query.problem,
                proposals=(proposal,),
                execution_resources=query.execution_resources,
                proposal_evaluation_specs=(
                    None
                    if proposal_evaluation_spec is None
                    else (proposal_evaluation_spec,)
                ),
            ),
        )
        if len(local_outcomes) != 1:
            msg = "kernel runner must return exactly one outcome for one proposal"
            raise ValueError(msg)

        local_outcome = local_outcomes[0]
        return EvaluationOutcome(
            record=local_outcome.record,
            evaluation_count=local_outcome.evaluation_count,
            kernel_diagnostics=KernelDiagnostics(
                backend="scipy.optimize.minimize",
                method=self.method,
                status=KernelStatus.STOPPED,
                message="local search disabled by run-method context",
            ),
        )

    def _scipy_options(
        self,
        *,
        context: ProposalLocalSearchContext | None,
    ) -> dict[str, int]:
        """Return SciPy options after one per-proposal budget override."""
        max_iterations = self.max_iterations
        if context is not None and context.local_budget is not None:
            max_iterations = context.local_budget

        if max_iterations is None:
            return {}
        return {"maxiter": max_iterations}

    def _evaluate_candidate(
        self,
        *,
        query: ProposalBatchQuery[BoundaryT, ContinuousCandidateT],
        candidate: ContinuousCandidateT,
        proposal_evaluation_spec: ProposalEvaluationSpec | None,
        runner: Callable[
            [ProposalBatchQuery[BoundaryT, ContinuousCandidateT]],
            tuple[EvaluationOutcome[ContinuousCandidateT], ...],
        ],
    ) -> EvaluationOutcome[ContinuousCandidateT]:
        """Evaluate one candidate through the supplied evaluator runner."""
        local_outcomes = runner(
            ProposalBatchQuery(
                problem=query.problem,
                proposals=(Proposal(candidate=candidate),),
                execution_resources=query.execution_resources,
                proposal_evaluation_specs=(
                    None
                    if proposal_evaluation_spec is None
                    else (proposal_evaluation_spec,)
                ),
            ),
        )
        if len(local_outcomes) != 1:
            msg = "kernel runner must return exactly one outcome for one proposal"
            raise ValueError(msg)
        return local_outcomes[0]

    def _optimize_proposal(
        self,
        *,
        query: ProposalBatchQuery[BoundaryT, ContinuousCandidateT],
        proposal_index: int,
        proposal: Proposal[ContinuousCandidateT],
        runner: Callable[
            [ProposalBatchQuery[BoundaryT, ContinuousCandidateT]],
            tuple[EvaluationOutcome[ContinuousCandidateT], ...],
        ],
    ) -> EvaluationOutcome[ContinuousCandidateT]:
        """Run one local descent episode for one original proposal."""
        context = self._proposal_context(query=query, proposal_index=proposal_index)
        proposal_evaluation_spec = (
            None
            if query.proposal_evaluation_specs is None
            else query.proposal_evaluation_specs[proposal_index]
        )
        if context is not None and not context.enabled:
            return self._evaluate_original_proposal(
                query=query,
                proposal=proposal,
                proposal_evaluation_spec=proposal_evaluation_spec,
                runner=runner,
            )

        codec = ContinuousStructuredSpaceCodec[
            BoundaryT,
            ContinuousCandidateT,
        ].from_space(query.problem.space)
        initial_coordinates = codec.coordinates_from_candidate(proposal.candidate)
        evaluation_count = 0

        def objective_in_coordinate_space(
            coordinates: Sequence[float],
        ) -> float:
            nonlocal evaluation_count
            local_candidate = codec.candidate_from_coordinates(
                proposal.candidate,
                coordinates,
            )
            local_outcome = self._evaluate_candidate(
                query=query,
                candidate=local_candidate,
                proposal_evaluation_spec=proposal_evaluation_spec,
                runner=runner,
            )
            evaluation_count += local_outcome.evaluation_count
            return local_outcome.record.value

        scipy_result = ScipyMinimizeResult.from_optimize_result(
            run_scipy_minimize(
                objective_in_coordinate_space=objective_in_coordinate_space,
                initial_coordinates=initial_coordinates,
                method=self.method,
                coordinate_bounds=codec.coordinate_bounds,
                tolerance=self.tolerance,
                options=self._scipy_options(context=context),
            )
        )
        optimized_candidate = codec.candidate_from_coordinates(
            proposal.candidate,
            scipy_result.coordinates,
        )
        refinement = _candidate_refinement_from_codec(
            codec=codec,
            source_candidate=proposal.candidate,
            refined_candidate=optimized_candidate,
        )
        return EvaluationOutcome(
            record=Observation.from_objective_value(
                proposal=proposal,
                proposal_evaluation_spec=proposal_evaluation_spec,
                candidate=optimized_candidate,
                value=scipy_result.objective_value,
                direction=query.problem.direction,
            ),
            evaluation_count=evaluation_count,
            kernel_diagnostics=scipy_result.diagnostics(method=self.method),
            refinement=refinement,
        )

    @override
    def run(
        self,
        query: ProposalBatchQuery[BoundaryT, ContinuousCandidateT],
        runner: Callable[
            [ProposalBatchQuery[BoundaryT, ContinuousCandidateT]],
            tuple[EvaluationOutcome[ContinuousCandidateT], ...],
        ],
    ) -> tuple[EvaluationOutcome[ContinuousCandidateT], ...]:
        """Run proposal-local SciPy minimization for each proposal in a batch.

        Parameters
        ----------
        query : ProposalBatchQuery[BoundaryT, ContinuousCandidateT]
            Proposal batch and evaluation context to optimize.
        runner : Callable[[ProposalBatchQuery[BoundaryT, ContinuousCandidateT]], tuple[EvaluationOutcome[ContinuousCandidateT], ...]]
            Evaluator runner used to score proposals during local search.

        Returns
        -------
        tuple[EvaluationOutcome[ContinuousCandidateT], ...]
            Locally improved outcomes aligned to ``query.proposals``.
        """
        return tuple(
            self._optimize_proposal(
                query=query,
                proposal_index=proposal_index,
                proposal=proposal,
                runner=runner,
            )
            for proposal_index, proposal in enumerate(query.proposals)
        )
