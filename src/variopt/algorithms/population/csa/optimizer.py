"""CSA optimizer boundary adapter over the explicit CSA engine state."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Generic, Literal, TypeVar, cast

import numpy as np
from typing_extensions import override

from variopt.generic_runtime import FrozenGenericSlotsCompat

from ....artifacts import Observation, Proposal
from ....distance import require_valid_distance
from ....diversity import DiversityMetric
from ....execution import (
    EXACT_ASYNC_EXECUTION_MODEL,
    SEQUENTIAL_EXECUTION_MODEL,
    SYNC_BATCH_EXECUTION_MODEL,
    ExecutionModel,
)
from ....json_types import JSONDict, JSONValue
from ....kernel import ProposalLocalSearchContext
from ....methods import RunMethod
from ....outcomes import EvaluationOutcome
from ....randomness import RandomSeed, RandomStateSnapshot
from ....sampling import CandidateSampler
from ....spaces import LeafPath, SearchSpace
from ....spaces.projections import compile_homogeneous_numeric_subspace
from ....spaces.serialization import (
    space_candidate_from_dict,
    space_candidate_to_dict,
)
from ....spaces.structured import (
    StructuredSearchSpace,
    is_structured_candidate_space,
    require_space_candidate_value,
)
from ....spaces.types import SpaceCandidateValue
from ....typevars import CandidateT
from .banking.bank import Bank, BankEntry
from .banking.clustering import CSAClusteringState
from .banking.growth import CSABankGrowthState
from .banking.reference import ReferenceBank
from .banking.update import CSABankUpdatePolicy
from .defaults import derive_csa_defaults
from .engine import (
    CSABankingState,
    CSAEngineState,
    CSAPendingProposals,
    CSAScoringState,
    apply_pending_boundary_action,
    apply_tell,
    commit_materialized_generation,
    dequeue_generation_candidate,
    materialize_generation,
    plan_next_ask,
)
from .generation.perturbation import CSAPerturbationSchedule, CSAPerturbationSpec
from .generation.proposal import CSAProposalState
from .generation.proposal.covariance import infer_numeric_subspace_displacement
from .generation.proposal.logic import (
    infer_structured_local_displacement_leaf_paths,
    proposal_local_search_context,
    record_proposal_attribution,
)
from .generation.proposal.state.attribution import (
    NumericSubspaceDisplacement,
    PlannedProposalAttribution,
    ProposalAttribution,
)
from .generation.state import GenerationRuntimeState
from .profile import CSAProfile, CSAResolvedProfile
from .progression.cutoff.state import CSACutoffState
from .progression.stage import CSAStageState
from .progression.state import CSAProgressionState
from .scoring.acceptance_state import CSAAcceptanceState
from .scoring.model_state import CSAScoreModelState
from .selection.state import SeedSelectionState
from .trace.events.state import CSAEventTraceState

BoundaryT = TypeVar("BoundaryT")
OutcomeCandidateT = TypeVar("OutcomeCandidateT")
StructuredBoundaryT = TypeVar("StructuredBoundaryT")
StructuredCandidateT = TypeVar("StructuredCandidateT", bound=SpaceCandidateValue)


@dataclass(frozen=True, slots=True)
class CSAOptimizer(FrozenGenericSlotsCompat,
    RunMethod[
        CSAEngineState[CandidateT],
        Proposal[CandidateT],
        Observation[CandidateT],
    ],
    Generic[BoundaryT, CandidateT],
):
    """Stateless CSA optimizer over an explicit immutable engine state.

    Parameters
    ----------
    space : SearchSpace[BoundaryT, CandidateT]
        Search space from which CSA proposals are drawn.
    diversity_metric : DiversityMetric[CandidateT]
        Diversity metric used for bank distances and clustering.
    bank_capacity : int
        Initial capacity of the CSA bank and reference bank.
    profile : CSAProfile[CandidateT]
        Boundary-level CSA configuration.
    sampler : CandidateSampler[CandidateT] | None, optional
        Optional sampler used when CSA draws directly from the search space.
    random_state : RandomSeed, optional
        Seed or random-state object used to initialize optimizer randomness.
    """

    space: SearchSpace[BoundaryT, CandidateT]
    diversity_metric: DiversityMetric[CandidateT]
    bank_capacity: int
    profile: CSAProfile[CandidateT] = field(kw_only=True)
    sampler: CandidateSampler[CandidateT] | None = field(default=None, kw_only=True)
    random_state: RandomSeed = None
    resolved_profile: CSAResolvedProfile[CandidateT] = field(init=False, repr=False)
    bank_update_policy: CSABankUpdatePolicy = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Resolve and validate the canonical CSA configuration.

        Raises
        ------
        ValueError
            Raised when bank-capacity, growth, or perturbation settings are
            inconsistent.
        """
        resolved_profile = self.profile.resolve()
        object.__setattr__(self, "resolved_profile", resolved_profile)
        object.__setattr__(self, "bank_update_policy", resolved_profile.update_policy)

        if self.bank_capacity <= 0:
            msg = "bank_capacity must be positive"
            raise ValueError(msg)

        if (
            resolved_profile.max_bank_capacity is not None
            and resolved_profile.max_bank_capacity < self.bank_capacity
        ):
            msg = "max_bank_capacity must be at least bank_capacity"
            raise ValueError(msg)

        if (
            resolved_profile.growth_policy.enabled
            and resolved_profile.growth_policy.maximum_capacity is not None
            and resolved_profile.growth_policy.maximum_capacity < self.bank_capacity
        ):
            msg = "growth_policy.maximum_capacity must be at least bank_capacity"
            raise ValueError(msg)

        if (
            resolved_profile.growth_policy.enabled
            and resolved_profile.max_bank_capacity is not None
            and resolved_profile.max_bank_capacity > self.bank_capacity
        ):
            msg = "adaptive bank growth and staged bank growth must not both be enabled"
            raise ValueError(msg)

        if self.bank_capacity < self.max_family_arity(
            resolved_profile.perturbation_schedule.regular_family,
        ):
            msg = "bank_capacity must be at least the regular family arity"
            raise ValueError(msg)

        if self.bank_capacity < self.max_family_arity(
            resolved_profile.perturbation_schedule.initial_family,
        ):
            msg = "bank_capacity must be at least the initial family arity"
            raise ValueError(msg)

    @staticmethod
    def from_space_defaults(
        *,
        space: StructuredSearchSpace[StructuredBoundaryT, StructuredCandidateT],
        bank_capacity: int,
        profile: CSAProfile[StructuredCandidateT] | None = None,
        sampler: CandidateSampler[StructuredCandidateT] | None = None,
        diversity_metric: DiversityMetric[StructuredCandidateT] | None = None,
        perturbation_schedule: (
            CSAPerturbationSchedule[StructuredCandidateT] | None
        ) = None,
        random_state: RandomSeed = None,
    ) -> "CSAOptimizer[StructuredBoundaryT, StructuredCandidateT]":
        """Build a CSA optimizer from structured-space defaults and overrides.

        Parameters
        ----------
        space : StructuredSearchSpace[StructuredBoundaryT, StructuredCandidateT]
            Structured search space that defines leaf semantics and default CSA
            components.
        bank_capacity : int
            Initial capacity of the CSA bank and reference bank.
        profile : CSAProfile[StructuredCandidateT] | None, default=None
            Optional boundary-level CSA profile override.
        sampler : CandidateSampler[StructuredCandidateT] | None, default=None
            Optional sampler override for direct space sampling.
        diversity_metric : DiversityMetric[StructuredCandidateT] | None, default=None
            Optional diversity metric override.
        perturbation_schedule : CSAPerturbationSchedule[StructuredCandidateT] | None, default=None
            Optional perturbation schedule override.
        random_state : RandomSeed, optional
            Seed or random-state object used to initialize optimizer
            randomness.

        Returns
        -------
        CSAOptimizer[StructuredBoundaryT, StructuredCandidateT]
            CSA optimizer configured from space-derived defaults and explicit
            overrides.

        Notes
        -----
        Override precedence is:

        1. Explicit keyword arguments to this constructor
        2. Fields already present on the provided ``profile``
        3. Derived defaults from ``space``
        """
        default_style: Literal["variopt", "joung_2018"] = "variopt"
        if profile is not None and profile.preset == "joung_2018":
            default_style = "joung_2018"

        defaults = derive_csa_defaults(
            space,
            style=default_style,
        )
        effective_schedule = defaults.perturbation_schedule
        if perturbation_schedule is not None:
            effective_schedule = perturbation_schedule

        effective_profile = profile
        if effective_profile is None:
            effective_profile = CSAProfile(
                perturbation_schedule=effective_schedule,
            )
        elif perturbation_schedule is not None or effective_profile.perturbation_schedule is None:
            effective_profile = replace(
                effective_profile,
                perturbation_schedule=effective_schedule,
            )

        return CSAOptimizer(
            space=space,
            diversity_metric=(
                defaults.diversity_metric
                if diversity_metric is None
                else diversity_metric
            ),
            bank_capacity=bank_capacity,
            profile=effective_profile,
            sampler=defaults.sampler if sampler is None else sampler,
            random_state=random_state,
        )

    @staticmethod
    def max_family_arity(
        family: Sequence[CSAPerturbationSpec[CandidateT]],
    ) -> int:
        """Return the maximal operator arity across a perturbation family.

        Parameters
        ----------
        family : Sequence[CSAPerturbationSpec[CandidateT]]
            Perturbation family whose operator arities are inspected.

        Returns
        -------
        int
            Maximum operator arity present in ``family``.
        """
        if len(family) == 0:
            return 0

        return max(spec.operator.arity for spec in family)

    @override
    def create_initial_state(self) -> CSAEngineState[CandidateT]:
        """Create the initial immutable CSA engine state.

        Returns
        -------
        CSAEngineState[CandidateT]
            Fresh engine state rooted in this optimizer configuration.
        """
        return self.create_state()

    @override
    def supported_execution_models(self) -> frozenset[ExecutionModel]:
        """Return supported execution models.

        Returns
        -------
        frozenset[ExecutionModel]
            Execution models accepted by the CSA ask/tell contract.
        """
        return frozenset(
            {
                SEQUENTIAL_EXECUTION_MODEL,
                SYNC_BATCH_EXECUTION_MODEL,
                EXACT_ASYNC_EXECUTION_MODEL,
            },
        )

    def create_state(
        self,
        *,
        trace_state: CSAEventTraceState[CandidateT] | None = None,
    ) -> CSAEngineState[CandidateT]:
        """Create an explicit CSA engine state for this configuration.

        Parameters
        ----------
        trace_state : CSAEventTraceState[CandidateT] | None, default=None
            Optional event-trace state attached to the created engine state.

        Returns
        -------
        CSAEngineState[CandidateT]
            Fresh engine state rooted in this optimizer configuration.
        """
        progression_state = CSAProgressionState(
            cutoff_state=self.build_initial_cutoff_state(),
            stage_state=CSAStageState(
                base_capacity=self.bank_capacity,
                max_capacity=(
                    self.bank_capacity
                    if self.resolved_profile.max_bank_capacity is None
                    else self.resolved_profile.max_bank_capacity
                ),
            ),
            base_cycle_limit=self.resolved_profile.cycle_limit,
            restart_lite=self.resolved_profile.restart_lite,
        )
        return CSAEngineState(
            random_state=RandomStateSnapshot.from_seed(self.random_state),
            banking_state=CSABankingState(
                bank=Bank[CandidateT](capacity=self.bank_capacity),
                reference_bank=ReferenceBank[CandidateT](capacity=self.bank_capacity),
                refresh_state=None,
                growth_state=CSABankGrowthState[CandidateT].from_policy(
                    self.resolved_profile.growth_policy,
                ),
                clustering_state=CSAClusteringState[CandidateT](
                    policy=self.resolved_profile.clustering_policy,
                ),
            ),
            progression_state=progression_state,
            selection_state=SeedSelectionState(),
            generation_state=GenerationRuntimeState[CandidateT](),
            proposal_state=CSAProposalState.from_policy(
                self.resolved_profile.proposal_policy,
            ),
            scoring_state=CSAScoringState(
                acceptance_state=CSAAcceptanceState.from_policy(
                    self.resolved_profile.acceptance_policy,
                ),
                model_state=CSAScoreModelState(
                    score_model=self.resolved_profile.score_model,
                ),
            ),
            pending_proposals=CSAPendingProposals[CandidateT](),
            trace_state=trace_state,
        )

    def state_to_dict(
        self,
        state: CSAEngineState[CandidateT],
        *,
        candidate_to_dict: Callable[[CandidateT], JSONValue] | None = None,
    ) -> JSONDict:
        """Return a JSON-safe checkpoint snapshot for one engine state.

        Parameters
        ----------
        state : CSAEngineState[CandidateT]
            Engine state to checkpoint.
        candidate_to_dict : Callable[[CandidateT], JSONValue] | None, default=None
            Optional candidate serializer. When omitted, structured spaces use
            the built-in recursive codec.

        Returns
        -------
        JSONDict
            Versioned JSON-safe checkpoint snapshot.

        Raises
        ------
        TypeError
            If no candidate codec is available for this optimizer.
        ValueError
            If ``state`` is not at a safe checkpoint boundary.
        """
        serializer: Callable[[CandidateT], JSONValue] = (
            self._default_candidate_to_dict
            if candidate_to_dict is None
            else candidate_to_dict
        )
        return state.to_dict(candidate_to_dict=serializer)

    def state_from_dict(
        self,
        data: Mapping[str, JSONValue],
        *,
        candidate_from_dict: Callable[[JSONValue], CandidateT] | None = None,
    ) -> CSAEngineState[CandidateT]:
        """Return one engine state restored from a JSON-safe checkpoint snapshot.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            Versioned JSON-safe checkpoint snapshot.
        candidate_from_dict : Callable[[JSONValue], CandidateT] | None, default=None
            Optional candidate deserializer. When omitted, structured spaces
            use the built-in recursive codec.

        Returns
        -------
        CSAEngineState[CandidateT]
            Reconstructed engine state.

        Raises
        ------
        TypeError
            If no candidate codec is available for this optimizer.
        ValueError
            If the snapshot format is unsupported or the decoded candidates are
            incompatible with the optimizer space.
        """
        deserializer: Callable[[JSONValue], CandidateT] = (
            self._default_candidate_from_dict
            if candidate_from_dict is None
            else candidate_from_dict
        )
        return CSAEngineState[CandidateT].from_dict(
            data,
            candidate_from_dict=deserializer,
            growth_policy=self.resolved_profile.growth_policy,
            clustering_policy=self.resolved_profile.clustering_policy,
            proposal_policy=self.resolved_profile.proposal_policy,
            acceptance_policy=self.resolved_profile.acceptance_policy,
            score_model=self.resolved_profile.score_model,
        )

    def _default_candidate_to_dict(
        self,
        candidate: CandidateT,
    ) -> JSONValue:
        if not is_structured_candidate_space(self.space):
            msg = (
                "candidate_to_dict is required when checkpointing a CSA optimizer "
                "over a non-structured search space"
            )
            raise TypeError(msg)
        candidate_value = require_space_candidate_value(
            candidate,
            operation="CSA checkpoint serialization",
        )
        self.space.validate(candidate_value)
        return space_candidate_to_dict(candidate_value)

    def _default_candidate_from_dict(
        self,
        data: JSONValue,
    ) -> CandidateT:
        if not is_structured_candidate_space(self.space):
            msg = (
                "candidate_from_dict is required when restoring a CSA optimizer "
                "over a non-structured search space"
            )
            raise TypeError(msg)

        candidate = space_candidate_from_dict(data)
        self.space.validate(candidate)
        return cast(CandidateT, candidate)

    @override
    def is_exhausted(self, state: CSAEngineState[CandidateT]) -> bool:
        """Report whether the optimizer can emit more proposals.

        Parameters
        ----------
        state : CSAEngineState[CandidateT]
            Engine state to inspect.

        Returns
        -------
        bool
            ``True`` when the lifecycle marked the run exhausted.
        """
        return state.progression_state.is_exhausted

    @override
    def ask(
        self,
        state: CSAEngineState[CandidateT],
        batch_size: int = 1,
    ) -> tuple[tuple[Proposal[CandidateT], ...], CSAEngineState[CandidateT]]:
        """Emit the next batch of CSA proposals and advanced engine state.

        Parameters
        ----------
        state : CSAEngineState[CandidateT]
            Current immutable engine state.
        batch_size : int, default=1
            Maximum number of proposals to emit.

        Returns
        -------
        tuple[tuple[Proposal[CandidateT], ...], CSAEngineState[CandidateT]]
            Proposal batch together with the advanced immutable engine state.

        Raises
        ------
        ValueError
            Raised when ``batch_size`` is not positive.
        """
        if batch_size <= 0:
            msg = "batch_size must be positive"
            raise ValueError(msg)

        engine_state = state
        proposals: list[Proposal[CandidateT]] = []
        for _ in range(batch_size):
            (
                candidate,
                tracks_generation,
                planned_attribution,
                engine_state,
            ) = self.propose_candidate(engine_state)
            proposal_id, engine_state = engine_state.allocate_proposal_id()
            proposal = Proposal(candidate=candidate, proposal_id=proposal_id)
            engine_state = engine_state.issue_proposal(
                proposal,
                tracks_generation=tracks_generation,
            )
            if planned_attribution is not None:
                engine_state = replace(
                    engine_state,
                    proposal_state=record_proposal_attribution(
                        engine_state.proposal_state,
                        ProposalAttribution.from_planned(
                            proposal_id=proposal_id,
                            attribution=planned_attribution,
                        ),
                    ),
                )
            proposals.append(proposal)

            if (
                tracks_generation
                and engine_state.generation_state.queue.is_empty
                and len(proposals) < batch_size
            ):
                break

        return tuple(proposals), engine_state

    @override
    def tell(
        self,
        state: CSAEngineState[CandidateT],
        observations: Sequence[Observation[CandidateT]],
    ) -> CSAEngineState[CandidateT]:
        """Advance the CSA engine state with one observation batch.

        Parameters
        ----------
        state : CSAEngineState[CandidateT]
            Current immutable engine state.
        observations : Sequence[Observation[CandidateT]]
            Observations aligned with currently pending proposals.

        Returns
        -------
        CSAEngineState[CandidateT]
            Updated immutable engine state after score, bank, and lifecycle
            updates.
        """
        return self._tell_with_explicit_local_displacements(
            state,
            observations,
        )

    @override
    def tell_outcomes(
        self,
        state: CSAEngineState[CandidateT],
        outcomes: Sequence[EvaluationOutcome[OutcomeCandidateT, Observation[CandidateT]]],
    ) -> CSAEngineState[CandidateT]:
        """Advance CSA state with full outcomes when refinement metadata exists.

        Parameters
        ----------
        state : CSAEngineState[CandidateT]
            Current immutable engine state.
        outcomes : Sequence[EvaluationOutcome[OutcomeCandidateT, Observation[CandidateT]]]
            Evaluation outcomes aligned with currently pending proposals.

        Returns
        -------
        CSAEngineState[CandidateT]
            Updated immutable engine state after score, bank, and lifecycle
            updates.
        """
        observations: list[Observation[CandidateT]] = []
        explicit_paths: list[tuple[LeafPath, ...] | None] | None = None
        for outcome_index, outcome in enumerate(outcomes):
            observations.append(outcome.record)
            refinement = outcome.refinement
            if explicit_paths is not None:
                explicit_paths.append(
                    None if refinement is None else refinement.changed_leaf_paths
                )
            elif refinement is not None:
                explicit_paths = [None for _index in range(outcome_index)]
                explicit_paths.append(refinement.changed_leaf_paths)

        if explicit_paths is None:
            return self.tell(state, tuple(observations))

        return self._tell_with_explicit_local_displacements(
            state,
            tuple(observations),
            explicit_local_displacement_leaf_paths=tuple(explicit_paths),
        )

    def _tell_with_explicit_local_displacements(
        self,
        state: CSAEngineState[CandidateT],
        observations: Sequence[Observation[CandidateT]],
        *,
        explicit_local_displacement_leaf_paths: (
            Sequence[tuple[LeafPath, ...] | None] | None
        ) = None,
    ) -> CSAEngineState[CandidateT]:
        """Advance CSA state with optional explicit local-displacement paths."""
        local_displacement_leaf_path_inference: (
            Callable[[CandidateT, CandidateT], tuple[LeafPath, ...]] | None
        ) = None
        numeric_subspace_displacement_inference: (
            Callable[[ProposalAttribution, CandidateT], NumericSubspaceDisplacement | None] | None
        ) = None
        if is_structured_candidate_space(self.space):
            structured_space = self.space

            def infer_structured_displacement_paths(
                proposal_candidate: CandidateT,
                observed_candidate: CandidateT,
            ) -> tuple[LeafPath, ...]:
                proposal_candidate_value = require_space_candidate_value(
                    proposal_candidate,
                    operation="CSA local displacement inference",
                )
                observed_candidate_value = require_space_candidate_value(
                    observed_candidate,
                    operation="CSA local displacement inference",
                )
                return infer_structured_local_displacement_leaf_paths(
                    space=structured_space,
                    proposal_candidate=proposal_candidate_value,
                    observed_candidate=observed_candidate_value,
                )

            local_displacement_leaf_path_inference = infer_structured_displacement_paths

            def infer_structured_numeric_subspace_displacement(
                attribution: ProposalAttribution,
                observed_candidate: CandidateT,
            ) -> NumericSubspaceDisplacement | None:
                numeric_subspace_attribution = attribution.numeric_subspace_attribution
                if numeric_subspace_attribution is None:
                    return None

                descriptor = compile_homogeneous_numeric_subspace(
                    structured_space,
                    leaf_paths=numeric_subspace_attribution.leaf_paths,
                )
                if descriptor is None:
                    return None

                observed_candidate_value = require_space_candidate_value(
                    observed_candidate,
                    operation="CSA numeric-subspace displacement inference",
                )
                return infer_numeric_subspace_displacement(
                    descriptor=descriptor,
                    attribution=numeric_subspace_attribution,
                    observed_candidate=observed_candidate_value,
                )

            numeric_subspace_displacement_inference = (
                infer_structured_numeric_subspace_displacement
            )

        if not state.scoring_state.acceptance_state.requires_random_state:
            return apply_tell(
                state,
                observations,
                bank_capacity=self.bank_capacity,
                diversity_metric=self.diversity_metric,
                cutoff_schedule=self.resolved_profile.cutoff_schedule,
                refresh_policy=self.resolved_profile.refresh_policy,
                update_policy=self.bank_update_policy,
                infer_average_distance=self.infer_average_distance_for_entries,
                infer_score_gap=self.infer_score_gap_for_entries,
                infer_local_displacement_leaf_paths=local_displacement_leaf_path_inference,
                explicit_local_displacement_leaf_paths=explicit_local_displacement_leaf_paths,
                infer_numeric_subspace_displacement=numeric_subspace_displacement_inference,
            )

        next_engine_state, next_random_state = state.random_state.advance(
            lambda random_state: apply_tell(
                state,
                observations,
                bank_capacity=self.bank_capacity,
                diversity_metric=self.diversity_metric,
                cutoff_schedule=self.resolved_profile.cutoff_schedule,
                refresh_policy=self.resolved_profile.refresh_policy,
                update_policy=self.bank_update_policy,
                random_state=random_state,
                infer_average_distance=self.infer_average_distance_for_entries,
                infer_score_gap=self.infer_score_gap_for_entries,
                infer_local_displacement_leaf_paths=local_displacement_leaf_path_inference,
                explicit_local_displacement_leaf_paths=explicit_local_displacement_leaf_paths,
                infer_numeric_subspace_displacement=numeric_subspace_displacement_inference,
            ),
        )
        return next_engine_state.replace_random_state(next_random_state)

    @override
    def proposal_kernel_hints(
        self,
        state: CSAEngineState[CandidateT],
        proposals: Sequence[Proposal[CandidateT]],
    ) -> tuple[ProposalLocalSearchContext | None, ...] | None:
        """Return per-proposal local-search hints derived from CSA history.

        Parameters
        ----------
        state : CSAEngineState[CandidateT]
            Current immutable engine state.
        proposals : Sequence[Proposal[CandidateT]]
            Proposals for which local-search hints are requested.

        Returns
        -------
        tuple[ProposalLocalSearchContext | None, ...] | None
            Per-proposal local-search contexts, or ``None`` when no hints are
            available for the current state.
        """
        if not is_structured_candidate_space(self.space):
            return None

        proposal_state = state.proposal_state
        if not proposal_state.policy.enabled:
            return None

        leaf_paths = self.space.leaf_paths()
        contexts = tuple(
            proposal_local_search_context(
                state=proposal_state,
                leaf_paths=leaf_paths,
                attribution=(
                    proposal_state.get_pending_attribution(proposal.proposal_id)
                    if proposal.proposal_id is not None
                    else None
                ),
            )
            for proposal in proposals
        )
        if all(context is None for context in contexts):
            return None
        return contexts

    def propose_candidate(
        self,
        state: CSAEngineState[CandidateT],
    ) -> tuple[
        CandidateT,
        bool,
        PlannedProposalAttribution | None,
        CSAEngineState[CandidateT],
    ]:
        """Produce one CSA candidate and the advanced engine state.

        Parameters
        ----------
        state : CSAEngineState[CandidateT]
            Current immutable engine state.

        Returns
        -------
        tuple[CandidateT, bool, PlannedProposalAttribution | None, CSAEngineState[CandidateT]]
            Candidate, generation-tracking flag, optional planned attribution,
            and the advanced immutable engine state.

        Raises
        ------
        RuntimeError
            Raised when the engine is exhausted or blocked on a pending
            lifecycle boundary action.
        """
        engine_state = state
        if self.is_exhausted(engine_state):
            msg = "cannot ask for new proposals from an exhausted CSAOptimizer"
            raise RuntimeError(msg)

        if engine_state.progression_state.has_pending_action:
            if not engine_state.pending_proposals.is_empty:
                msg = (
                    "cannot ask for new proposals while a CSA run-boundary "
                    "transition is pending on outstanding proposals"
                )
                raise RuntimeError(msg)

            engine_state = apply_pending_boundary_action(
                engine_state,
                refresh_policy=self.resolved_profile.refresh_policy,
                diversity_metric=self.diversity_metric,
                cutoff_schedule=self.resolved_profile.cutoff_schedule,
                infer_average_distance=self.infer_average_distance_for_entries,
                infer_score_gap=self.infer_score_gap_for_entries,
            )

        ask_plan = plan_next_ask(engine_state)
        if ask_plan.kind in {"space_sample", "refresh_sample"}:
            if (
                ask_plan.kind == "refresh_sample"
                and engine_state.banking_state.refresh_state is not None
                and engine_state.banking_state.refresh_state.has_enough_entries
                and not engine_state.pending_proposals.is_empty
            ):
                msg = (
                    "cannot ask for new proposals while refresh completion is pending "
                    "on outstanding proposals"
                )
                raise RuntimeError(msg)

            candidate, next_random_state = engine_state.random_state.advance(
                self.sample_candidate,
            )
            self.space.validate(candidate)
            return (
                candidate,
                False,
                None,
                engine_state.replace_random_state(next_random_state),
            )

        if ask_plan.kind == "materialize_generation":
            materialized_generation, next_random_state = engine_state.random_state.advance(
                lambda random_state: materialize_generation(
                    engine_state=engine_state,
                    resolved_profile=self.resolved_profile,
                    space=self.space,
                    diversity_metric=self.diversity_metric,
                    random_state=random_state,
                ),
            )
            generated_candidate, next_engine_state = commit_materialized_generation(
                engine_state.replace_random_state(next_random_state),
                materialized_generation,
            )
            return (
                generated_candidate.candidate,
                True,
                generated_candidate.planned_attribution,
                next_engine_state,
            )

        generated_candidate, next_engine_state = dequeue_generation_candidate(engine_state)
        return (
            generated_candidate.candidate,
            True,
            generated_candidate.planned_attribution,
            next_engine_state,
        )

    def sample_candidate(self, random_state: np.random.RandomState) -> CandidateT:
        """Sample one candidate directly from the configured boundary sampler.

        Parameters
        ----------
        random_state : np.random.RandomState
            Random-state instance used for sampling.

        Returns
        -------
        CandidateT
            Candidate sampled from ``self.sampler`` or directly from the search
            space.
        """
        sampler = self.sampler
        if sampler is None:
            return self.space.sample(random_state)
        return sampler.sample(random_state)

    def build_initial_cutoff_state(self) -> CSACutoffState:
        """Build the initial cutoff state.

        Returns
        -------
        CSACutoffState
            Initial cutoff state provided by the configured cutoff schedule.
        """
        return self.resolved_profile.cutoff_schedule.build_initial_state()

    def infer_average_distance_for_entries(
        self,
        entries: Sequence[BankEntry[CandidateT]],
    ) -> float:
        """Estimate the average pairwise bank distance.

        Parameters
        ----------
        entries : Sequence[BankEntry[CandidateT]]
            Bank entries whose pairwise distances are summarized.

        Returns
        -------
        float
            Mean pairwise diversity distance, or ``0.0`` when fewer than two
            entries are present.
        """
        if len(entries) < 2:
            return 0.0

        distance_sum = 0.0
        pair_count = 0
        for left_index, left_entry in enumerate(entries[:-1]):
            for right_entry in entries[left_index + 1 :]:
                distance_sum += require_valid_distance(
                    self.diversity_metric.distance(
                        left_entry.candidate,
                        right_entry.candidate,
                    )
                )
                pair_count += 1

        if pair_count == 0:
            return 0.0

        return distance_sum / float(pair_count)

    def infer_score_gap_for_entries(
        self,
        entries: Sequence[BankEntry[CandidateT]],
    ) -> float | None:
        """Estimate the score spread across bank entries.

        Parameters
        ----------
        entries : Sequence[BankEntry[CandidateT]]
            Bank entries whose score spread is summarized.

        Returns
        -------
        float | None
            Difference between maximum and minimum objective value, or
            ``None`` when no entries are available.
        """
        if not entries:
            return None

        values = tuple(entry.value for entry in entries)
        return max(values) - min(values)
