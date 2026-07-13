"""Shared support for CSA white-box CSA tests."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, final

import numpy as np
import pytest
from typing_extensions import TypedDict, Unpack, override

from variopt import (
    DiversityMetric,
    EvaluationRequest,
    IntegerSpace,
    Objective,
    Observation,
    Problem,
    Proposal,
    SearchSpace,
    Study,
    VariationOperator,
)
from variopt.algorithms.population.csa import (
    Bank,
    CSAAcceptancePolicy,
    CSAAdaptivePotential,
    CSAAdaptivePotentialAxis,
    CSABankGrowthPolicy,
    CSABankUpdatePolicy,
    CSABiasedPotential,
    CSACutoffSchedule,
    CSANicheQualityPolicy,
    CSAOptimizer,
    CSAPerturbationSchedule,
    CSAPerturbationSpec,
    CSAProfile,
    CSAScoreModel,
)
from variopt.algorithms.population.csa.banking.bank import BankEntry
from variopt.algorithms.population.csa.banking.reference import (
    ReferenceBank,
)
from variopt.algorithms.population.csa.banking.update.admission import (
    admit_observation,
)
from variopt.algorithms.population.csa.banking.update.result import (
    significant_update_indices,
)
from variopt.algorithms.population.csa.engine import (
    CSAEngineState,
    begin_stage_transition,
)
from variopt.algorithms.population.csa.generation.proposal import CSAProposalPolicy
from variopt.algorithms.population.csa.progression.cutoff.logic import (
    advance_cutoff_state,
)
from variopt.algorithms.population.csa.progression.cutoff.state import (
    CSACutoffState,
)
from variopt.algorithms.population.csa.progression.stage import (
    CSAStageState,
)
from variopt.algorithms.population.csa.progression.state import (
    CSAProgressionState,
)
from variopt.algorithms.population.csa.scoring.model_state import (
    CSAScoreModelState,
)
from variopt.algorithms.population.csa.selection.policy import (
    prepare_seed_batch,
    select_partner_indices,
)
from variopt.algorithms.population.csa.selection.routing import (
    should_use_reference_primary,
)
from variopt.algorithms.population.csa.selection.state import (
    SeedSelectionState,
)
from variopt.evaluators import SequentialEvaluator

__all__ = [
    "AbsoluteDistance",
    "Bank",
    "BankEntry",
    "CSAAcceptancePolicy",
    "CSAAdaptivePotential",
    "CSAAdaptivePotentialAxis",
    "CSABankGrowthPolicy",
    "CSABankUpdatePolicy",
    "CSABiasedPotential",
    "CSACutoffSchedule",
    "CSACutoffState",
    "CSANicheQualityPolicy",
    "CSAOptimizer",
    "CSAOptimizerDriver",
    "CSAOptimizerKwargs",
    "CSAOptimizerTestCase",
    "CSAProfile",
    "CSAStageState",
    "CSAPerturbationSchedule",
    "CSAPerturbationSpec",
    "CSAScoreModel",
    "CollapseToZero",
    "DiversityMetric",
    "EncodeBinaryParents",
    "EvaluationRequest",
    "IntegerSpace",
    "LegacyPerturbationConfig",
    "NaNDistance",
    "NegativeDistance",
    "Objective",
    "Observation",
    "Problem",
    "Proposal",
    "RecordingTernaryParents",
    "ReferenceBank",
    "RepeatParent",
    "ScriptedIntegerSpace",
    "SearchSpace",
    "SeedSelectionState",
    "SequentialEvaluator",
    "SquareObjective",
    "Study",
    "VariationOperator",
    "admit_observation",
    "evaluate_observations",
    "make_optimizer",
    "perturbation_schedule",
    "prepare_seed_batch",
    "schedule",
    "select_partner_indices",
    "should_use_reference_primary",
    "significant_update_indices",
]


class SquareObjective(Objective[int]):
    """Toy objective with a minimum at zero."""

    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate * candidate)


class AbsoluteDistance(DiversityMetric[int]):
    """Absolute-value distance for integer candidates."""

    @override
    def distance(self, left: int, right: int) -> float:
        return float(abs(left - right))


class NaNDistance(DiversityMetric[int]):
    """Broken diversity metric that returns NaN."""

    @override
    def distance(self, left: int, right: int) -> float:
        _ = (left, right)
        return float("nan")


class NegativeDistance(DiversityMetric[int]):
    """Broken diversity metric that returns a negative distance."""

    @override
    def distance(self, left: int, right: int) -> float:
        _ = (left, right)
        return -1.0


class CollapseToZero(VariationOperator[int]):
    """Unary variation that deterministically collapses candidates to zero."""

    @property
    @override
    def arity(self) -> int:
        return 1

    @override
    def apply(
        self,
        parents: Sequence[int],
        random_state: np.random.RandomState,
    ) -> int:
        _ = (parents, random_state)
        return 0


class RepeatParent(VariationOperator[int]):
    """Unary variation that returns the first parent unchanged."""

    @property
    @override
    def arity(self) -> int:
        return 1

    @override
    def apply(
        self,
        parents: Sequence[int],
        random_state: np.random.RandomState,
    ) -> int:
        _ = random_state
        return parents[0]


class EncodeBinaryParents(VariationOperator[int]):
    """Binary variation that encodes the ordered parent pair into one integer."""

    @property
    @override
    def arity(self) -> int:
        return 2

    @override
    def apply(
        self,
        parents: Sequence[int],
        random_state: np.random.RandomState,
    ) -> int:
        _ = random_state
        return parents[0] * 100 + parents[1]


class RecordingTernaryParents(VariationOperator[int]):
    """Ternary variation that records the ordered parent triple."""

    def __init__(self) -> None:
        self.last_parents: tuple[int, ...] | None = None

    @property
    @override
    def arity(self) -> int:
        return 3

    @override
    def apply(
        self,
        parents: Sequence[int],
        random_state: np.random.RandomState,
    ) -> int:
        _ = random_state
        self.last_parents = tuple(parents)
        return parents[0]


@final
class ScriptedIntegerSpace(SearchSpace[int, int]):
    """Test search space that samples from a fixed script before falling back."""

    def __init__(self, scripted_candidates: Sequence[int]) -> None:
        self._scripted_candidates = list(scripted_candidates)

    @override
    def normalize(self, raw_candidate: int) -> int:
        self.validate(raw_candidate)
        return raw_candidate

    @override
    def validate(self, candidate: int) -> None:
        if isinstance(candidate, bool):
            msg = "candidate must not be a bool"
            raise TypeError(msg)

    @override
    def sample(self, random_state: np.random.RandomState) -> int:
        _ = random_state
        if not self._scripted_candidates:
            msg = "no scripted candidates remain"
            raise ValueError(msg)

        return self._scripted_candidates.pop(0)


def evaluate_observations(
    problem: Problem[int, int],
    evaluator: SequentialEvaluator[int, int],
    proposals: Sequence[Proposal[int]],
) -> tuple[Observation[int], ...]:
    return tuple(
        outcome.observation
        for outcome in evaluator.evaluate(
            problem,
            tuple(EvaluationRequest(proposal=proposal) for proposal in proposals),
        )
    )


@dataclass(frozen=True, slots=True)
class LegacyPerturbationConfig:
    regular_children_per_seed: int = 1
    initial_children_per_seed: int = 1
    mutation_children_per_operator: int = 1
    shuffle_children: bool = True

    def __post_init__(self) -> None:
        if self.regular_children_per_seed < 0:
            msg = "regular_children_per_seed must be non-negative"
            raise ValueError(msg)

        if self.initial_children_per_seed < 0:
            msg = "initial_children_per_seed must be non-negative"
            raise ValueError(msg)

        if self.mutation_children_per_operator < 0:
            msg = "mutation_children_per_operator must be non-negative"
            raise ValueError(msg)


class CSAOptimizerKwargs(TypedDict, total=False):
    seed_count: int
    initial_new_bank_cut: int
    random_seed_mode: int
    weighted_partner_selection: bool
    max_bank_capacity: int | None
    cutoff_schedule: CSACutoffSchedule
    acceptance_policy: CSAAcceptancePolicy
    growth_policy: CSABankGrowthPolicy
    restart_lite: bool
    cycle_limit: int
    update_policy: CSABankUpdatePolicy
    score_model: CSAScoreModel[int]
    proposal_policy: CSAProposalPolicy
    random_state: int | None


def schedule(
    *,
    initial_distance_cutoff: float | None = None,
    minimum_distance_cutoff: float | None = None,
    reduction_factor: float = 1.0,
    reduction_method: Literal["exponential", "linear"] = "exponential",
    stagnation_update_limit: int = 0,
    recover_steps: int = 0,
    recover_mode: Literal[
        "none",
        "score_gap_increase",
        "score_gap_decrease",
    ] = "none",
) -> CSACutoffSchedule:
    return CSACutoffSchedule(
        initial_distance_cutoff=initial_distance_cutoff,
        minimum_distance_cutoff=minimum_distance_cutoff,
        reduction_factor=reduction_factor,
        reduction_method=reduction_method,
        stagnation_update_limit=stagnation_update_limit,
        recover_steps=recover_steps,
        recover_mode=recover_mode,
    )


def perturbation_schedule(
    *,
    regular_children_per_seed: int = 1,
    initial_children_per_seed: int = 1,
    mutation_children_per_operator: int = 1,
    shuffle_children: bool = True,
) -> LegacyPerturbationConfig:
    return LegacyPerturbationConfig(
        regular_children_per_seed=regular_children_per_seed,
        initial_children_per_seed=initial_children_per_seed,
        mutation_children_per_operator=mutation_children_per_operator,
        shuffle_children=shuffle_children,
    )


def make_optimizer(
    *,
    space: SearchSpace[int, int],
    diversity_metric: DiversityMetric[int],
    bank_capacity: int,
    perturbation_schedule: LegacyPerturbationConfig
    | CSAPerturbationSchedule[int]
    | None = None,
    variation_operator: VariationOperator[int] | None = None,
    initial_variation_operator: VariationOperator[int] | None = None,
    mutation_operators: Sequence[VariationOperator[int]] = (),
    **kwargs: Unpack[CSAOptimizerKwargs],
) -> "CSAOptimizerDriver":
    if isinstance(perturbation_schedule, CSAPerturbationSchedule):
        canonical_schedule = perturbation_schedule
    else:
        schedule_config = (
            LegacyPerturbationConfig()
            if perturbation_schedule is None
            else perturbation_schedule
        )
        if (
            variation_operator is None
            and initial_variation_operator is None
            and len(mutation_operators) == 0
        ):
            msg = "at least one perturbation family must be configured"
            raise ValueError(msg)

        regular_family = ()
        if (
            variation_operator is not None
            and schedule_config.regular_children_per_seed > 0
        ):
            regular_family = (
                CSAPerturbationSpec(
                    operator=variation_operator,
                    count=schedule_config.regular_children_per_seed,
                ),
            )

        initial_family = ()
        if (
            initial_variation_operator is not None
            and schedule_config.initial_children_per_seed > 0
        ):
            initial_family = (
                CSAPerturbationSpec(
                    operator=initial_variation_operator,
                    count=schedule_config.initial_children_per_seed,
                ),
            )

        mutation_family = tuple(
            CSAPerturbationSpec(
                operator=operator,
                count=schedule_config.mutation_children_per_operator,
            )
            for operator in mutation_operators
            if schedule_config.mutation_children_per_operator > 0
        )
        canonical_schedule = CSAPerturbationSchedule(
            regular_family=regular_family,
            initial_family=initial_family,
            mutation_family=mutation_family,
            shuffle_children=schedule_config.shuffle_children,
        )

    seed_count = kwargs["seed_count"] if "seed_count" in kwargs else 1
    initial_new_bank_cut = (
        kwargs["initial_new_bank_cut"] if "initial_new_bank_cut" in kwargs else 2
    )
    random_seed_mode = kwargs["random_seed_mode"] if "random_seed_mode" in kwargs else 0
    weighted_partner_selection = (
        kwargs["weighted_partner_selection"]
        if "weighted_partner_selection" in kwargs
        else False
    )
    max_bank_capacity = (
        kwargs["max_bank_capacity"] if "max_bank_capacity" in kwargs else None
    )
    cutoff_schedule = (
        kwargs["cutoff_schedule"] if "cutoff_schedule" in kwargs else schedule()
    )
    acceptance_policy = (
        kwargs["acceptance_policy"]
        if "acceptance_policy" in kwargs
        else CSAAcceptancePolicy()
    )
    growth_policy = (
        kwargs["growth_policy"] if "growth_policy" in kwargs else CSABankGrowthPolicy()
    )
    restart_lite = kwargs["restart_lite"] if "restart_lite" in kwargs else True
    cycle_limit = kwargs["cycle_limit"] if "cycle_limit" in kwargs else 3
    update_policy = (
        kwargs["update_policy"] if "update_policy" in kwargs else CSABankUpdatePolicy()
    )
    score_model: CSAScoreModel[int] = (
        kwargs["score_model"] if "score_model" in kwargs else CSAScoreModel()
    )
    proposal_policy = (
        kwargs["proposal_policy"]
        if "proposal_policy" in kwargs
        else CSAProposalPolicy()
    )
    random_state = kwargs["random_state"] if "random_state" in kwargs else None
    preset = kwargs["preset"] if "preset" in kwargs else "joung_2018"

    profile = CSAProfile(
        perturbation_schedule=canonical_schedule,
        preset=preset,
        seed_count=seed_count,
        initial_new_bank_cut=initial_new_bank_cut,
        random_seed_mode=random_seed_mode,
        weighted_partner_selection=weighted_partner_selection,
        max_bank_capacity=max_bank_capacity,
        cutoff_schedule=cutoff_schedule,
        acceptance_policy=acceptance_policy,
        growth_policy=growth_policy,
        restart_lite=restart_lite,
        cycle_limit=cycle_limit,
        update_policy=update_policy,
        score_model=score_model,
        proposal_policy=proposal_policy,
    )

    return CSAOptimizerDriver.create(
        CSAOptimizer(
            space=space,
            diversity_metric=diversity_metric,
            bank_capacity=bank_capacity,
            profile=profile,
            random_state=random_state,
        )
    )


@dataclass(slots=True)
class CSAOptimizerDriver:
    """Stateful test driver over the stateless CSA optimizer contract."""

    optimizer: CSAOptimizer[int, int]
    engine_state: CSAEngineState[int]

    @classmethod
    def create(cls, optimizer: CSAOptimizer[int, int]) -> "CSAOptimizerDriver":
        return cls(optimizer=optimizer, engine_state=optimizer.create_initial_state())

    @property
    def bank_capacity(self) -> int:
        return self.optimizer.bank_capacity

    @property
    def is_exhausted(self) -> bool:
        return self.optimizer.is_exhausted(self.engine_state)

    @property
    def state(self) -> CSAProgressionState:
        return self.engine_state.progression_state

    @property
    def progression_state(self) -> CSAProgressionState:
        return self.engine_state.progression_state

    @property
    def bank(self) -> Bank[int]:
        return self.engine_state.banking_state.bank

    @bank.setter
    def bank(self, bank: Bank[int]) -> None:
        self.engine_state = replace(
            self.engine_state,
            banking_state=replace(self.engine_state.banking_state, bank=bank),
        )

    @property
    def reference_bank(self) -> ReferenceBank[int]:
        return self.engine_state.banking_state.reference_bank

    @reference_bank.setter
    def reference_bank(self, reference_bank: ReferenceBank[int]) -> None:
        self.engine_state = replace(
            self.engine_state,
            banking_state=replace(
                self.engine_state.banking_state,
                reference_bank=reference_bank,
            ),
        )

    @property
    def cutoff_state(self) -> CSACutoffState:
        return self.engine_state.progression_state.cutoff_state

    @cutoff_state.setter
    def cutoff_state(self, cutoff_state: CSACutoffState) -> None:
        self.engine_state = replace(
            self.engine_state,
            progression_state=self.engine_state.progression_state.replace_cutoff_state(
                cutoff_state,
            ),
        )

    @property
    def lifecycle_state(self) -> CSAProgressionState:
        return self.engine_state.progression_state

    @lifecycle_state.setter
    def lifecycle_state(self, lifecycle_state: CSAProgressionState) -> None:
        self.engine_state = replace(
            self.engine_state,
            progression_state=lifecycle_state,
        )

    @property
    def selection_state(self) -> SeedSelectionState:
        return self.engine_state.selection_state

    @selection_state.setter
    def selection_state(self, selection_state: SeedSelectionState) -> None:
        self.engine_state = replace(self.engine_state, selection_state=selection_state)

    @property
    def score_model_state(self) -> CSAScoreModelState[int]:
        return self.engine_state.scoring_state.model_state

    @property
    def pending_by_id(self) -> dict[str, Proposal[int]]:
        return {
            proposal.proposal_id: proposal
            for proposal in self.engine_state.pending_proposals.proposals
            if proposal.proposal_id is not None
        }

    @pending_by_id.setter
    def pending_by_id(self, pending: dict[str, Proposal[int]]) -> None:
        self.engine_state = replace(
            self.engine_state,
            pending_proposals=type(self.engine_state.pending_proposals)(
                proposals=tuple(pending.values()),
            ),
        )

    @property
    def bank_update_policy(self) -> CSABankUpdatePolicy:
        return self.optimizer.bank_update_policy

    def set_pending_proposals(self, proposals: Sequence[Proposal[int]]) -> None:
        """Register pending proposals by proposal identifier."""
        pending: dict[str, Proposal[int]] = {}
        for proposal in proposals:
            proposal_id = proposal.proposal_id
            if proposal_id is None:
                msg = "pending proposals must have proposal_id values"
                raise ValueError(msg)
            pending[proposal_id] = proposal
        self.pending_by_id = pending

    def ask(self, batch_size: int = 1) -> tuple[Proposal[int], ...]:
        proposals, self.engine_state = self.optimizer.ask(
            self.engine_state,
            batch_size=batch_size,
        )
        return proposals

    def tell(self, observations: Sequence[Observation[int]]) -> None:
        self.engine_state = self.optimizer.tell(self.engine_state, observations)

    def begin_stage_transition(self, transition: tuple[CSAStageState, bool]) -> None:
        self.engine_state = begin_stage_transition(self.engine_state, transition)

    def advance_state(self, *, unused_entry_count: int) -> bool:
        entries = self.engine_state.banking_state.bank.entries
        cutoff_schedule = self.optimizer.resolved_profile.cutoff_schedule
        if cutoff_schedule.requires_reduction_observation:
            msg = (
                "CSAOptimizerDriver.advance_state cannot synthesize adaptive "
                "cutoff observations"
            )
            raise AssertionError(msg)
        next_progression_state, cycle_increment = advance_cutoff_state(
            state=self.engine_state.progression_state,
            schedule=cutoff_schedule,
            score_gap=self.optimizer.infer_score_gap_for_entries(entries),
            unused_entry_count=unused_entry_count,
        )
        self.engine_state = replace(
            self.engine_state,
            progression_state=next_progression_state,
        )
        return cycle_increment


class CSAOptimizerTestCase:
    """Shared white-box helpers for CSA optimizer tests."""

    def fill_bank(
        self,
        *,
        optimizer: CSAOptimizerDriver,
        problem: Problem[int, int],
        evaluator: SequentialEvaluator[int, int],
    ) -> None:
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=optimizer.bank_capacity),
            )
        )

    def enter_refresh(
        self,
        *,
        optimizer: CSAOptimizerDriver,
        problem: Problem[int, int],
        evaluator: SequentialEvaluator[int, int],
    ) -> None:
        self.fill_bank(
            optimizer=optimizer,
            problem=problem,
            evaluator=evaluator,
        )
        for _ in range(8):
            if optimizer.state.refresh_in_progress:
                return

            optimizer.tell(
                evaluate_observations(
                    problem,
                    evaluator,
                    optimizer.ask(batch_size=optimizer.bank_capacity),
                )
            )

        pytest.fail(
            "failed to enter refresh mode within the expected number of generations"
        )

    def enter_stage_growth(
        self,
        *,
        optimizer: CSAOptimizerDriver,
        problem: Problem[int, int],
        evaluator: SequentialEvaluator[int, int],
    ) -> tuple[Proposal[int], ...]:
        self.fill_bank(
            optimizer=optimizer,
            problem=problem,
            evaluator=evaluator,
        )
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=1),
            )
        )
        optimizer.tell(
            evaluate_observations(
                problem,
                evaluator,
                optimizer.ask(batch_size=1),
            )
        )
        return optimizer.ask(batch_size=2)

    def prime_full_bank(
        self,
        *,
        optimizer: CSAOptimizerDriver,
        entries: tuple[BankEntry[int], ...],
        distance_cutoff: float,
    ) -> None:
        optimizer.bank = Bank(
            capacity=optimizer.bank_capacity,
            entries=entries,
        )
        optimizer.reference_bank = ReferenceBank(
            capacity=optimizer.bank_capacity,
            entries=entries,
        )
        optimizer.cutoff_state = CSACutoffState(
            distance_cutoff=distance_cutoff,
            minimum_distance_cutoff=distance_cutoff,
            cutoff_recover_limit=distance_cutoff,
        )
