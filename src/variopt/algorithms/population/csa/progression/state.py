"""CSA progression aggregate state definitions."""

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field, replace
from typing import Literal

from typing_extensions import Self

from .....json_types import (
    JSONDict,
    JSONValue,
    require_json_bool,
    require_json_int,
    require_json_list,
    require_json_mapping,
)
from .cutoff.state import CSACutoffState
from .stage import CSAStageState

BoundaryActionKind = Literal["refresh", "stage_transition"]


@dataclass(frozen=True, slots=True)
class PendingBoundaryAction:
    """Pending action to apply at the next CSA run boundary.

    Parameters
    ----------
    kind : BoundaryActionKind
        Kind of pending run-boundary action.
    stage_transition : tuple[CSAStageState, bool] | None, default=None
        Stage-transition payload when ``kind`` is ``\"stage_transition\"``.
    """

    kind: BoundaryActionKind
    stage_transition: tuple[CSAStageState, bool] | None = None

    def __post_init__(self) -> None:
        """Reject malformed pending boundary actions."""
        if self.kind == "refresh":
            if self.stage_transition is not None:
                msg = "refresh actions must not carry a stage transition"
                raise ValueError(msg)
            return

        if self.kind == "stage_transition" and self.stage_transition is None:
            msg = "stage_transition actions must carry a stage transition payload"
            raise ValueError(msg)

    @classmethod
    def refresh(cls) -> Self:
        """Return a plain refresh action.

        Returns
        -------
        Self
            Pending action requesting a refresh at the next run boundary.
        """
        return cls(kind="refresh")

    @classmethod
    def stage_transition_action(
        cls,
        transition: tuple[CSAStageState, bool],
    ) -> Self:
        """Return a stage-transition action.

        Parameters
        ----------
        transition : tuple[CSAStageState, bool]
            Next stage state together with the accompanying refresh flag.

        Returns
        -------
        Self
            Pending stage-transition action.
        """
        return cls(
            kind="stage_transition",
            stage_transition=transition,
        )

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the pending boundary action.

        Returns
        -------
        JSONDict
            JSON-safe pending-action snapshot.
        """
        return {
            "kind": self.kind,
            "stage_transition": (
                None
                if self.stage_transition is None
                else {
                    "stage_state": self.stage_transition[0].to_dict(),
                    "refresh_required": self.stage_transition[1],
                }
            ),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
    ) -> Self:
        """Build a pending boundary action from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe pending-action snapshot.

        Returns
        -------
        Self
            Reconstructed pending boundary action.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        raw_kind = data.get("kind")
        raw_stage_transition = data.get("stage_transition")
        if raw_kind == "refresh":
            kind: BoundaryActionKind = "refresh"
        elif raw_kind == "stage_transition":
            kind = "stage_transition"
        else:
            msg = "pending boundary action snapshot requires valid kind"
            raise TypeError(msg)

        if raw_stage_transition is None:
            return cls(kind=kind)

        stage_transition_data = require_json_mapping(
            raw_stage_transition,
            field_name="stage_transition",
        )
        raw_stage_state = require_json_mapping(
            stage_transition_data.get("stage_state"),
            field_name="stage_transition.stage_state",
        )
        refresh_required = require_json_bool(
            stage_transition_data.get("refresh_required"),
            field_name="stage_transition.refresh_required",
        )
        return cls(
            kind=kind,
            stage_transition=(
                CSAStageState.from_dict(raw_stage_state),
                refresh_required,
            ),
        )


@dataclass(frozen=True, slots=True)
class CSAProgressionState:
    """Canonical progression aggregate for CSA cutoff and run-boundary state.

    Parameters
    ----------
    cutoff_state : CSACutoffState, default=CSACutoffState()
        Cutoff scheduling state.
    stage_state : CSAStageState, default=CSAStageState(base_capacity=1, max_capacity=1)
        Stage/bank-growth state.
    base_cycle_limit : int, default=3
        Default cycle limit before a run-boundary action is requested.
    restart_lite : bool, default=True
        Whether restart-lite refresh is used after the final stage.
    pending_action : PendingBoundaryAction | None, default=None
        Run-boundary action waiting to be applied.
    is_exhausted : bool, default=False
        Whether progression has no further stage or refresh actions available.
    stage_transition_count : int, default=0
        Number of applied stage transitions.
    refresh_count : int, default=0
        Number of completed refresh actions.
    refresh_mask : frozenset[int], default=frozenset()
        Temporary newcomer mask active during refresh handling.
    """

    cutoff_state: CSACutoffState = field(default_factory=CSACutoffState)
    stage_state: CSAStageState = field(
        default_factory=lambda: CSAStageState(base_capacity=1, max_capacity=1),
    )
    base_cycle_limit: int = 3
    restart_lite: bool = True
    pending_action: PendingBoundaryAction | None = None
    is_exhausted: bool = False
    stage_transition_count: int = 0
    refresh_count: int = 0
    refresh_mask: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        """Reject invalid progression definitions."""
        if self.base_cycle_limit < 0:
            msg = "base_cycle_limit must be non-negative"
            raise ValueError(msg)

        if self.stage_transition_count < 0:
            msg = "stage_transition_count must be non-negative"
            raise ValueError(msg)

        if self.refresh_count < 0:
            msg = "refresh_count must be non-negative"
            raise ValueError(msg)

        if any(index < 0 for index in self.refresh_mask):
            msg = "refresh_mask must contain only non-negative indices"
            raise ValueError(msg)

        if self.pending_action is not None and self.is_exhausted:
            msg = "an exhausted progression state must not carry a pending action"
            raise ValueError(msg)

    @property
    def iteration_count(self) -> int:
        """Return the cutoff-runtime iteration count."""
        return self.cutoff_state.iteration_count

    @property
    def cycle_count(self) -> int:
        """Return the cutoff-runtime cycle count."""
        return self.cutoff_state.cycle_count

    @property
    def distance_cutoff(self) -> float | None:
        """Return the active distance cutoff."""
        return self.cutoff_state.distance_cutoff

    @property
    def minimum_distance_cutoff(self) -> float | None:
        """Return the active minimum distance cutoff."""
        return self.cutoff_state.minimum_distance_cutoff

    @property
    def cutoff_recover_limit(self) -> float | None:
        """Return the active cutoff recovery ceiling."""
        return self.cutoff_state.cutoff_recover_limit

    @property
    def previous_score_gap(self) -> float | None:
        """Return the previous score-gap value used by recovery logic."""
        return self.cutoff_state.previous_score_gap

    @property
    def refresh_in_progress(self) -> bool:
        """Return whether cutoff runtime is currently in refresh mode."""
        return self.cutoff_state.refresh_in_progress

    @property
    def cutoff_is_initialized(self) -> bool:
        """Return whether cutoff scheduling has been initialized."""
        return self.cutoff_state.cutoff_is_initialized

    @property
    def cutoff_at_minimum(self) -> bool:
        """Return whether the active cutoff has reached its minimum."""
        return self.cutoff_state.cutoff_at_minimum

    @property
    def current_cycle_limit(self) -> int:
        """Return the active cycle limit for the current stage."""
        if self.stage_state.stage_index > 0 and self.stage_state.stage_round == 0:
            return 0

        return self.base_cycle_limit

    @property
    def has_pending_action(self) -> bool:
        """Return whether a run-boundary action is waiting to be applied."""
        return self.pending_action is not None

    def to_dict(self) -> JSONDict:
        """Return a JSON-safe mapping for the progression state.

        Returns
        -------
        JSONDict
            JSON-safe progression-state snapshot.
        """
        return {
            "cutoff_state": self.cutoff_state.to_dict(),
            "stage_state": self.stage_state.to_dict(),
            "base_cycle_limit": self.base_cycle_limit,
            "restart_lite": self.restart_lite,
            "pending_action": (
                None
                if self.pending_action is None
                else self.pending_action.to_dict()
            ),
            "is_exhausted": self.is_exhausted,
            "stage_transition_count": self.stage_transition_count,
            "refresh_count": self.refresh_count,
            "refresh_mask": list(self.refresh_mask),
        }

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, JSONValue],
    ) -> Self:
        """Build a progression state from a JSON-safe mapping.

        Parameters
        ----------
        data : Mapping[str, JSONValue]
            JSON-safe progression-state snapshot.

        Returns
        -------
        Self
            Reconstructed progression state.

        Raises
        ------
        TypeError
            If the snapshot carries invalid field types.
        """
        raw_cutoff_state = require_json_mapping(
            data.get("cutoff_state"),
            field_name="cutoff_state",
        )
        raw_stage_state = require_json_mapping(
            data.get("stage_state"),
            field_name="stage_state",
        )
        base_cycle_limit = require_json_int(
            data.get("base_cycle_limit"),
            field_name="base_cycle_limit",
        )
        restart_lite = require_json_bool(
            data.get("restart_lite"),
            field_name="restart_lite",
        )
        raw_pending_action = data.get("pending_action")
        is_exhausted = require_json_bool(
            data.get("is_exhausted"),
            field_name="is_exhausted",
        )
        stage_transition_count = require_json_int(
            data.get("stage_transition_count"),
            field_name="stage_transition_count",
        )
        refresh_count = require_json_int(
            data.get("refresh_count"),
            field_name="refresh_count",
        )
        raw_refresh_mask = require_json_list(
            data.get("refresh_mask"),
            field_name="refresh_mask",
        )
        pending_action_data: JSONDict | None = None
        if raw_pending_action is not None:
            pending_action_data = require_json_mapping(
                raw_pending_action,
                field_name="pending_action",
            )

        refresh_mask: list[int] = []
        for raw_index in raw_refresh_mask:
            if not isinstance(raw_index, int):
                msg = "progression-state snapshot refresh_mask values must be integers"
                raise TypeError(msg)
            refresh_mask.append(raw_index)

        return cls(
            cutoff_state=CSACutoffState.from_dict(raw_cutoff_state),
            stage_state=CSAStageState.from_dict(raw_stage_state),
            base_cycle_limit=base_cycle_limit,
            restart_lite=restart_lite,
            pending_action=(
                None
                if pending_action_data is None
                else PendingBoundaryAction.from_dict(pending_action_data)
            ),
            is_exhausted=is_exhausted,
            stage_transition_count=stage_transition_count,
            refresh_count=refresh_count,
            refresh_mask=frozenset(refresh_mask),
        )

    @property
    def seed_mask(self) -> frozenset[int]:
        """Return the active seed mask across stage and refresh subdomains."""
        return self.stage_state.seed_mask | self.refresh_mask

    @property
    def partner_mask(self) -> frozenset[int]:
        """Return the active partner mask across stage and refresh subdomains."""
        return self.stage_state.partner_mask | self.refresh_mask

    def replace_cutoff_state(self, cutoff_state: CSACutoffState) -> Self:
        """Return a copy with one replacement cutoff-runtime state.

        Parameters
        ----------
        cutoff_state : CSACutoffState
            Replacement cutoff scheduling state.

        Returns
        -------
        Self
            Progression state with ``cutoff_state`` replaced.
        """
        return replace(self, cutoff_state=cutoff_state)

    def initialize_cutoff(
        self,
        *,
        distance_cutoff: float,
        minimum_distance_cutoff: float,
        previous_score_gap: float | None = None,
    ) -> Self:
        """Return a copy with cutoff scheduling initialized.

        Parameters
        ----------
        distance_cutoff : float
            Initial active distance cutoff.
        minimum_distance_cutoff : float
            Floor for the active distance cutoff.
        previous_score_gap : float | None, default=None
            Previous score-gap value used by cutoff recovery logic.

        Returns
        -------
        Self
            Progression state with initialized cutoff scheduling.
        """
        return replace(
            self,
            cutoff_state=self.cutoff_state.initialize_cutoff(
                distance_cutoff=distance_cutoff,
                minimum_distance_cutoff=minimum_distance_cutoff,
                previous_score_gap=previous_score_gap,
            ),
        )

    def advance_iteration(
        self,
        *,
        distance_cutoff: float | None = None,
        cycle_increment: bool = False,
        cutoff_recover_limit: float | None = None,
        previous_score_gap: float | None = None,
    ) -> Self:
        """Return the next progression state after one cutoff iteration.

        Parameters
        ----------
        distance_cutoff : float | None, default=None
            Optional replacement active distance cutoff.
        cycle_increment : bool, default=False
            Whether to increment the cycle counter.
        cutoff_recover_limit : float | None, default=None
            Optional cutoff recovery ceiling.
        previous_score_gap : float | None, default=None
            Optional previous score-gap value for recovery logic.

        Returns
        -------
        Self
            Progression state after advancing the cutoff runtime by one
            iteration.
        """
        return replace(
            self,
            cutoff_state=self.cutoff_state.advance_iteration(
                distance_cutoff=distance_cutoff,
                cycle_increment=cycle_increment,
                cutoff_recover_limit=cutoff_recover_limit,
                previous_score_gap=previous_score_gap,
            ),
        )

    def begin_refresh(self) -> Self:
        """Return a state that has entered restart-lite refresh mode.

        Returns
        -------
        Self
            Progression state with refresh bookkeeping activated and the
            transient refresh mask cleared.
        """
        return replace(
            self,
            cutoff_state=self.cutoff_state.begin_refresh(),
            refresh_mask=frozenset(),
        )

    def complete_refresh(
        self,
        *,
        distance_cutoff: float,
        minimum_distance_cutoff: float,
        previous_score_gap: float | None = None,
    ) -> Self:
        """Return a state that has completed refresh with a new cutoff schedule.

        Parameters
        ----------
        distance_cutoff : float
            Reinitialized active distance cutoff.
        minimum_distance_cutoff : float
            Reinitialized minimum distance cutoff.
        previous_score_gap : float | None, default=None
            Previous score-gap value used by cutoff recovery logic.

        Returns
        -------
        Self
            Progression state with refresh completed and the refresh mask
            cleared.
        """
        return replace(
            self,
            cutoff_state=self.cutoff_state.complete_refresh(
                distance_cutoff=distance_cutoff,
                minimum_distance_cutoff=minimum_distance_cutoff,
                previous_score_gap=previous_score_gap,
            ),
            refresh_mask=frozenset(),
        )

    def without_updated_seed_mask(self, updated_indices: AbstractSet[int]) -> Self:
        """Return a copy with updated indices removed from the seed mask.

        Parameters
        ----------
        updated_indices : collections.abc.Set[int]
            Indices updated during the current run step.

        Returns
        -------
        Self
            Progression state with those indices removed from stage and refresh
            seed masks.
        """
        if not updated_indices:
            return self

        return replace(
            self,
            stage_state=self.stage_state.without_updated_seed_mask(updated_indices),
            refresh_mask=frozenset(
                index for index in self.refresh_mask if index not in updated_indices
            ),
        )

    def with_refresh_mask(self, refresh_mask: frozenset[int]) -> Self:
        """Return a copy with one replacement refresh newcomer mask.

        Parameters
        ----------
        refresh_mask : frozenset[int]
            Replacement transient newcomer mask.

        Returns
        -------
        Self
            Progression state with the refresh mask replaced.
        """
        return replace(self, refresh_mask=refresh_mask)

    def clear_refresh_mask(self) -> Self:
        """Return a copy with the transient refresh mask cleared.

        Returns
        -------
        Self
            Progression state with an empty transient refresh mask.
        """
        if not self.refresh_mask:
            return self

        return replace(self, refresh_mask=frozenset())

    def request_boundary(self) -> Self:
        """Return a copy with the next run-boundary action scheduled if needed.

        Returns
        -------
        Self
            Progression state with a newly scheduled boundary action when the
            cycle limit has been exceeded.
        """
        if (
            self.pending_action is not None
            or self.is_exhausted
            or self.cycle_count <= self.current_cycle_limit
        ):
            return self

        next_transition = self.stage_state.next_transition()
        if next_transition is None:
            if not self.restart_lite:
                return replace(self, is_exhausted=True)

            pending_action = PendingBoundaryAction.refresh()
        else:
            pending_action = PendingBoundaryAction.stage_transition_action(
                next_transition,
            )

        return replace(self, pending_action=pending_action)

    def consume_pending_action(self) -> tuple[PendingBoundaryAction, Self]:
        """Return and clear the pending boundary action.

        Returns
        -------
        tuple[PendingBoundaryAction, Self]
            Pending action together with the cleared progression state.

        Raises
        ------
        ValueError
            If no pending action is available.
        """
        if self.pending_action is None:
            msg = "no pending boundary action is available"
            raise ValueError(msg)

        return self.pending_action, replace(self, pending_action=None)

    def apply_stage_transition(self, transition: tuple[CSAStageState, bool]) -> Self:
        """Return a copy whose active stage has been updated.

        Parameters
        ----------
        transition : tuple[CSAStageState, bool]
            Stage-transition payload returned by ``CSAStageState.next_transition``.

        Returns
        -------
        Self
            Progression state with the new stage installed and transition count
            incremented.
        """
        next_stage_state, _ = transition
        return replace(
            self,
            stage_state=next_stage_state,
            stage_transition_count=self.stage_transition_count + 1,
            refresh_mask=frozenset(),
        )

    def record_refresh(self) -> Self:
        """Return a copy with one refresh counted.

        Returns
        -------
        Self
            Progression state with ``refresh_count`` incremented.
        """
        return replace(self, refresh_count=self.refresh_count + 1)
