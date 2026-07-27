"""Adversarial tests for synchronous Joblib worker-session transport."""

import os
import pickle
import stat
import tempfile
import time
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from mmap import mmap
from pathlib import Path
from secrets import randbits
from typing import Literal, cast

import numpy as np
import pytest
from typing_extensions import Never, override

from variopt import (
    EvaluationAttemptBatch,
    EvaluationRequest,
    IntegerSpace,
    Kernel,
    Objective,
    Observation,
    Problem,
    Proposal,
    ProposalBatchQuery,
    RunMethod,
    Study,
)
from variopt.artifacts import ObservationPayload
from variopt.evaluators import JoblibEvaluator
from variopt.evaluators.joblib.session import (
    JoblibProblemSnapshotTransport,
    JoblibWorkerSession,
)
from variopt.evaluators.joblib.sync import JoblibProblemTransportMode
from variopt.evaluators.joblib.worker_context import (
    SESSION_TOKEN_BYTES,
    decode_problem_envelope,
)


class OffsetObjective(Objective[int]):
    """Return a candidate plus a fixed offset."""

    def __init__(self, offset: float, *, delay: float = 0.0) -> None:
        self.offset = offset
        self.delay = delay

    @override
    def evaluate(self, candidate: int) -> float:
        if self.delay > 0.0:
            time.sleep(self.delay)
        return float(candidate) + self.offset


class ExplodingObjective(Objective[int]):
    """Raise for one candidate to exercise user-failure capture."""

    @override
    def evaluate(self, candidate: int) -> float:
        if candidate == 2:
            raise ValueError("objective exploded")
        return float(candidate)


class UnserializableObjective(Objective[int]):
    """Reject cloudpickle serialization before any task can be submitted."""

    def __reduce__(self) -> Never:
        raise TypeError("objective cannot be serialized")

    @override
    def evaluate(self, candidate: int) -> float:
        return float(candidate)


class DecodeIdentityObjective(Objective[int]):
    """Expose one process-local nonce assigned whenever the problem is decoded."""

    def __init__(self) -> None:
        self.decode_nonce = 0

    def __getstate__(self) -> dict[str, int]:
        return {"decode_nonce": 0}

    def __setstate__(self, state: dict[str, int]) -> None:
        _ = state
        self.decode_nonce = randbits(52)

    @override
    def evaluate(self, candidate: int) -> float:
        time.sleep(0.02)
        return float(self.decode_nonce)


class ProcessIdentityObjective(Objective[int]):
    """Expose the worker process identifier as an objective value."""

    @override
    def evaluate(self, candidate: int) -> float:
        _ = candidate
        time.sleep(0.02)
        return float(os.getpid())


class FiniteProposalMethod(RunMethod[int, Proposal[int], Observation[int]]):
    """Emit integer proposals until a fixed state limit."""

    def __init__(self, limit: int) -> None:
        self.limit = limit

    @override
    def create_initial_state(self) -> int:
        return 0

    @override
    def is_exhausted(self, state: int) -> bool:
        return state >= self.limit

    @override
    def ask(
        self,
        state: int,
        batch_size: int = 1,
    ) -> tuple[tuple[Proposal[int], ...], int]:
        proposal_count = min(batch_size, self.limit - state)
        proposals = tuple(
            Proposal(
                candidate=(state + index) % 10,
                proposal_id=f"proposal-{state + index}",
            )
            for index in range(proposal_count)
        )
        return proposals, state + proposal_count

    @override
    def tell(
        self,
        state: int,
        observations: Sequence[Observation[int]],
    ) -> int:
        _ = observations
        return state


class AlternateProblemKernel(
    Kernel[
        ProposalBatchQuery[int, int, ObservationPayload],
        EvaluationAttemptBatch[int, ObservationPayload],
    ]
):
    """Dispatch a query carrying a different problem instance."""

    def __init__(self, problem: Problem[int, int, ObservationPayload]) -> None:
        self.problem = problem

    @override
    def run(
        self,
        query: ProposalBatchQuery[int, int, ObservationPayload],
        runner: Callable[
            [ProposalBatchQuery[int, int, ObservationPayload]],
            EvaluationAttemptBatch[int, ObservationPayload],
        ],
    ) -> EvaluationAttemptBatch[int, ObservationPayload]:
        return runner(replace(query, problem=self.problem))


def make_problem(
    objective: Objective[int],
) -> Problem[int, int, ObservationPayload]:
    """Build one scalar integer test problem."""
    return Problem(
        space=IntegerSpace(low=0, high=10),
        objective=objective,
    )


def make_requests(*candidates: int) -> tuple[EvaluationRequest[int], ...]:
    """Build one request tuple with stable proposal identifiers."""
    return tuple(
        EvaluationRequest(
            proposal=Proposal(candidate=candidate, proposal_id=f"p-{index}"),
        )
        for index, candidate in enumerate(candidates)
    )


def successful_values(
    attempts: EvaluationAttemptBatch[int, ObservationPayload],
) -> tuple[float, ...]:
    """Return scalar payload values from one successful attempt batch."""
    return tuple(success.payload.value for success in attempts.successes)


class JoblibWorkerSessionTests:
    """Exercise lifecycle, isolation, and failure boundaries."""

    def test_default_transport_remains_per_request(self) -> None:
        evaluator = JoblibEvaluator[int, int, ObservationPayload]()

        assert evaluator.problem_transport == "per_request"
        with pytest.raises(ValueError, match="problem_transport"):
            _ = JoblibEvaluator[int, int, ObservationPayload](
                problem_transport=cast(JoblibProblemTransportMode, "invalid"),
            )

    def test_reuses_one_snapshot_across_multiple_batches(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        problem = make_problem(DecodeIdentityObjective())
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend="loky",
            n_jobs=2,
            problem_transport="worker_session",
        )
        create_calls: list[Problem[int, int, ObservationPayload]] = []
        original_create = JoblibProblemSnapshotTransport.create.__func__

        def counting_create(
            cls: type[JoblibProblemSnapshotTransport[int, int, ObservationPayload]],
            bound_problem: Problem[int, int, ObservationPayload],
        ) -> JoblibProblemSnapshotTransport[int, int, ObservationPayload]:
            create_calls.append(bound_problem)
            return original_create(cls, bound_problem)

        monkeypatch.setattr(
            JoblibProblemSnapshotTransport,
            "create",
            classmethod(counting_create),
        )

        with evaluator._open_attempt_run_scope(problem) as attempt_evaluator:
            first = attempt_evaluator.evaluate_attempts(
                problem,
                make_requests(1, 2, 3, 4, 5, 6),
            )
            second = attempt_evaluator.evaluate_attempts(
                problem,
                make_requests(7, 8, 9, 10, 1, 2),
            )

        assert create_calls == [problem]
        assert set(successful_values(first)) == set(successful_values(second))
        assert 1 <= len(set(successful_values(first))) <= 2

    def test_sequential_sessions_replace_problem_context(self) -> None:
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend="loky",
            n_jobs=2,
            problem_transport="worker_session",
        )
        first_problem = make_problem(OffsetObjective(10.0))
        second_problem = make_problem(OffsetObjective(100.0))

        first = evaluator.evaluate_attempts(first_problem, make_requests(1, 2))
        second = evaluator.evaluate_attempts(second_problem, make_requests(1, 2))

        assert successful_values(first) == (11.0, 12.0)
        assert successful_values(second) == (101.0, 102.0)

    def test_outcome_path_preserves_request_order(self) -> None:
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend="loky",
            n_jobs=2,
            problem_transport="worker_session",
        )

        outcomes = evaluator.evaluate(
            make_problem(OffsetObjective(10.0)),
            make_requests(3, 1, 2),
        )

        assert tuple(
            outcome.observation.proposal.proposal_id for outcome in outcomes
        ) == ("p-0", "p-1", "p-2")
        assert tuple(outcome.observation.value for outcome in outcomes) == (
            13.0,
            11.0,
            12.0,
        )

    def test_concurrent_sessions_sharing_one_evaluator_remain_isolated(self) -> None:
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend="loky",
            n_jobs=2,
            problem_transport="worker_session",
        )
        first_problem = make_problem(OffsetObjective(10.0, delay=0.01))
        second_problem = make_problem(OffsetObjective(100.0, delay=0.01))
        requests = make_requests(1, 2, 3, 4, 5, 6)

        with ThreadPoolExecutor(max_workers=2) as executor:
            first_future = executor.submit(
                evaluator.evaluate_attempts,
                first_problem,
                requests,
            )
            second_future = executor.submit(
                evaluator.evaluate_attempts,
                second_problem,
                requests,
            )
            first = first_future.result()
            second = second_future.result()

        assert successful_values(first) == (11.0, 12.0, 13.0, 14.0, 15.0, 16.0)
        assert successful_values(second) == (
            101.0,
            102.0,
            103.0,
            104.0,
            105.0,
            106.0,
        )

    def test_interleaved_sessions_restore_the_requested_generation(self) -> None:
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend="loky",
            n_jobs=2,
            problem_transport="worker_session",
        )
        first_problem = make_problem(OffsetObjective(10.0))
        second_problem = make_problem(OffsetObjective(100.0))

        with (
            evaluator._open_attempt_run_scope(first_problem) as first_session,
            evaluator._open_attempt_run_scope(second_problem) as second_session,
        ):
            first_before = first_session.evaluate_attempts(
                first_problem,
                make_requests(1, 2, 3, 4),
            )
            second = second_session.evaluate_attempts(
                second_problem,
                make_requests(1, 2, 3, 4),
            )
            first_after = first_session.evaluate_attempts(
                first_problem,
                make_requests(5, 6, 7, 8),
            )

        assert successful_values(first_before) == (11.0, 12.0, 13.0, 14.0)
        assert successful_values(second) == (101.0, 102.0, 103.0, 104.0)
        assert successful_values(first_after) == (15.0, 16.0, 17.0, 18.0)

    def test_replacement_workers_lazily_install_the_session_problem(
        self,
    ) -> None:
        problem = make_problem(ProcessIdentityObjective())
        session = JoblibWorkerSession(
            problem=problem,
            backend="loky",
            n_jobs=2,
            worker_idle_timeout=0.05,
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            with session:
                first = session.evaluate_attempts(
                    problem,
                    make_requests(1, 2, 3, 4, 5, 6),
                )
                time.sleep(0.3)
                second = session.evaluate_attempts(
                    problem,
                    make_requests(1, 2, 3, 4, 5, 6),
                )

        assert all(
            "A worker stopped while some jobs were given" in str(warning.message)
            for warning in caught_warnings
        )
        assert set(successful_values(first)).isdisjoint(successful_values(second))

    def test_wrong_token_and_malformed_transport_fail_before_evaluation(
        self,
        tmp_path: Path,
    ) -> None:
        problem = make_problem(OffsetObjective(0.0))
        transport = JoblibProblemSnapshotTransport[int, int, ObservationPayload].create(
            problem
        )
        try:
            with pytest.raises(ValueError, match="generation"):
                _ = decode_problem_envelope(
                    token=b"x" * SESSION_TOKEN_BYTES,
                    transport=transport.mapping,
                )
        finally:
            transport.close()

        malformed_path = tmp_path / "malformed.snapshot"
        malformed_path.write_bytes(b"not a pickle")
        malformed_mapping = np.memmap(malformed_path, dtype=np.uint8, mode="r")
        try:
            with pytest.raises(pickle.UnpicklingError):
                _ = decode_problem_envelope(
                    token=b"y" * SESSION_TOKEN_BYTES,
                    transport=malformed_mapping,
                )
        finally:
            mapping_base = malformed_mapping.base
            if isinstance(mapping_base, mmap):
                mapping_base.close()

    def test_setup_failure_removes_partial_transport(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend="loky",
            n_jobs=2,
            problem_transport="worker_session",
        )

        with pytest.raises(TypeError, match="cannot be serialized"):
            _ = evaluator.evaluate_attempts(
                make_problem(UnserializableObjective()),
                make_requests(1),
            )

        assert tuple(tmp_path.iterdir()) == ()

    def test_snapshot_transport_is_owner_only_and_removed_on_close(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        transport = JoblibProblemSnapshotTransport[
            int,
            int,
            ObservationPayload,
        ].create(make_problem(OffsetObjective(0.0)))
        session_directories = tuple(tmp_path.iterdir())
        assert len(session_directories) == 1
        session_directory = session_directories[0]
        snapshot_files = tuple(session_directory.iterdir())
        assert len(snapshot_files) == 1
        assert stat.S_IMODE(session_directory.stat().st_mode) == 0o700
        assert stat.S_IMODE(snapshot_files[0].stat().st_mode) == 0o600

        transport.close()

        assert tuple(tmp_path.iterdir()) == ()

    def test_exceptional_scope_exit_removes_snapshot_transport(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        session = JoblibWorkerSession(
            problem=make_problem(OffsetObjective(0.0)),
            backend="loky",
            n_jobs=2,
        )

        with pytest.raises(RuntimeError, match="sentinel"), session:
            assert len(tuple(tmp_path.iterdir())) == 1
            raise RuntimeError("sentinel")

        assert tuple(tmp_path.iterdir()) == ()

    @pytest.mark.parametrize(
        ("backend", "n_jobs"),
        (("threading", 2), ("loky", 1)),
    )
    def test_direct_backends_do_not_create_process_transport(
        self,
        monkeypatch: pytest.MonkeyPatch,
        backend: Literal["loky", "threading"],
        n_jobs: int,
    ) -> None:
        def fail_create(
            cls: type[JoblibProblemSnapshotTransport[int, int, ObservationPayload]],
            problem: Problem[int, int, ObservationPayload],
        ) -> Never:
            _ = cls, problem
            raise AssertionError("process transport must not be created")

        monkeypatch.setattr(
            JoblibProblemSnapshotTransport,
            "create",
            classmethod(fail_create),
        )
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend=backend,
            n_jobs=n_jobs,
            problem_transport="worker_session",
        )
        attempts = evaluator.evaluate_attempts(
            make_problem(OffsetObjective(5.0)),
            make_requests(1, 2),
        )

        assert successful_values(attempts) == (6.0, 7.0)

    def test_empty_batch_does_not_open_a_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail_enter(
            session: JoblibWorkerSession[int, int, ObservationPayload],
        ) -> Never:
            _ = session
            raise AssertionError("empty batch must not open a worker session")

        monkeypatch.setattr(JoblibWorkerSession, "__enter__", fail_enter)
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend="loky",
            n_jobs=2,
            problem_transport="worker_session",
        )

        attempts = evaluator.evaluate_attempts(make_problem(OffsetObjective(0.0)), ())

        assert attempts.attempt_count == 0

    def test_dispatch_after_close_and_alternate_problem_are_rejected(self) -> None:
        first_problem = make_problem(OffsetObjective(0.0))
        second_problem = make_problem(OffsetObjective(10.0))
        session = JoblibWorkerSession(
            problem=first_problem,
            n_jobs=1,
            backend="loky",
        )
        with session, pytest.raises(ValueError, match="bound Problem"):
            _ = session.evaluate_attempts(second_problem, make_requests(1))

        with pytest.raises(RuntimeError, match="not active"):
            _ = session.evaluate_attempts(first_problem, make_requests(1))
        with pytest.raises(RuntimeError, match="reopened"):
            _ = session.__enter__()
        with pytest.raises(TypeError, match="runtime-only"):
            _ = pickle.dumps(session)

    def test_user_exception_remains_an_evaluation_failure(self) -> None:
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend="loky",
            n_jobs=2,
            problem_transport="worker_session",
        )

        attempts = evaluator.evaluate_attempts(
            make_problem(ExplodingObjective()),
            make_requests(1, 2, 3),
        )

        assert attempts.success_indices == (0, 2)
        assert attempts.failure_indices == (1,)
        assert attempts.failures[0].exception.exception_type == "builtins.ValueError"
        assert attempts.failures[0].exception.message == "objective exploded"

    def test_study_run_owns_one_scope_and_skips_scope_without_work(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        create_calls: list[Problem[int, int, ObservationPayload]] = []
        original_create = JoblibProblemSnapshotTransport.create.__func__

        def counting_create(
            cls: type[JoblibProblemSnapshotTransport[int, int, ObservationPayload]],
            problem: Problem[int, int, ObservationPayload],
        ) -> JoblibProblemSnapshotTransport[int, int, ObservationPayload]:
            create_calls.append(problem)
            return original_create(cls, problem)

        monkeypatch.setattr(
            JoblibProblemSnapshotTransport,
            "create",
            classmethod(counting_create),
        )
        problem = make_problem(OffsetObjective(0.0))
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend="loky",
            n_jobs=2,
            problem_transport="worker_session",
        )
        study = Study(
            problem=problem,
            run_method=FiniteProposalMethod(limit=3),
            evaluator=evaluator,
        )

        report, state = study.run(max_evaluations=3, batch_size=1)
        zero_report, _ = study.run(max_evaluations=0, batch_size=1)
        exhausted_report, _ = study.run(
            max_evaluations=3,
            batch_size=1,
            initial_state=3,
        )

        assert report.evaluation_count == 3
        assert state == 3
        assert zero_report.evaluation_count == 0
        assert exhausted_report.evaluation_count == 0
        assert create_calls == [problem]

    def test_optimize_delegates_to_one_run_owned_scope(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        create_count: list[None] = []
        original_create = JoblibProblemSnapshotTransport.create.__func__

        def counting_create(
            cls: type[JoblibProblemSnapshotTransport[int, int, ObservationPayload]],
            problem: Problem[int, int, ObservationPayload],
        ) -> JoblibProblemSnapshotTransport[int, int, ObservationPayload]:
            create_count.append(None)
            return original_create(cls, problem)

        monkeypatch.setattr(
            JoblibProblemSnapshotTransport,
            "create",
            classmethod(counting_create),
        )
        study = Study(
            problem=make_problem(OffsetObjective(0.0)),
            run_method=FiniteProposalMethod(limit=3),
            evaluator=JoblibEvaluator[int, int, ObservationPayload](
                backend="loky",
                n_jobs=2,
                problem_transport="worker_session",
            ),
        )

        result, state = study.optimize(max_evaluations=3, batch_size=1)

        assert result.evaluation_count == 3
        assert state == 3
        assert len(create_count) == 1

    def test_custom_kernel_alternate_problem_is_mode_specific(self) -> None:
        primary_problem = make_problem(OffsetObjective(0.0))
        alternate_problem = make_problem(OffsetObjective(100.0))
        run_method = FiniteProposalMethod(limit=1)
        worker_session_study = Study(
            problem=primary_problem,
            run_method=run_method,
            evaluator=JoblibEvaluator[int, int, ObservationPayload](
                backend="loky",
                n_jobs=2,
                problem_transport="worker_session",
            ),
            kernel=AlternateProblemKernel(alternate_problem),
        )
        per_request_study = Study(
            problem=primary_problem,
            run_method=run_method,
            evaluator=JoblibEvaluator[int, int, ObservationPayload](
                backend="loky",
                n_jobs=2,
            ),
            kernel=AlternateProblemKernel(alternate_problem),
        )

        with pytest.raises(ValueError, match="bound Problem"):
            _ = worker_session_study.step(0)

        records, _ = per_request_study.step(0)
        assert tuple(record.value for record in records) == (100.0,)

    def test_evaluator_and_study_pickle_store_configuration_only(self) -> None:
        evaluator = JoblibEvaluator[int, int, ObservationPayload](
            backend="loky",
            n_jobs=2,
            problem_transport="worker_session",
        )
        problem = make_problem(OffsetObjective(0.0))
        study = Study(
            problem=problem,
            run_method=FiniteProposalMethod(limit=1),
            evaluator=evaluator,
        )

        with evaluator._open_attempt_run_scope(problem) as session:
            _ = session.evaluate_attempts(problem, make_requests(1))
            restored_evaluator = pickle.loads(pickle.dumps(evaluator))
            restored_study = pickle.loads(pickle.dumps(study))

        assert restored_evaluator == evaluator
        assert restored_study.evaluator == evaluator
