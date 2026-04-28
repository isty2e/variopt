"""Tests for exact-async study package facade exports."""


import variopt.study.exact_async as exact_async_package
from variopt.study.exact_async import (
    StudyExactAsyncSessionLifecycle,
    StudyExactAsyncStepResumeHandle,
    StudyExactAsyncStepSession,
)
from variopt.study.exact_async.artifacts import (
    StudyExactAsyncSessionLifecycle as StudyExactAsyncSessionLifecycleSubmodule,
)
from variopt.study.exact_async.artifacts import (
    StudyExactAsyncStepResumeHandle as StudyExactAsyncStepResumeHandleSubmodule,
)
from variopt.study.exact_async.session import (
    StudyExactAsyncStepSession as StudyExactAsyncStepSessionSubmodule,
)


class StudyExactAsyncExportTests:
    """Regression tests for exact-async study facade identity."""

    def test_exact_async_facade_re_exports_artifact_symbols(self) -> None:
        assert StudyExactAsyncSessionLifecycle is StudyExactAsyncSessionLifecycleSubmodule
        assert StudyExactAsyncStepResumeHandle is StudyExactAsyncStepResumeHandleSubmodule
        assert StudyExactAsyncStepSession is StudyExactAsyncStepSessionSubmodule

    def test_exact_async_facade_omits_orchestration_helpers(self) -> None:
        assert not (hasattr(exact_async_package, "evaluate_batch_exact_async"))
        assert not (hasattr(exact_async_package, "open_exact_async_step_session"))
        assert not (hasattr(exact_async_package, "resume_exact_async_step_session"))
