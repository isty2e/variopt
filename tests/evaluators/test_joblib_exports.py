"""Tests for joblib evaluator facade exports."""

import variopt.evaluators.joblib as joblib_package
from variopt.evaluators import AsyncJoblibEvaluator, JoblibEvaluator
from variopt.evaluators.joblib import (
    AsyncJoblibEvaluator as AsyncJoblibEvaluatorPackage,
)
from variopt.evaluators.joblib import (
    JoblibEvaluator as JoblibEvaluatorPackage,
)
from variopt.evaluators.joblib.asynchronous import (
    AsyncJoblibEvaluator as AsyncJoblibEvaluatorSubmodule,
)
from variopt.evaluators.joblib.batches import (
    AsyncJoblibRequestInput as AsyncJoblibRequestInputSubmodule,
)
from variopt.evaluators.joblib.sync import JoblibEvaluator as JoblibEvaluatorSubmodule


class JoblibExportTests:
    """Regression tests for joblib evaluator facade identity."""

    def test_root_and_package_facades_re_export_canonical_evaluators(self) -> None:
        assert JoblibEvaluator is JoblibEvaluatorPackage
        assert JoblibEvaluator is JoblibEvaluatorSubmodule
        assert AsyncJoblibEvaluator is AsyncJoblibEvaluatorPackage
        assert AsyncJoblibEvaluator is AsyncJoblibEvaluatorSubmodule

    def test_package_facade_omits_async_request_input(self) -> None:
        assert not (hasattr(joblib_package, "AsyncJoblibRequestInput"))
        assert (
            AsyncJoblibRequestInputSubmodule.__module__
            == "variopt.evaluators.joblib.batches"
        )
