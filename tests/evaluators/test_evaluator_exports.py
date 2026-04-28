"""Regression tests for root evaluator facade exports."""

import importlib
import sys
from typing import cast

from variopt.evaluators import AsyncJoblibEvaluator, JoblibEvaluator
from variopt.evaluators.joblib import (
    AsyncJoblibEvaluator as AsyncJoblibEvaluatorPackage,
)
from variopt.evaluators.joblib import (
    JoblibEvaluator as JoblibEvaluatorPackage,
)
from variopt.evaluators.mpi import MpiEvaluator as MpiEvaluatorSubmodule
from variopt.evaluators.mpi import (
    MpiExecutorFactory as MpiExecutorFactorySubmodule,
)


class EvaluatorExportTests:
    """Lock the release-facing evaluator facade boundaries."""

    def test_root_facade_re_exports_joblib_evaluators(self) -> None:
        assert JoblibEvaluator is JoblibEvaluatorPackage
        assert AsyncJoblibEvaluator is AsyncJoblibEvaluatorPackage

    def test_root_facade_keeps_mpi_exports_lazy(self) -> None:
        _ = sys.modules.pop("variopt.evaluators", None)

        module = importlib.import_module("variopt.evaluators")

        assert "MpiEvaluator" not in module.__dict__
        assert "MpiExecutorFactory" not in module.__dict__
        assert cast(object, getattr(module, "MpiEvaluator")) is MpiEvaluatorSubmodule
        assert cast(object, getattr(module, "MpiExecutorFactory")) is MpiExecutorFactorySubmodule
