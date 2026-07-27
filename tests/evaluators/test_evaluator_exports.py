"""Regression tests for root evaluator facade exports."""

import importlib
import sys
from typing import cast

import variopt.evaluators as evaluator_facade
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
        assert cast(object, module.MpiEvaluator) is MpiEvaluatorSubmodule
        assert cast(object, module.MpiExecutorFactory) is MpiExecutorFactorySubmodule

    def test_evaluator_facade_omits_request_local_execution_internals(self) -> None:
        internal_names = (
            "BoundedRequestLocalEvaluationRunner",
            "RequestLocalEpisodeEvaluator",
            "execute_request_local_episode",
            "ordered_request_local_episodes",
        )

        assert all(name not in evaluator_facade.__all__ for name in internal_names)
        assert all(not hasattr(evaluator_facade, name) for name in internal_names)
