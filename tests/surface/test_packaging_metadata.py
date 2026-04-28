"""Regression tests for release-facing packaging metadata."""

from pathlib import Path
from typing import TypedDict, cast

import tomli

ProjectMetadata = TypedDict(
    "ProjectMetadata",
    {
        "dependencies": list[str],
        "optional-dependencies": dict[str, list[str]],
    },
)

class PyprojectMetadata(TypedDict):
    """Typed subset of pyproject metadata required by packaging tests."""

    project: ProjectMetadata


def _load_pyproject_metadata() -> PyprojectMetadata:
    """Load the typed release-facing pyproject metadata subset."""
    pyproject_data = tomli.loads(
        Path("pyproject.toml").read_text(),
    )
    return cast(PyprojectMetadata, cast(object, pyproject_data))


class PackagingMetadataTests:
    """Lock core and optional dependency boundaries in pyproject metadata."""

    def test_core_dependencies_include_runtime_requirements(self) -> None:
        pyproject_data = _load_pyproject_metadata()

        dependencies = set(pyproject_data["project"]["dependencies"])

        assert "joblib>=1.4.0" in dependencies
        assert "numpy>=1.23.5" in dependencies
        assert "scipy>=1.10.0" in dependencies
        assert "typing_extensions>=4.12.0" in dependencies

    def test_core_runtime_dependencies_are_not_modeled_as_optional_extras(self) -> None:
        pyproject_data = _load_pyproject_metadata()

        optional_dependencies = pyproject_data["project"]["optional-dependencies"]

        assert "parallel" not in optional_dependencies
        assert "scipy" not in optional_dependencies
        assert "pymoo" not in optional_dependencies
        assert "ioh" not in optional_dependencies
        assert "coco" not in optional_dependencies

    def test_docs_extra_declares_mkdocs_site_dependencies(self) -> None:
        pyproject_data = _load_pyproject_metadata()

        optional_dependencies = pyproject_data["project"]["optional-dependencies"]

        assert "docs" in optional_dependencies
        assert "mkdocs>=1.6.1" in optional_dependencies["docs"]
        assert "mkdocstrings>=0.27.0" in optional_dependencies["docs"]
        assert "mkdocstrings-python>=1.12.2" in optional_dependencies["docs"]

    def test_test_extra_declares_release_gate_dependencies(self) -> None:
        pyproject_data = _load_pyproject_metadata()

        optional_dependencies = pyproject_data["project"]["optional-dependencies"]

        assert "test" in optional_dependencies
        assert "basedpyright>=1.20.0" in optional_dependencies["test"]
        assert "pytest>=8.0.0" in optional_dependencies["test"]
        assert "ruff>=0.8.0" in optional_dependencies["test"]
        assert "tomli>=2.0.0" in optional_dependencies["test"]

    def test_wheel_only_packages_variopt_runtime(self) -> None:
        pyproject_data = tomli.loads(Path("pyproject.toml").read_text())

        hatch_config = cast(dict[str, object], pyproject_data["tool"])["hatch"]
        build_config = cast(dict[str, object], hatch_config)["build"]
        targets = cast(dict[str, object], build_config)["targets"]
        wheel_config = cast(dict[str, object], targets)["wheel"]

        assert cast(dict[str, object], wheel_config)["packages"] == ["src/variopt"]

    def test_sdist_excludes_repo_local_workflow_artifacts(self) -> None:
        pyproject_data = tomli.loads(Path("pyproject.toml").read_text())

        hatch_config = cast(dict[str, object], pyproject_data["tool"])["hatch"]
        build_config = cast(dict[str, object], hatch_config)["build"]
        targets = cast(dict[str, object], build_config)["targets"]
        sdist_config = cast(dict[str, object], targets)["sdist"]
        excludes = set(cast(list[str], cast(dict[str, object], sdist_config)["exclude"]))

        assert "/dist" in excludes
        assert "/tests" in excludes
        assert "/uv.lock" in excludes
