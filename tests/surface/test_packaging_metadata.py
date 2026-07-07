"""Regression tests for release-facing packaging metadata."""

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

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


def _load_toml(text: str) -> dict[str, object]:
    """Load TOML using the stdlib parser when available."""
    try:
        toml_module = importlib.import_module("tomllib")
    except ModuleNotFoundError:
        toml_module = importlib.import_module("tomli")
    loads = cast(Callable[[str], object], getattr(toml_module, "loads"))
    parsed = loads(text)
    if not isinstance(parsed, dict):
        msg = "TOML parser returned a non-mapping document"
        raise TypeError(msg)
    return cast(dict[str, object], parsed)


def _load_pyproject_metadata() -> PyprojectMetadata:
    """Load the typed release-facing pyproject metadata subset."""
    pyproject_data = _load_toml(
        Path("pyproject.toml").read_text(),
    )
    return cast(PyprojectMetadata, cast(object, pyproject_data))


def _workflow_uses_references(path: Path) -> tuple[str, ...]:
    """Return GitHub Actions `uses:` references from one workflow file."""
    references: list[str] = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("- uses: "):
            references.append(stripped.removeprefix("- uses: ").strip())
    return tuple(references)


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
        pyproject_data = _load_toml(Path("pyproject.toml").read_text())

        hatch_config = cast(dict[str, object], pyproject_data["tool"])["hatch"]
        build_config = cast(dict[str, object], hatch_config)["build"]
        targets = cast(dict[str, object], build_config)["targets"]
        wheel_config = cast(dict[str, object], targets)["wheel"]

        assert cast(dict[str, object], wheel_config)["packages"] == ["src/variopt"]

    def test_sdist_excludes_repo_local_workflow_artifacts(self) -> None:
        pyproject_data = _load_toml(Path("pyproject.toml").read_text())

        hatch_config = cast(dict[str, object], pyproject_data["tool"])["hatch"]
        build_config = cast(dict[str, object], hatch_config)["build"]
        targets = cast(dict[str, object], build_config)["targets"]
        sdist_config = cast(dict[str, object], targets)["sdist"]
        excludes = set(cast(list[str], cast(dict[str, object], sdist_config)["exclude"]))

        assert "/dist" in excludes
        assert "/tests" in excludes
        assert "/uv.lock" in excludes

    def test_local_uv_lock_is_not_release_metadata(self) -> None:
        pyproject_data = _load_toml(Path("pyproject.toml").read_text())
        gitignore_entries = set(Path(".gitignore").read_text().splitlines())
        ci_workflow = Path(".github/workflows/ci.yml").read_text()

        hatch_config = cast(dict[str, object], pyproject_data["tool"])["hatch"]
        build_config = cast(dict[str, object], hatch_config)["build"]
        targets = cast(dict[str, object], build_config)["targets"]
        sdist_config = cast(dict[str, object], targets)["sdist"]
        excludes = set(cast(list[str], cast(dict[str, object], sdist_config)["exclude"]))

        assert "uv.lock" in gitignore_entries
        assert "/uv.lock" in excludes
        assert "cache-dependency-glob: pyproject.toml" in ci_workflow
        assert "cache-dependency-glob: uv.lock" not in ci_workflow

    def test_ci_action_references_use_version_tags(self) -> None:
        workflow_paths = (
            Path(".github/workflows/canary.yml"),
            Path(".github/workflows/ci.yml"),
            Path(".github/workflows/docs.yml"),
        )

        references = tuple(
            reference
            for workflow_path in workflow_paths
            for reference in _workflow_uses_references(workflow_path)
        )

        assert references
        for reference in references:
            assert "@" in reference
            _, ref = reference.rsplit("@", 1)
            assert ref not in {"main", "master", "HEAD"}
            assert ref.startswith("v")

    def test_docs_workflow_builds_without_deploying_site(self) -> None:
        docs_workflow = Path(".github/workflows/docs.yml").read_text()

        assert "mkdocs build --strict" in docs_workflow
        assert "gh-pages" not in docs_workflow
        assert "peaceiris/actions-gh-pages" not in docs_workflow
        assert "mkdocs gh-deploy" not in docs_workflow

    def test_ci_runs_when_release_metadata_inputs_change(self) -> None:
        ci_workflow = Path(".github/workflows/ci.yml").read_text()

        for release_metadata_path in (
            "pyproject.toml",
            ".gitignore",
            ".github/workflows/canary.yml",
            ".github/workflows/ci.yml",
            ".github/workflows/docs.yml",
        ):
            assert ci_workflow.count(f'- "{release_metadata_path}"') == 2

    def test_dependency_canary_tracks_latest_resolution_without_lockfile(self) -> None:
        canary_workflow = Path(".github/workflows/canary.yml").read_text()

        assert "schedule:" in canary_workflow
        assert "workflow_dispatch:" in canary_workflow
        assert "uv run --upgrade --python 3.13 --extra test" in canary_workflow
        assert "uv run --upgrade --python 3.13 --extra docs" in canary_workflow
        assert "uv build --wheel" in canary_workflow
        assert '"${wheel_path}"' in canary_workflow
        assert '"${wheel_path}[mpi]"' in canary_workflow
        assert "--surface base" in canary_workflow
        assert "--surface joblib-private" in canary_workflow
        assert "--surface mpi" in canary_workflow
        assert "--locked" not in canary_workflow
        assert "--frozen" not in canary_workflow
        assert "cache-dependency-glob: uv.lock" not in canary_workflow
