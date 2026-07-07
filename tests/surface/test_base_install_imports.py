"""Regression tests for the supported base-install import surface."""

from tests.release_support import collect_base_import_failures


class BaseInstallImportSmokeTests:
    """Lock the supported public modules for a base install."""

    def test_base_install_surface_imports_cleanly(self) -> None:
        assert collect_base_import_failures() == ()
