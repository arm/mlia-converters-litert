# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Installation module for the TOSA Converter For Tflite."""

from __future__ import annotations

from functools import cache
import logging
import shutil
import importlib
import sys

from mlia.backend.install import (
    DownloadAndInstall,
    Installation,
    PyPackageBackendInstallation,
)

logger = logging.getLogger(__name__)

TOSA_CONVERTER_PACKAGE = "tosa-converter-for-tflite==2026.2.0"
TOSA_CONVERTER_DISTRIBUTION = "tosa-converter-for-tflite"


def get_tosa_converter_for_tflite_backend_installation() -> Installation:
    """Get TOSA converter for tflite backend package."""
    return PyPackageBackendInstallation(
        name="tosa-converter-for-tflite",
        description="Tool to convert a tflite file to TOSA",
        packages_to_install=[TOSA_CONVERTER_PACKAGE],
        packages_to_uninstall=[TOSA_CONVERTER_DISTRIBUTION],
        expected_packages=[TOSA_CONVERTER_DISTRIBUTION],
    )


def _resolve_executable(exe_name: str) -> str | None:
    """Resolve the converter executable from PATH."""
    resolved = shutil.which(exe_name)
    if resolved:
        return resolved
    return None


def _module_available(module_name: str) -> bool:
    """Return True if the module can be imported."""
    try:
        importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name or module_name.startswith(f"{exc.name}."):
            return False
        raise
    return True


@cache
def ensure_tosa_converter_for_tflite_installed() -> tuple[str, ...]:
    """Ensure the TOSA converter executable is available and return argv prefix."""
    exe_name = "tosa-converter-for-tflite"
    module_name = "tosa_converter_for_tflite.cli"
    importlib.invalidate_caches()
    resolved = _resolve_executable(exe_name)
    if resolved:
        return (resolved,)
    if _module_available(module_name):
        return (sys.executable, "-m", module_name)

    installation = get_tosa_converter_for_tflite_backend_installation()
    install_type = DownloadAndInstall()

    if not installation.supports(install_type):
        raise RuntimeError(
            "Auto-install failed: 'tosa-converter-for-tflite' package "
            "installation is not available."
        )

    logger.info(
        "Installing 'tosa-converter-for-tflite' from the configured Python "
        "package index."
    )
    installation.install(install_type)

    importlib.invalidate_caches()
    resolved = _resolve_executable(exe_name)
    if resolved:
        return (resolved,)
    if _module_available(module_name):
        return (sys.executable, "-m", module_name)

    raise RuntimeError(
        "Auto-install succeeded but the 'tosa_converter_for_tflite' module "
        "could not be imported."
    )
