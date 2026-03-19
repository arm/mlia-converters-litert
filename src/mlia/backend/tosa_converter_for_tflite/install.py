# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Installation module for the TOSA Converter For Tflite."""

from __future__ import annotations

from mlia.backend.install import (
    Installation,
    PyPackageBackendInstallation,
)


def get_tosa_converter_for_tflite_backend_installation() -> Installation:
    """Get TOSA converter for tflite backend whl."""
    return PyPackageBackendInstallation(
        name="tosa-converter-for-tflite",
        description="Tool to convert a tflite file to TOSA",
        packages_to_install=[],
        packages_to_uninstall=["tosa-converter-for-tflite"],
        expected_packages=["tosa-converter-for-tflite"],
        vendor_path="tosa-converter-for-tflite",
    )
