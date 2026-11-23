# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Locate the TFLite-to-TOSA converter in the current backend environment."""

from __future__ import annotations

from functools import cache
import importlib
import sys


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
    """Return the installed converter's command or explain how to repair it."""
    module_name = "tosa_converter_for_tflite.cli"
    importlib.invalidate_caches()
    if _module_available(module_name):
        return (sys.executable, "-m", module_name)

    raise RuntimeError(
        "Provisioned runtime is missing the LiteRT converter; "
        "reinstall it with mlia backend install tosa-converter-for-tflite --force."
    )
