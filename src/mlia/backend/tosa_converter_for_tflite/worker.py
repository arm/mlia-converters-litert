# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Convert a TFLite model to TOSA in the selected backend environment."""

from pathlib import Path
from typing import Any


def run(request: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read the requested model and options, then return the output model file."""
    from mlia.backend.tosa_converter_for_tflite.conversion import TosaConverterForTflite

    payload = request["payload"]
    output = TosaConverterForTflite()(
        Path(payload["model"]),
        Path(request["output_dir"]),
        **payload.get("options", {}),
    )
    return {}, [
        {
            "path": str(output),
            "kind": "model",
            "representation": output.suffix.lstrip("."),
        }
    ]


def validate_runtime(request: dict[str, Any]) -> None:
    """Ensure the provisioned converter is available without installing anything."""
    from mlia.backend.tosa_converter_for_tflite.install import (
        ensure_tosa_converter_for_tflite_installed,
    )

    ensure_tosa_converter_for_tflite_installed()
