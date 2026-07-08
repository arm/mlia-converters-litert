# SPDX-FileCopyrightText: Copyright 2025-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Convert TensorFlow Lite models with the TOSA Converter For Tflite."""

from __future__ import annotations

import logging
from typing import Any
from pathlib import Path

from mlia.backend.tosa_converter_for_tflite.install import (
    ensure_tosa_converter_for_tflite_installed,
)
from mlia.utils.logging import log_action
from mlia.utils.proc import (
    Command,
    OutputConsumer,
    OutputLogger,
    process_command_output,
)

logger = logging.getLogger(__name__)


class TosaConverterForTflite:
    """Wrapper class to run the TOSA Converter For Tflite."""

    SUPPORTED_KWARGS: set[str] = {"output_format", "emit_debug_info"}
    SUPPORTED_OUTPUT_FORMATS: set[str] = {"mlir-text", "mlir-bytecode"}

    def __init__(self) -> None:
        """Set up some paths to run the TOSA Converter For Tflite."""
        self.output_consumers: list[OutputConsumer] = [
            OutputLogger(logger, logging.INFO)
        ]

    def _correct_kwargs(self, kwargs: dict[str, Any]) -> bool:
        """Return whether kwargs match the converter's supported signature."""
        if not set(kwargs).issubset(self.SUPPORTED_KWARGS):
            return False

        output_format = kwargs.get("output_format", "mlir-text")
        if output_format not in self.SUPPORTED_OUTPUT_FORMATS:
            return False

        emit_debug_info = kwargs.get("emit_debug_info")
        return emit_debug_info is None or isinstance(emit_debug_info, bool)

    def __call__(
        self,
        tflite_file: Path,
        output_dir: Path,
        *,
        output_format: str = "mlir-text",
        emit_debug_info: bool | None = None,
    ) -> Path:
        """
        Run the TOSA Converter For Tflite with the given TensorFlow Lite file.

        Returns the path of the TOSA MLIR output file created in the output dir.
        """
        del emit_debug_info
        if not output_dir.is_dir():
            raise NotADirectoryError(
                f"Path '{output_dir}' is not a directory. Unable to run "
                "TOSA Converter For Tflite."
            )
        if not tflite_file.is_file():
            raise FileNotFoundError(
                f"TensorFlow Lite model file '{tflite_file}' not found. "
                "Unable to run TOSA Converter For Tflite."
            )
        with log_action("Running TOSA Converter For Tflite..."):
            logger.debug("TOSA Converter For Tflite:")

            converter_cmd = ensure_tosa_converter_for_tflite_installed()

            tosa_file = self._run_converter(
                tflite_file, output_dir, converter_cmd, output_format
            )

            logger.debug("Output file: %s", tosa_file)

        return tosa_file

    def supports(
        self,
        model: object,
        target_format: str,
        kwargs: dict[str, Any],
    ) -> bool:
        """Return whether this converter can handle the given model."""
        if target_format != "tosa":
            return False
        if not isinstance(model, Path):
            return False
        if model.suffix != ".tflite":
            return False
        return self._correct_kwargs(kwargs)

    def _create_converter_command(
        self,
        tflite_file: Path,
        tosa_file: Path,
        converter_cmd: tuple[str, ...],
        output_format: str,
    ) -> Command:
        """Create the command to run the TOSA Converter For Tflite."""
        cmd = Command(
            cmd=[
                *converter_cmd,
                str(tflite_file),
                "-o",
                str(tosa_file),
                *self._extra_arguments(output_format),
            ],
        )
        return cmd

    def _run_converter(
        self,
        tflite_file: Path,
        output_dir: Path,
        converter_cmd: tuple[str, ...],
        output_format: str,
    ) -> Path:
        """Run the TOSA Converter For Tflite and return the TOSA MLIR output file."""
        if output_format not in self.SUPPORTED_OUTPUT_FORMATS:
            raise ValueError(f"Unsupported output format: {output_format}")

        suffix = ".tosamlir" if output_format == "mlir-text" else ".tosa.mlirbc"
        tosa_file = output_dir / f"{tflite_file.stem}{suffix}"
        cmd = self._create_converter_command(
            tflite_file, tosa_file, converter_cmd, output_format
        )
        process_command_output(cmd, self.output_consumers)

        if not tosa_file.is_file():
            raise FileNotFoundError(
                "No output from the TOSA Converter For Tflite found. "
                f"File {tosa_file} does not exist."
            )
        logger.debug(
            "TOSA Converter For Tflite run successfully. See output: %s",
            tosa_file,
        )

        return tosa_file

    def _extra_arguments(self, output_format: str = "mlir-text") -> list[str]:
        """Return any extra arguments to be used with the TOSA Converter For Tflite."""
        return ["--text"] if output_format == "mlir-text" else []
