# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""TFLite converter plugin module."""

from mlia.backend.tosa_converter_for_tflite.conversion import TosaConverterForTflite
from mlia.plugins.plugins import Plugin

from mlia.transformers.registry import Transformer
from mlia.utils.registry import Registry


class TFLiteToTosaConverterPlugin(Plugin):
    """TFLite to TOSA Converter Plugin."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: Registry[Transformer]) -> None:
        """Register the converter with the registry."""
        registry.register("tflite_to_tosa", TosaConverterForTflite())
