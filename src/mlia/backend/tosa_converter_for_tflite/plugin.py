# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Register the TFLite-to-TOSA backend for MLIA installation and discovery."""

from mlia.backend.config import BackendConfiguration, BackendType, System
from mlia.backend.registry import BackendRegistry
from mlia.plugins.plugins import BackendPlugin


class LiteRTBackendPlugin(BackendPlugin):
    """Describe the LiteRT backend's capabilities and installation requirements."""

    plugin_interface_version = "0.0.1"

    @staticmethod
    def register(registry: BackendRegistry) -> None:
        """Add the converter backend to the supplied registry."""
        registry.register(
            "tosa-converter-for-tflite",
            BackendConfiguration(
                supported_advice=[],
                supported_systems=[System.LINUX_AMD64],
                backend_type=BackendType.WHEEL,
                installation=None,
                runtime_operation="litert/tflite-to-tosa",
                selectable=False,
            ),
            pretty_name="TOSA converter for tflite",
        )
