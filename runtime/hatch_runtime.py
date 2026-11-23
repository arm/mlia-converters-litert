# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Include the converter worker and its installation metadata in the plugin package."""

from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from mlia.runtime.build import runtime_resource_files


class RuntimeBuildHook(BuildHookInterface):
    """Build the converter worker with the same version as the plugin package."""

    def initialize(self, version: str, build_data: dict) -> None:
        """Include the verified files for this wheel or source distribution."""
        build_data.setdefault("force_include", {}).update(
            runtime_resource_files(
                Path(self.root),
                self.metadata.name,
                self.metadata.version,
                self.target_name,
            )
        )
