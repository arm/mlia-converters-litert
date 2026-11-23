<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# TFLite converter runtime

The converter package is defined in [litert/pyproject.toml](litert/pyproject.toml).
It provides `litert/tflite-to-tosa`, which takes a TFLite model and returns a TOSA
model file. Conversion options are passed to the TFLite-to-TOSA converter.

`uv build` embeds the worker wheel in the plugin. MLIA installs it and its
requirements in a separate environment on first use. Its version matches the
plugin wheel. Update the converter dependency in that package definition when
changing the toolchain.

The worker implementation is [TFLite converter worker](../src/mlia/backend/tosa_converter_for_tflite/worker.py).
