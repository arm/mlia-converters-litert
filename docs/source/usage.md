<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Usage and Integration

## Overview

This package registers the transformer name `tflite_to_tosa` through the
`mlia.plugin.transformer` entry point. The implementation is provided by
`TFLiteToTosaConverterPlugin` in
`src/mlia/backend/tosa_converter_for_tflite/converter_plugin.py`.

## Naming convention

This repo uses two names that matter in different places:

- `tflite_to_tosa`: the transformer name exposed to MLIA and used in API or
  CLI-facing flows
- `tosa_converter_for_tflite`: the implementation package name used in the codebase.

## Integration model

The repository is designed to be installed alongside `mlia`, not used as a
standalone CLI tool. Once installed, MLIA can discover the converter and route
TFLite conversion requests through the transformer registry.

## Typical workflow

The converter usually participates in a larger MLIA flow:

1. MLIA receives a `.tflite` model.
2. The converter plugin transforms it into a TOSA-oriented form.
3. A downstream backend consumes the converted artifact.
4. MLIA returns target-specific analysis results.

## Output controls

The transformer supports two output formats:

- `mlir-text`, which writes a `.tosamlir` artifact and is the default.
- `mlir-bytecode`, which writes a `.tosa.mlirbc` artifact.

MLIA may also pass `emit_debug_info`; the option is accepted for workflow
compatibility, but the current wrapper does not add a converter flag for it.

## Source layout

- `conversion.py`: conversion logic.
- `converter_plugin.py`: MLIA plugin registration.
- `install.py`: installation metadata for MLIA-managed backend tooling.

## Cross-links

- See [conversion_flow.md](conversion_flow.md) for a more pipeline-oriented view.
- See [conversion_outputs.md](conversion_outputs.md) for what success looks like from
  this converter's perspective.
- See [troubleshooting.md](troubleshooting.md) for conversion-specific failures.
