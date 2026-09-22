<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA LiteRT Converter Documentation

This directory contains the MkDocs content for the
`mlia-converters-litert` repository.

## Included pages

- `source/index.md`: documentation landing page
- `source/usage.md`: plugin purpose, packaging model, and MLIA integration
- `source/conversion_flow.md`: how LiteRT / TensorFlow Lite `.tflite` models
  move through this converter
- `source/conversion_outputs.md`: conversion-stage outputs, success signals, and diagnostics
- `source/cli.md`: CLI examples for automatic and explicit converter usage
- `source/troubleshooting.md`: converter-specific troubleshooting notes
- `source/development.md`: local development, testing, and maintenance notes

## Build

Install the documentation dependencies in your environment, then build from the
repository root:

```bash
uv sync --no-sources --no-install-project --only-group docs
uv run --no-sync mkdocs build --strict
```

For local preview:

```bash
uv run --no-sync mkdocs serve
```

The generated site will be written to `.mkdocs/site/`.

## Scope

These docs cover the LiteRT / TensorFlow Lite `.tflite`-to-TOSA conversion path
provided by `mlia-converters-litert`.

## Relationship to the core and target repos

Use the main `mlia` repo for shared CLI and output-structure concepts. Use the
target repos for the backend-specific metrics that appear after conversion. Use
this docs tree for conversion-path behaviour, diagnostics, and debugging.
