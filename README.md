<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# MLIA TFLite Converter Plugin

This repository contains the MLIA converter plugin that translates TFLite
models into TOSA so they can be consumed by MLIA backends and target flows that
operate on TOSA artifacts.

The package is distributed as `mlia-converters-tflite`. When installed, it
registers the backend key `tflite_to_tosa` with MLIA through the plugin
entry-point system.

## Table of Contents

- [Overview](#overview)
- [Repository contents](#repository-contents)
- [Installation](#installation)
- [How MLIA uses this plugin](#how-mlia-uses-this-plugin)
- [Reporting bugs](#reporting-bugs)
- [Development (uv)](#development-uv)
- [Documentation](#documentation)
- [Trademarks and copyrights](#trademarks-and-copyrights)

## Overview

This plugin provides the conversion bridge between TFLite model files and
TOSA-based MLIA backends. It gives the wider MLIA ecosystem a dedicated package
for TFLite-to-TOSA conversion and keeps converter logic separate from the core
MLIA framework.

The implementation package lives under
`src/mlia/backend/tosa_converter_for_tflite/` and includes:

- Conversion logic.
- Converter registration.
- Backend installation metadata used by MLIA.

## Repository contents

- `src/mlia/backend/tosa_converter_for_tflite/`: implementation package and
  plugin registration.
- `tests/`: unit tests for converter registration and conversion behaviour.
- `pre_commit_hooks/`: local repository hooks shared with CI quality checks.
- `hatch_build.py`: packaging hook used during builds.

## Installation

Install the package into an environment that already contains `mlia`:

```bash
pip install mlia-converters-tflite
```

For source-based development with `uv`:

```bash
uv sync --dev
```

The project requires Python 3.10 and keeps its direct dependency surface small,
with `mlia` as the primary runtime dependency.

## How MLIA uses this plugin

MLIA discovers this repository through the `mlia.plugin.converter` entry point.
When installed, the plugin registers the backend key `tflite_to_tosa`.

This is the important naming split:

- `tflite_to_tosa` is the backend key used in MLIA configuration and CLI flows
- `tosa_converter_for_tflite` is the implementation package name used in the codebase

That means downstream MLIA components can:

- Discover the converter without hard-coded import paths.
- Request a TFLite-to-TOSA conversion through the converter registry.
- Treat the converter as a separately versioned plugin package.

For more implementation detail, see [docs/README.md](docs/README.md).

## Reporting bugs

Report bugs by creating GitHub issues. Use the
[`arm/mlia` issue tracker](https://github.com/arm/mlia/issues) by default.

Only open an issue in
[`arm/mlia-converters-tflite`](https://github.com/arm/mlia-converters-tflite/issues)
when the bug is clearly and specifically in this TFLite converter plugin.

## Development (uv)

This repository uses `uv` for environment management and test execution. Ensure
Python 3.10 is available (see `.python-version`), then install dependencies:

```bash
uv sync --dev
```

Run unit tests (uses dependencies installed from the package index, including `mlia`):

```bash
uv run pytest --no-success-flaky-report tests/
```

Run a quick test subset:

```bash
uv run pytest --no-success-flaky-report -m "not slow" tests/
```

Lint checks:

```bash
uv run pre-commit run --all-files
```

Build a wheel:

```bash
uv build --wheel
```

## CI Parity With mlia-core

CI jobs follow the same structure as mlia-core (lint/build/test_quick) and use
uv-based commands. Deviations are documented in the workflow files where the
repo lacks equivalent tooling (for example, pre-commit configuration).

## Documentation

Additional repository documentation lives in [docs/README.md](docs/README.md).

## Trademarks and copyrights

- Arm is a registered trademark or trademark of Arm Limited (or its subsidiaries) in the U.S. and/or elsewhere.
- TensorFlow is a trademark of Google LLC.
- Linux is the registered trademark of Linus Torvalds in the U.S. and elsewhere.
- Python is a registered trademark of the PSF.
