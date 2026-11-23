---
# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
sidebar_position: 6
---

# Troubleshooting

## General issues

### Transformer plugin not available

- Confirm the package is installed in the active environment.
- Check MLIA's plugin discovery flow in the wider environment.
- Reinstall the package if the `tflite_to_tosa` transformer name is not being
  discovered.

### Wrong input type

- This repo is intended for LiteRT / TensorFlow Lite `.tflite` models.
- If the input file is not a valid `.tflite` artifact, the conversion path can fail
  before downstream analysis begins.

## Conversion-specific issues

### Conversion fails on model contents

- Reduce the issue to a smaller known-good `.tflite` model if possible.
- Check whether the model contains unsupported or awkward operator patterns for
  the conversion path.
- Treat the failure as a conversion issue first, not a target-performance issue.

### Downstream backend never receives a usable artifact

- Inspect the run output and logs to confirm whether conversion finished.
- If conversion completed, move debugging to the downstream backend.
- If conversion did not complete, focus on conversion and legalization details.

## Dependency-related issues

### Environment mismatch

- The frontend needs `mlia-converters-litert`; the converter's dependencies belong
  in the selected runtime environment.
- Use `mlia backend list --runtimes` to find that environment. Inspect it without
  activation using `uv pip list --python /path/to/runtime/venv/bin/python`.
- Repair the default converter with
  `mlia backend install tosa-converter-for-tflite --force`. For a custom selection,
  use its full runtime name instead of the backend name.
- Read the installation or worker log named in the error before retrying.

## Escalation path

If the converter appears healthy but the run still fails, move to the target or
estimator repo that consumes the produced artifact.
