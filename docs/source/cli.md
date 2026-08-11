<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# CLI Guide

This repo does not introduce a separate top-level command. The converter is used
through MLIA runs that start from a `.tflite` model.

## Transformer naming

When you need to refer to this converter explicitly:

- Use `tflite_to_tosa` as the transformer name in MLIA configuration or
  diagnostics.
- Treat `tosa_converter_for_tflite` as the implementation package name, not the
  CLI name.

## Typical automatic use

In the normal path, MLIA selects the converter automatically when the input and
downstream backend require it:

```bash
mlia check model.tflite --target-profile <target-profile> --performance
```

## Debug the conversion path

If you want to make the downstream analysis path explicit, pin the target
backend and let MLIA choose the `.tflite` transformer when it needs a TOSA
artifact:

```bash
mlia check model.tflite \
  --target-profile <target-profile> \
  --performance \
  --backend <downstream-backend>
```

## Practical debugging sequence

When a LiteRT / TensorFlow Lite `.tflite`-driven run fails, a useful sequence is:

1. Confirm the downstream target and backend plugins are installed.
2. Rerun with an explicit downstream backend to reduce ambiguity.
3. Inspect the wider MLIA error, transformer selection, and backend selection.
4. Use the troubleshooting page for conversion-stage failures.
