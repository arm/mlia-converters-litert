<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
--->

# Conversion Flow

## Supported input

This repo is focused on TensorFlow Lite models in `.tflite` form.

## What the converter does

At a high level, the converter:

- Reads the TFLite model representation.
- Maps supported operations into the TOSA-oriented path used by MLIA.
- Prepares artifacts suitable for downstream MLIA backends.

## Operational model

In most workflows this converter is a dependency backend. You usually see the
results of the later analysis backend rather than interacting with this package
as a destination in its own right.

## Maintenance considerations

When changing the conversion path, keep registration, conversion behaviour, and
any downstream assumptions aligned so the rest of the MLIA pipeline continues to
consume the produced artifacts correctly.
