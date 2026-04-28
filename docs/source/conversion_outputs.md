<!---
SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
SPDX-License-Identifier: Apache-2.0
-->

# Conversion Outputs and Diagnostics

## Overview

This repository is a converter plugin. Its job is to accept a supported input
model, produce a conversion-stage artifact, and hand that artifact to a later
MLIA backend. That means the important outputs here are conversion results and
diagnostics, not final target metrics.

## What this converter contributes

In a successful workflow, this converter contributes:

- A TOSA-oriented intermediate representation derived from a `.tflite` input.
- Conversion-stage logs or diagnostics during the MLIA run.
- Artifacts that a downstream backend can consume for further analysis.

## What success looks like

A successful end-to-end run usually looks like this:

1. The `.tflite` model is accepted.
2. The converter produces an intermediate artifact.
3. A downstream backend consumes that artifact.
4. The final user-facing metrics come from the downstream backend.

## What this repo does not own

This repo does not define the final performance or target-level compatibility
metrics for a run. If you are looking for values such as cycle counts, memory
figures, or target-specific compatibility summaries, those belong to the backend
that runs after conversion.

## Useful signals to look for

The most useful signals in this repo are usually:

- Whether the `.tflite` model was accepted and converted.
- Whether the produced artifact could be consumed by the next backend.
- Whether legalization or conversion failed before downstream analysis started.
- Whether the failure happened during conversion or after conversion completed.

## How to interpret failures

If conversion fails, investigate the model or conversion path first. If
conversion succeeds and later analysis fails, move the investigation to the
downstream target or estimator backend.
