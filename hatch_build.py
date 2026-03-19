# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Hatch build hook to download vendored artifacts at build time."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

VENDOR_DIR = Path("src/mlia/_vendor/artifacts/tosa-converter-for-tflite")
WHEEL_NAME = "tosa_converter_for_tflite-2025.11.0.dev0-cp310-cp310-linux_x86_64.whl"
SHA256_FILE = VENDOR_DIR / ".sha256"

ENV_URLS = "VENDORED_ARTIFACTS_URLS"
URL_KEY = "tosa-converter-for-tflite"
ENV_USER = "UV_INDEX_INTERNAL_USERNAME"
ENV_TOKEN = "UV_INDEX_INTERNAL_PASSWORD"


def _read_expected_sha256(path: Path) -> str:
    content = path.read_text(encoding="utf-8").strip()
    if not content:
        raise RuntimeError(f"Empty sha256 file: {path}")
    return content.split()[0]


def _file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_file(url: str, dest: Path) -> None:
    username = os.environ.get(ENV_USER)
    token = os.environ.get(ENV_TOKEN)
    if not username or not token:
        raise RuntimeError(
            f"Missing Artifactory credentials. Set {ENV_USER} and {ENV_TOKEN}."
        )

    req = urllib.request.Request(url)
    req.add_header("Username", username)
    req.add_header("X-JFrog-Art-Api", token)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(req) as resp, dest.open("wb") as out:  # nosec B310
        out.write(resp.read())


def _load_vendor_urls() -> dict[str, str]:
    raw = os.environ.get(ENV_URLS, "")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{ENV_URLS} must be valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{ENV_URLS} must be a JSON object mapping keys to URLs.")
    return data


class CustomBuildHook(BuildHookInterface):
    """Download vendored artifacts before building."""

    def initialize(self, version: str, build_data: dict) -> None:
        """Ensure the vendored wheel is present and has the expected hash."""
        del version, build_data
        expected_sha = _read_expected_sha256(SHA256_FILE)
        wheel_path = VENDOR_DIR / WHEEL_NAME

        if wheel_path.exists():
            actual_sha = _file_sha256(wheel_path)
            if actual_sha == expected_sha:
                return
            wheel_path.unlink()

        urls = _load_vendor_urls()
        url = urls.get(URL_KEY)
        if not url:
            raise RuntimeError(
                f"Missing vendor URL for '{URL_KEY}'. Set {ENV_URLS} with a "
                "JSON map including this key."
            )

        _download_file(url, wheel_path)
        actual_sha = _file_sha256(wheel_path)
        if actual_sha != expected_sha:
            wheel_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Downloaded vendor wheel hash mismatch. "
                f"Expected {expected_sha}, got {actual_sha}."
            )
