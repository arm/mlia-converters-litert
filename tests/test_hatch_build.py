# SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the Hatch metadata hook."""

from pathlib import Path
from unittest.mock import Mock
import pytest
from hatch_build import MetadataHook


def test_metadata_hook_uses_commit_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """README metadata links should target the current commit."""
    (tmp_path / "README.md").write_text(
        "[Docs](docs.md)\n![Image](image.png)\n[Section](#section)", encoding="utf-8"
    )
    (tmp_path / "docs.md").touch()
    (tmp_path / "image.png").touch()
    monkeypatch.setattr(
        "hatch_build.subprocess.run",
        Mock(
            side_effect=[
                Mock(stdout=f"{tmp_path}\n"),
                Mock(stdout="0123456789abcdef\n"),
            ]
        ),
    )
    metadata: dict = {"version": "0.1.0"}
    MetadataHook(str(tmp_path), {}).update(metadata)
    assert metadata["readme"]["text"] == (
        "[Docs](https://github.com/arm/mlia-converters-litert/blob/0123456789abcdef/docs.md)\n"
        "![Image](https://raw.githubusercontent.com/arm/mlia-converters-litert/0123456789abcdef/image.png)\n"
        "[Section](https://github.com/arm/mlia-converters-litert/blob/0123456789abcdef/README.md#section)"
    )


@pytest.mark.parametrize(
    ("version", "revision"),
    [("0.12.2", "v0.12.2"), ("0.12.3.dev9+40a5004", "40a5004")],
)
def test_metadata_hook_falls_back(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, version: str, revision: str
) -> None:
    """README links should use the version or its hash without Git metadata."""
    (tmp_path / "README.md").write_text("[Docs](docs.md)", encoding="utf-8")
    (tmp_path / "docs.md").touch()
    monkeypatch.setattr("hatch_build.subprocess.run", Mock(side_effect=OSError))
    metadata: dict = {"version": version}
    MetadataHook(str(tmp_path), {}).update(metadata)
    assert (
        metadata["readme"]["text"]
        == f"[Docs](https://github.com/arm/mlia-converters-litert/blob/{revision}/docs.md)"
    )
