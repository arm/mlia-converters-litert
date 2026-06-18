# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the TOSA Converter for TFLite."""

from __future__ import annotations

from pathlib import Path
from collections.abc import Generator
from unittest.mock import MagicMock

import pytest

from mlia.backend.tosa_converter_for_tflite.conversion import TosaConverterForTflite
from mlia.backend.tosa_converter_for_tflite import install as tosa_install
from mlia.backend.install import InstallFromVendorPackage


# mypy: disable-error-code=misc
@pytest.fixture(name="tosa_converter_for_tflite")
def fixture_tosa_converter_for_tflite() -> Generator[
    TosaConverterForTflite, None, None
]:
    """Create an instance of the TOSA Converter for TFLite."""
    tosa_converter_for_tflite = TosaConverterForTflite()
    yield tosa_converter_for_tflite


def test_tosa_converter_for_tflite(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test for class TosaConverterForTflite."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    model_file = tmp_path / "model.tflite"
    model_file.touch()

    monkeypatch.setattr(
        "mlia.backend.tosa_converter_for_tflite.conversion.process_command_output",
        MagicMock(
            side_effect=lambda *args: (
                output_dir / f"{model_file.stem}.tosamlir"
            ).touch()
        ),
    )
    monkeypatch.setattr(
        "mlia.backend.tosa_converter_for_tflite.conversion.ensure_tosa_converter_for_tflite_installed",
        MagicMock(return_value=("tosa-converter-for-tflite",)),
    )
    result = tosa_converter_for_tflite(model_file, output_dir)

    assert result == output_dir / f"{model_file.stem}.tosamlir"


def test_tosa_converter_for_tflite_supports_tflite_to_tosa(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
) -> None:
    """Accept a TFLite model path when targeting TOSA with no kwargs."""
    model_file = tmp_path / "model.tflite"

    assert tosa_converter_for_tflite.supports(model_file, "tosa", {}) is True


@pytest.mark.parametrize(
    ("model", "target_format"),
    [
        pytest.param("model.tflite", "tosa", id="model-must-be-path"),
        pytest.param(Path("model.mlir"), "tosa", id="model-must-be-tflite"),
        pytest.param(Path("model.tflite"), "torch", id="target-must-be-tosa"),
    ],
)
def test_tosa_converter_for_tflite_supports_rejects_invalid_model_or_target(
    tosa_converter_for_tflite: TosaConverterForTflite,
    model: object,
    target_format: str,
) -> None:
    """Reject unsupported source/target combinations."""
    assert tosa_converter_for_tflite.supports(model, target_format, {}) is False


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"enable_quantization": True}, id="enable-quantization"),
        pytest.param({"example_inputs": ["input.npy"]}, id="example-inputs"),
    ],
)
def test_tosa_converter_for_tflite_supports_rejects_unsupported_kwargs(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
    kwargs: dict[str, object],
) -> None:
    """Reject kwargs that are not supported by the wrapper."""
    model_file = tmp_path / "model.tflite"

    assert tosa_converter_for_tflite.supports(model_file, "tosa", kwargs) is False


def test_tosa_converter_for_tflite_no_output_dir(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
) -> None:
    """Test for class TosaConverterForTflite with an invalid output directory."""
    with pytest.raises(NotADirectoryError):
        tosa_converter_for_tflite(tmp_path / "model.tflite", tmp_path / "output")


def test_tosa_converter_for_tflite_front_end_fail(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
) -> None:
    """Test for class TosaConverterForTflite with a missing input file."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    with pytest.raises(FileNotFoundError):
        tosa_converter_for_tflite(tmp_path / "model.tflite", output_dir)


def test_tosa_converter_for_tflite_front_end_no_output(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test for class TosaConverterForTflite when no output is produced."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    model_file = tmp_path / "model.tflite"
    model_file.touch()

    monkeypatch.setattr(
        "mlia.backend.tosa_converter_for_tflite.conversion.process_command_output",
        MagicMock(),
    )
    monkeypatch.setattr(
        "mlia.backend.tosa_converter_for_tflite.conversion.ensure_tosa_converter_for_tflite_installed",
        MagicMock(return_value=("tosa-converter-for-tflite",)),
    )

    with pytest.raises(FileNotFoundError):
        tosa_converter_for_tflite(model_file, output_dir)


def test_ensure_tosa_converter_for_tflite_installed_missing_vendor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail with a clear message when the vendored wheel is missing."""
    monkeypatch.setattr(tosa_install.shutil, "which", lambda _: None)
    monkeypatch.setattr(tosa_install, "_resolve_executable", lambda *_: None)
    monkeypatch.setattr(tosa_install, "_module_available", lambda *_: False)
    monkeypatch.setattr(
        tosa_install, "get_tosa_converter_for_tflite_backend_installation", MagicMock()
    )

    installation = tosa_install.get_tosa_converter_for_tflite_backend_installation()
    installation.supports.return_value = False

    with pytest.raises(
        RuntimeError, match="vendored 'tosa-converter-for-tflite' wheel"
    ):
        tosa_install.ensure_tosa_converter_for_tflite_installed()


def test_ensure_tosa_converter_for_tflite_installed_resolves_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return the executable when it is already on PATH."""
    monkeypatch.setattr(
        tosa_install,
        "_resolve_executable",
        lambda *_: "/tmp/tosa-converter-for-tflite",
    )
    monkeypatch.setattr(tosa_install, "_module_available", lambda *_: False)
    tosa_install.ensure_tosa_converter_for_tflite_installed.cache_clear()

    assert tosa_install.ensure_tosa_converter_for_tflite_installed() == (
        "/tmp/tosa-converter-for-tflite",
    )


def test_ensure_tosa_converter_for_tflite_installed_module_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return module invocation when the console script is missing."""
    monkeypatch.setattr(tosa_install, "_resolve_executable", lambda *_: None)
    monkeypatch.setattr(tosa_install, "_module_available", lambda *_: True)
    tosa_install.ensure_tosa_converter_for_tflite_installed.cache_clear()

    assert tosa_install.ensure_tosa_converter_for_tflite_installed() == (
        tosa_install.sys.executable,
        "-m",
        "tosa_converter_for_tflite.cli",
    )


def test_ensure_tosa_converter_for_tflite_installed_vendor_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install vendored wheel when no executable or module is available."""
    resolve = MagicMock(side_effect=[None, "/tmp/tosa-converter-for-tflite"])
    monkeypatch.setattr(tosa_install, "_resolve_executable", resolve)
    monkeypatch.setattr(tosa_install, "_module_available", lambda *_: False)
    monkeypatch.setattr(tosa_install.importlib, "invalidate_caches", MagicMock())

    installation = MagicMock()
    installation.supports.return_value = True
    installation.install = MagicMock()
    monkeypatch.setattr(
        tosa_install,
        "get_tosa_converter_for_tflite_backend_installation",
        lambda: installation,
    )

    tosa_install.ensure_tosa_converter_for_tflite_installed.cache_clear()
    result = tosa_install.ensure_tosa_converter_for_tflite_installed()

    assert result == ("/tmp/tosa-converter-for-tflite",)
    installation.install.assert_called_once()
    assert isinstance(installation.install.call_args.args[0], InstallFromVendorPackage)


def test_tosa_converter_for_tflite_module_available_missing_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return False when a parent package is missing for a submodule."""

    def _raise(*_args: object, **_kwargs: object) -> None:
        exc = ModuleNotFoundError("No module named 'tosa_converter_for_tflite'")
        exc.name = "tosa_converter_for_tflite"
        raise exc

    monkeypatch.setattr(tosa_install.importlib, "import_module", _raise)

    assert tosa_install._module_available("tosa_converter_for_tflite.cli") is False


def test_tosa_converter_for_tflite_create_front_end_command(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
) -> None:
    """Test for function _create_front_end_command of TosaConverterForTflite."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    in_file = tmp_path / "in"
    out_file = tmp_path / "out"

    cmd = tosa_converter_for_tflite._create_converter_command(
        in_file, out_file, ("tosa-converter-for-tflite",)
    )

    assert cmd.cmd
    assert all(isinstance(arg, str) for arg in cmd.cmd)
    assert str(in_file) in cmd.cmd
    assert str(out_file) in cmd.cmd
    assert "--text" in cmd.cmd
