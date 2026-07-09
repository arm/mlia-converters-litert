# SPDX-FileCopyrightText: Copyright 2023-2026, Arm Limited and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the TOSA Converter for TFLite."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from mlia.backend.install import DownloadAndInstall, InstallFromVendorPackage
from mlia.backend.tosa_converter_for_tflite import install as tosa_install
from mlia.backend.tosa_converter_for_tflite.conversion import TosaConverterForTflite


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
                output_dir / f"{model_file.stem}.tosa.mlirbc"
            ).touch()
        ),
    )
    monkeypatch.setattr(
        "mlia.backend.tosa_converter_for_tflite.conversion.ensure_tosa_converter_for_tflite_installed",
        MagicMock(return_value=("tosa-converter-for-tflite",)),
    )
    result = tosa_converter_for_tflite(model_file, output_dir)

    assert result == output_dir / f"{model_file.stem}.tosa.mlirbc"


def test_tosa_converter_for_tflite_supports_tflite_to_tosa(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
) -> None:
    """Accept a TFLite model path when targeting TOSA with no kwargs."""
    model_file = tmp_path / "model.tflite"

    assert tosa_converter_for_tflite.supports(model_file, "tosa", {}) is True


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"output_format": "mlir-text"}, id="text"),
        pytest.param({"output_format": "mlir-bytecode"}, id="bytecode"),
        pytest.param({"emit_debug_info": True}, id="debug-info"),
        pytest.param(
            {"output_format": "mlir-bytecode", "emit_debug_info": True},
            id="bytecode-debug-info",
        ),
    ],
)
def test_tosa_converter_for_tflite_supports_output_options(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
    kwargs: dict[str, object],
) -> None:
    """Accept output options used by Neural Technology transformer requests."""
    model_file = tmp_path / "model.tflite"

    assert tosa_converter_for_tflite.supports(model_file, "tosa", kwargs) is True


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
        pytest.param({"output_format": "unsupported"}, id="unsupported-format"),
        pytest.param({"emit_debug_info": "yes"}, id="invalid-debug-info"),
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


def test_tosa_converter_for_tflite_installation_uses_public_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install metadata should target the public package, not a vendored wheel."""
    package_manager = MagicMock()
    monkeypatch.setattr(
        "mlia.backend.install.get_package_manager", lambda: package_manager
    )

    installation = tosa_install.get_tosa_converter_for_tflite_backend_installation()

    assert not installation.supports(InstallFromVendorPackage())
    assert installation.supports(DownloadAndInstall())

    installation.install(DownloadAndInstall())

    package_manager.install.assert_called_once_with(
        ["tosa-converter-for-tflite==2026.2.0"]
    )


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


def test_ensure_tosa_converter_for_tflite_installed_public_package_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Install public package when no executable or module is available."""
    resolve = MagicMock(side_effect=[None, "/tmp/tosa-converter-for-tflite"])
    monkeypatch.setattr(tosa_install, "_resolve_executable", resolve)
    monkeypatch.setattr(tosa_install, "_module_available", lambda *_: False)
    monkeypatch.setattr(tosa_install.importlib, "invalidate_caches", MagicMock())

    installation = MagicMock()
    installation.supports.side_effect = lambda install_type: isinstance(
        install_type, DownloadAndInstall
    )
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
    assert isinstance(installation.install.call_args.args[0], DownloadAndInstall)


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
    assert "--bytecode" in cmd.cmd
    assert "--text" not in cmd.cmd


def test_tosa_converter_for_tflite_create_text_front_end_command(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
) -> None:
    """Text output should remain available for compatibility analysis."""
    in_file = tmp_path / "in"
    out_file = tmp_path / "out"

    cmd = tosa_converter_for_tflite._create_converter_command(
        in_file,
        out_file,
        ("tosa-converter-for-tflite",),
        output_format="mlir-text",
    )

    assert "--text" in cmd.cmd
    assert "--bytecode" not in cmd.cmd
    assert "--emit-debug-info" not in cmd.cmd


def test_tosa_converter_for_tflite_create_text_front_end_command_with_debug_info(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
) -> None:
    """Text output can request source locations for compatibility analysis."""
    in_file = tmp_path / "in"
    out_file = tmp_path / "out"

    cmd = tosa_converter_for_tflite._create_converter_command(
        in_file,
        out_file,
        ("tosa-converter-for-tflite",),
        output_format="mlir-text",
        emit_debug_info=True,
    )

    assert "--text" in cmd.cmd
    assert "--emit-debug-info" in cmd.cmd


def test_tosa_converter_for_tflite_outputs_mlir_bytecode_for_model_converter(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TFLite performance path should feed model-converter accepted MLIR bytecode."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    model_file = tmp_path / "model.tflite"
    model_file.touch()

    def _create_expected_output(*_args: object) -> None:
        (output_dir / f"{model_file.stem}.tosa.mlirbc").touch()

    process_mock = MagicMock(side_effect=_create_expected_output)
    monkeypatch.setattr(
        "mlia.backend.tosa_converter_for_tflite.conversion.process_command_output",
        process_mock,
    )

    tosa_file = tosa_converter_for_tflite._run_converter(
        model_file, output_dir, ("tosa-converter-for-tflite",)
    )

    assert tosa_file == output_dir / "model.tosa.mlirbc"
    cmd = process_mock.call_args.args[0]
    assert str(tosa_file) in cmd.cmd
    assert "--bytecode" in cmd.cmd
    assert "--text" not in cmd.cmd


def test_tosa_converter_for_tflite_outputs_text_for_compatibility(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compatibility path should be able to request parseable textual MLIR."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    model_file = tmp_path / "model.tflite"
    model_file.touch()

    def _create_expected_output(*_args: object) -> None:
        (output_dir / f"{model_file.stem}.tosamlir").touch()

    process_mock = MagicMock(side_effect=_create_expected_output)
    monkeypatch.setattr(
        "mlia.backend.tosa_converter_for_tflite.conversion.process_command_output",
        process_mock,
    )

    tosa_file = tosa_converter_for_tflite._run_converter(
        model_file,
        output_dir,
        ("tosa-converter-for-tflite",),
        output_format="mlir-text",
        emit_debug_info=True,
    )

    assert tosa_file == output_dir / "model.tosamlir"
    cmd = process_mock.call_args.args[0]
    assert str(tosa_file) in cmd.cmd
    assert "--text" in cmd.cmd
    assert "--emit-debug-info" in cmd.cmd
    assert "--bytecode" not in cmd.cmd


def test_tosa_converter_for_tflite_patches_missing_rescale_rounding_mode(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Add missing rescale rounding mode to textual TOSA MLIR output."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    model_file = tmp_path / "model.tflite"
    model_file.touch()
    output_file = output_dir / f"{model_file.stem}.tosamlir"

    def _write_tosa_mlir(*_args: object) -> None:
        output_file.write_text(
            "\n".join(
                [
                    (
                        "%0 = tosa.rescale %arg0 "
                        "{input_unsigned = false, output_unsigned = false, "
                        "per_channel = false, scale32 = true} : "
                        "(tensor<1x8xi32>) -> tensor<1x8xi8>"
                    ),
                    (
                        "%1 = tosa.rescale %arg1 "
                        "{input_unsigned = false, output_unsigned = false, "
                        'per_channel = false, rounding_mode = "DOUBLE_ROUND", '
                        "scale32 = true} : "
                        "(tensor<1x8xi32>) -> tensor<1x8xi8>"
                    ),
                    (
                        "%2 = tosa.rescale %arg2 "
                        "{input_unsigned = false, output_unsigned = false, "
                        "per_channel = false, rounding_mode = SINGLE_ROUND, "
                        "scale32 = true} : "
                        "(tensor<1x8xi32>) -> tensor<1x8xi8>"
                    ),
                    (
                        "%3 = tosa.resize %arg3, %scale, %offset, %border "
                        "{} : (tensor<1x4x8x16xi8>, !tosa.shape<4>, "
                        "!tosa.shape<2>, !tosa.shape<2>) -> "
                        "tensor<1x8x16x16xi8> loc(#loc1)"
                    ),
                    (
                        "%4 = tosa.resize %arg4, %scale, %offset, %border "
                        "{} : (tensor<1x4x8x16xi8>, !tosa.shape<4>, "
                        "!tosa.shape<2>, !tosa.shape<2>) -> "
                        "tensor<1x8x16x16xi32> loc(#loc2)"
                    ),
                    (
                        "%5 = tosa.resize %arg5, %scale, %offset, %border "
                        '{mode = "NEAREST_NEIGHBOR"} : '
                        "(tensor<1x4x8x16xi8>, !tosa.shape<4>, "
                        "!tosa.shape<2>, !tosa.shape<2>) -> "
                        "tensor<1x8x16x16xi8>"
                    ),
                    (
                        "%6 = tosa.resize %arg6, %scale, %offset, %border "
                        "{} : (tensor<1x4x8x16xf32>, !tosa.shape<4>, "
                        "!tosa.shape<2>, !tosa.shape<2>) -> "
                        "tensor<1x8x16x16xf32>"
                    ),
                    (
                        "%7 = tosa.add %arg0, %arg1 : "
                        "(tensor<1xi32>, tensor<1xi32>) -> tensor<1xi32>"
                    ),
                    '#loc1 = loc("model/resize/ResizeNearestNeighbor"(#loc))',
                    '#loc2 = loc("model/resize/ResizeBilinear"(#loc))',
                ]
            )
        )

    monkeypatch.setattr(
        "mlia.backend.tosa_converter_for_tflite.conversion.process_command_output",
        _write_tosa_mlir,
    )
    monkeypatch.setattr(
        "mlia.backend.tosa_converter_for_tflite.conversion.ensure_tosa_converter_for_tflite_installed",
        MagicMock(return_value=("tosa-converter-for-tflite",)),
    )

    assert (
        tosa_converter_for_tflite(
            model_file,
            output_dir,
            output_format="mlir-text",
        )
        == output_file
    )

    contents = output_file.read_text()
    assert (
        "{input_unsigned = false, output_unsigned = false, "
        "per_channel = false, scale32 = true, rounding_mode = DOUBLE_ROUND}"
    ) in contents
    assert 'rounding_mode = "DOUBLE_ROUND"' not in contents
    assert contents.count("rounding_mode = SINGLE_ROUND") == 1
    assert contents.count("rounding_mode = DOUBLE_ROUND") == 2
    assert (
        "tosa.resize %arg3, %scale, %offset, %border {mode = NEAREST_NEIGHBOR}"
        in contents
    )
    assert "tosa.resize %arg4, %scale, %offset, %border {mode = BILINEAR}" in contents
    assert 'mode = "NEAREST_NEIGHBOR"' not in contents
    assert contents.count("mode = NEAREST_NEIGHBOR") == 2
    assert contents.count("mode = BILINEAR") == 1
    assert (
        "%6 = tosa.resize %arg6, %scale, %offset, %border {} : "
        "(tensor<1x4x8x16xf32>, !tosa.shape<4>, !tosa.shape<2>, "
        "!tosa.shape<2>) -> tensor<1x8x16x16xf32>"
    ) in contents


def test_tosa_converter_for_tflite_does_not_patch_bytecode_output(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave bytecode output untouched."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    model_file = tmp_path / "model.tflite"
    model_file.touch()
    output_file = output_dir / f"{model_file.stem}.tosa.mlirbc"

    monkeypatch.setattr(
        "mlia.backend.tosa_converter_for_tflite.conversion.process_command_output",
        MagicMock(side_effect=lambda *_args: output_file.write_bytes(b"bytecode")),
    )
    monkeypatch.setattr(
        "mlia.backend.tosa_converter_for_tflite.conversion.ensure_tosa_converter_for_tflite_installed",
        MagicMock(return_value=("tosa-converter-for-tflite",)),
    )

    assert tosa_converter_for_tflite(model_file, output_dir) == output_file
    assert output_file.read_bytes() == b"bytecode"


def test_tosa_converter_for_tflite_bytecode_output_path(
    tosa_converter_for_tflite: TosaConverterForTflite,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bytecode output should use the MLIR bytecode filename."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    model_file = tmp_path / "model.tflite"
    model_file.touch()

    monkeypatch.setattr(
        "mlia.backend.tosa_converter_for_tflite.conversion.process_command_output",
        MagicMock(
            side_effect=lambda *args: (
                output_dir / f"{model_file.stem}.tosa.mlirbc"
            ).touch()
        ),
    )
    monkeypatch.setattr(
        "mlia.backend.tosa_converter_for_tflite.conversion.ensure_tosa_converter_for_tflite_installed",
        MagicMock(return_value=("tosa-converter-for-tflite",)),
    )

    result = tosa_converter_for_tflite(
        model_file,
        output_dir,
        output_format="mlir-bytecode",
        emit_debug_info=True,
    )

    assert result == output_dir / f"{model_file.stem}.tosa.mlirbc"
