from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest

from physrealx.config import load_mapping
from physrealx.diagnostics import inspect_runtime


def _fake_torch(
    *,
    cuda_runtime: str | None,
    available: bool,
    devices: tuple[str, ...] = (),
) -> ModuleType:
    module = ModuleType("torch")
    dynamic_module = cast(Any, module)
    dynamic_module.__version__ = "2.5.1+cu121"
    dynamic_module.version = SimpleNamespace(cuda=cuda_runtime)
    dynamic_module.cuda = SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: len(devices),
        get_device_name=lambda index: devices[index],
    )
    return module


def test_load_mapping_supports_json_yaml_unicode_and_literal_environment_values(
    tmp_path: Path,
) -> None:
    expected = {
        "name": "\u62bd\u5c49",
        "path": "$PHYSREALX_DATA/asset.usd",
        "nested": {"enabled": True, "count": 3},
    }
    json_path = tmp_path / "config.json"
    yaml_path = tmp_path / "config.YAML"
    json_path.write_text(json.dumps(expected, ensure_ascii=False), encoding="utf-8")
    yaml_path.write_text(
        (
            "name: \u62bd\u5c49\n"
            "path: $PHYSREALX_DATA/asset.usd\n"
            "nested:\n"
            "  enabled: true\n"
            "  count: 3\n"
        ),
        encoding="utf-8",
    )

    assert load_mapping(json_path) == expected
    assert load_mapping(yaml_path) == expected


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("config.txt", "key=value", "unsupported configuration format"),
        ("array.json", "[]", "configuration root must be a mapping"),
        ("array.yaml", "- one\n- two\n", "configuration root must be a mapping"),
    ],
)
def test_load_mapping_rejects_unsupported_formats_and_non_mappings(
    name: str,
    content: str,
    message: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_mapping(path)


def test_load_mapping_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mapping(tmp_path / "missing.yaml")


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("duplicate.json", '{"method": "one", "method": "two"}'),
        ("duplicate.yml", "method: one\nmethod: two\n"),
    ],
)
def test_load_mapping_rejects_duplicate_keys(
    name: str,
    content: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate key"):
        load_mapping(path)


def test_runtime_diagnostics_reports_ready_cu121_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _fake_torch(
        cuda_runtime="12.1",
        available=True,
        devices=("NVIDIA RTX 4090", "NVIDIA RTX 4090"),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "PhysReal-X")

    report = inspect_runtime()

    assert report.torch_installed
    assert report.torch_version == "2.5.1+cu121"
    assert report.torch_cuda_runtime == "12.1"
    assert report.cuda_available
    assert report.cuda_device_count == 2
    assert report.cuda_devices == ("NVIDIA RTX 4090", "NVIDIA RTX 4090")
    assert report.cu121_ready
    assert not report.issues
    assert report.to_dict()["platform"] == report.platform


def test_runtime_diagnostics_collects_runtime_cuda_and_environment_issues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "torch",
        _fake_torch(cuda_runtime=None, available=False),
    )
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "another-environment")

    report = inspect_runtime()

    assert not report.cuda_available
    assert not report.cu121_ready
    assert report.issues == (
        "PyTorch was not compiled for CUDA 12.1 (reported CPU-only)",
        "CUDA is not available to PyTorch",
        "active Conda environment is 'another-environment', expected 'PhysReal-X'",
    )


def test_runtime_diagnostics_can_skip_cuda_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    class UnprobeableCuda:
        @staticmethod
        def is_available() -> bool:
            raise AssertionError("CUDA must not be probed")

    torch = _fake_torch(cuda_runtime="12.1", available=False)
    cast(Any, torch).cuda = UnprobeableCuda()
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setenv("CONDA_DEFAULT_ENV", "PhysReal-X")

    report = inspect_runtime(probe_cuda=False)

    assert report.torch_installed
    assert not report.cuda_available
    assert not report.cu121_ready
    assert not report.issues


def test_runtime_diagnostics_handles_missing_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.delenv("CONDA_DEFAULT_ENV", raising=False)

    report = inspect_runtime()

    assert not report.torch_installed
    assert report.torch_version is None
    assert report.issues == ("PyTorch is not installed; GPU training adapters are unavailable",)
