from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
PROJECT_SOURCE = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_SOURCE))

from physrealx.external_workers import WorkerDeploymentFile  # noqa: E402

INSTALLER_SPEC = importlib.util.spec_from_file_location(
    "physrealx_worker_installer",
    PROJECT_ROOT / "workers" / "install.py",
)
assert INSTALLER_SPEC is not None and INSTALLER_SPEC.loader is not None
installer = importlib.util.module_from_spec(INSTALLER_SPEC)
sys.modules[INSTALLER_SPEC.name] = installer
INSTALLER_SPEC.loader.exec_module(installer)


def _executable(path: Path, version: tuple[int, int]) -> Path:
    path.write_text(
        f"#!/usr/bin/env python3\nimport json\nprint(json.dumps([{version[0]}, {version[1]}]))\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def _omnidc_manifest(path: Path, **updates: Any) -> Path:
    document: dict[str, Any] = {
        "schema": "physrealx.omnidc-installation/1",
        "source_uri": "https://github.com/princeton-vl/OMNI-DC",
        "source_revision": "a" * 40,
        "source_root": "/opt/omnidc",
        "source_tree_digest": f"sha256:{'b' * 64}",
        "runtime_files": [],
        "checkpoint": {},
        "self_test": {},
    }
    document.update(updates)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_worker_catalog_and_entrypoints_exclude_removed_toy_workers() -> None:
    assert set(installer.WORKERS) == {
        "coacd",
        "colmap",
        "ditto",
        "metric_alignment",
        "mujoco_sysid",
        "omnidc",
        "openusd",
    }
    pyproject = (PROJECT_ROOT / "workers" / "pyproject.toml").read_text(encoding="utf-8")
    assert "physrealx-worker-ditto" in pyproject
    assert "physrealx-worker-metric-alignment" in pyproject
    assert "physrealx-worker-omnidc" in pyproject
    assert "physrealx-worker-mujoco =" not in pyproject
    assert "physrealx-worker-blenderproc" not in pyproject
    assert "physrealx-worker-lerobot" not in pyproject
    assert installer.WORKERS["coacd"].input_contract == "VisualMesh/1"
    assert (
        installer.WORKERS["coacd"].output_contract == "CollisionGeometry/1+CollisionBuildReport/1"
    )


def test_ditto_binding_pins_official_runtime_and_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = _executable(tmp_path / "python3.8", (3, 8))
    repository = tmp_path / "Ditto"
    repository.mkdir()
    tracked = repository / "LICENSE"
    tracked.write_text("MIT", encoding="utf-8")
    checkpoint = tmp_path / "ditto.ckpt"
    checkpoint.write_bytes(b"official-checkpoint-fixture")

    def git_text(_repository: Path, *arguments: str) -> str:
        values = {
            ("rev-parse", "HEAD"): installer._DITTO_REVISION,
            ("remote", "get-url", "origin"): "https://github.com/UT-Austin-RPL/Ditto.git",
            ("status", "--porcelain", "--untracked-files=no"): "",
        }
        return values[arguments]

    monkeypatch.setattr(installer, "_git_text", git_text)
    monkeypatch.setattr(installer, "_ditto_tracked_files", lambda _repository: (tracked,))

    binding = installer._ditto_binding(
        python,
        repository,
        checkpoint,
        "Ditto_s2m",
    )

    assignments = dict(binding.environment_assignments)
    expected_digest = f"sha256:{hashlib.sha256(checkpoint.read_bytes()).hexdigest()}"
    assert assignments == {
        "PHYSREALX_DITTO_PYTHON": str(python.resolve()),
        "PHYSREALX_DITTO_REPOSITORY": str(repository.resolve()),
        "PHYSREALX_DITTO_CHECKPOINT": str(checkpoint.resolve()),
        "PHYSREALX_DITTO_CHECKPOINT_DIGEST": expected_digest,
        "PHYSREALX_DITTO_EXPERIMENT": "Ditto_s2m",
    }
    assert binding.accelerator is installer.MethodAccelerator.CUDA
    assert binding.maturity is installer.MethodMaturity.EXPERIMENTAL
    assert binding.external_runtime_files == (
        python.resolve(),
        checkpoint.resolve(),
        tracked,
    )


def test_ditto_binding_rejects_missing_or_non_python38_runtime(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="requires --ditto-python"):
        installer._ditto_binding(None, None, None, "Ditto_s2m")

    python = _executable(tmp_path / "python3.9", (3, 9))
    repository = tmp_path / "Ditto"
    repository.mkdir()
    checkpoint = tmp_path / "ditto.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(RuntimeError, match=r"must use Python 3.8"):
        installer._ditto_binding(python, repository, checkpoint, "Ditto_s2m")


def test_ditto_binding_rejects_unpinned_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python = _executable(tmp_path / "python3.8", (3, 8))
    repository = tmp_path / "Ditto"
    repository.mkdir()
    checkpoint = tmp_path / "ditto.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        installer,
        "_git_text",
        lambda _repository, *_arguments: "d" * 40,
    )
    with pytest.raises(RuntimeError, match="pinned revision"):
        installer._ditto_binding(python, repository, checkpoint, "Ditto_s2m")


def test_omnidc_binding_requires_exact_versioned_manifest(tmp_path: Path) -> None:
    valid = _omnidc_manifest(tmp_path / "installation.json")
    binding = installer._omnidc_binding(valid)
    assert binding.definition.source_revision == "a" * 40
    assert binding.worker_arguments == (
        "--installation-manifest",
        str(valid.resolve()),
    )
    assert binding.accelerator is installer.MethodAccelerator.CUDA
    assert binding.maturity is installer.MethodMaturity.EXPERIMENTAL

    unknown = _omnidc_manifest(tmp_path / "unknown.json", unexpected=True)
    with pytest.raises(RuntimeError, match="fields are invalid"):
        installer._omnidc_binding(unknown)
    mutable = _omnidc_manifest(tmp_path / "mutable.json", source_revision="main")
    with pytest.raises(RuntimeError, match="full immutable Git commit"):
        installer._omnidc_binding(mutable)


def test_install_writes_deterministic_ditto_process_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    module = project / "workers" / "src" / "physrealx_workers" / "ditto_worker.py"
    module.parent.mkdir(parents=True)
    module.write_text("def main(): return 0\n", encoding="utf-8")
    requirements = project / "workers" / "requirements" / "ditto.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("numpy==1.26.4\n", encoding="utf-8")
    root = tmp_path / "environments"
    adapter_python = root / "ditto" / "bin" / "python"
    adapter_python.parent.mkdir(parents=True)
    adapter_python.write_bytes(b"python")
    adapter_python.chmod(0o755)
    workspace = tmp_path / "workspace"
    checkpoint = tmp_path / "ditto.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    repository_file = tmp_path / "Ditto" / "LICENSE"
    repository_file.parent.mkdir()
    repository_file.write_text("MIT", encoding="utf-8")
    binding = installer.WorkerBinding(
        definition=installer.WORKERS["ditto"],
        environment_assignments=(
            ("PHYSREALX_DITTO_PYTHON", "/opt/ditto/bin/python"),
            ("PHYSREALX_DITTO_REPOSITORY", "/opt/ditto/source"),
            ("PHYSREALX_DITTO_CHECKPOINT", str(checkpoint)),
            ("PHYSREALX_DITTO_CHECKPOINT_DIGEST", f"sha256:{'c' * 64}"),
            ("PHYSREALX_DITTO_EXPERIMENT", "Ditto_s2m"),
        ),
        external_runtime_files=(checkpoint, repository_file),
        accelerator=installer.MethodAccelerator.CUDA,
        maturity=installer.MethodMaturity.EXPERIMENTAL,
    )
    monkeypatch.setattr(installer, "_worker_binding", lambda *_args, **_kwargs: binding)
    monkeypatch.setattr(installer, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(installer, "_python_version", lambda _python: "3.11")
    monkeypatch.setattr(
        installer,
        "_environment_record",
        lambda *_args, **_kwargs: (
            b'{"schema":"physrealx.worker-environment/1","protocol_version":"2","runtime_files":[]}'
        ),
    )

    installer._install(
        "ditto",
        project,
        root,
        workspace,
        colmap_binary=None,
        ditto_python=None,
        ditto_repository=None,
        ditto_checkpoint=None,
        ditto_experiment="Ditto_s2m",
        omnidc_installation_manifest=None,
        register_only=True,
    )

    deployment_path = workspace / "workers.d" / "ditto.json"
    validated = WorkerDeploymentFile.model_validate_json(deployment_path.read_bytes())
    assert validated.worker.descriptor.name == "ditto"
    deployment = json.loads(deployment_path.read_text())["worker"]
    command = deployment["command"]
    assert Path(command[0]).name == "env"
    assert command[1:6] == [f"{name}={value}" for name, value in binding.environment_assignments]
    assert command[6:] == [str(adapter_python), str(module)]
    assert deployment["probe_command"] == [*command, "--probe"]
    assert deployment["inherit_environment"] == [
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
    ]
    descriptor = deployment["descriptor"]
    assert descriptor["maturity"] == "experimental"
    assert descriptor["runtime"]["accelerator"] == "cuda"
    assert descriptor["runtime"]["python"] == "3.11"
