#!/usr/bin/env python3
"""Create isolated worker environments and workspace deployment manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import venv
from dataclasses import dataclass
from pathlib import Path

from physrealx.contracts import API_VERSION
from physrealx.domain import (
    MethodAccelerator,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodMaturity,
    MethodRuntime,
)


@dataclass(frozen=True)
class WorkerDefinition:
    name: str
    capability: MethodCapability
    version: str
    source_uri: str
    source_revision: str
    license_spdx: str
    input_contract: str
    output_contract: str


WORKERS = {
    "coacd": WorkerDefinition(
        "coacd",
        MethodCapability.COLLISION_BUILD,
        "1.0.14",
        "https://github.com/SarahWeiii/CoACD",
        "v1.0.14",
        "MIT",
        "VisualMesh",
        "CollisionGeometry",
    ),
    "mujoco": WorkerDefinition(
        "mujoco",
        MethodCapability.SIMULATION_RUN,
        "3.12.0",
        "https://github.com/google-deepmind/mujoco",
        "3.12.0",
        "Apache-2.0",
        "SceneTaskPair",
        "StateTrajectory+SimulationReport",
    ),
    "blenderproc": WorkerDefinition(
        "blenderproc",
        MethodCapability.RENDER_GENERATE,
        "2.8.0",
        "https://github.com/DLR-RM/BlenderProc",
        "v2.8.0",
        "GPL-3.0-only",
        "StateTrajectory",
        "RenderStreams+CameraMetadata",
    ),
    "lerobot": WorkerDefinition(
        "lerobot_v2_1",
        MethodCapability.EPISODE_PACK,
        "0.3.3",
        "https://github.com/huggingface/lerobot",
        "v0.3.3",
        "Apache-2.0",
        "StateTrajectory",
        "EpisodeBundle+EpisodeManifest",
    ),
}


def _run(command: list[str], cwd: Path) -> None:
    environment = dict(os.environ)
    environment.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    environment.setdefault("PIP_DEFAULT_TIMEOUT", "180")
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def _install_blender_runtime(
    environment: Path,
    runtime: Path,
    project: Path,
) -> None:
    """Install the protocol and adapter into Blender's isolated Python path."""

    blenderproc = environment / "bin" / "blenderproc"
    _run(
        [
            str(blenderproc),
            "pip",
            "--custom-blender-path",
            str(runtime),
            "install",
            "numpy==1.26.4",
            "trimesh==4.4.0",
            "scikit-learn==1.5.0",
            str(project),
            str(project / "workers"),
        ],
        project,
    )


def _environment_record(
    environment: Path,
    project: Path,
    worker: str,
    blender_runtime: Path | None,
) -> bytes:
    python = environment / "bin" / "python"
    freeze = subprocess.run(
        [str(python), "-m", "pip", "freeze", "--all"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.splitlines()
    module = project / "workers" / "src" / "physrealx_workers" / f"{worker}_worker.py"
    module_paths = json.loads(
        subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import importlib.util,json;"
                    f"names=['physrealx.external_workers','physrealx_workers.{worker}_worker'];"
                    "print(json.dumps([importlib.util.find_spec(n).origin for n in names]))"
                ),
            ],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
    )
    runtime_files = [Path(value).resolve(strict=True) for value in module_paths]
    runtime_files.append(module.resolve(strict=True))
    record = {
        "schema": "physrealx.worker-environment/1",
        "python": subprocess.run(
            [str(python), "--version"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ).stdout.strip(),
        "packages": sorted(freeze),
        "worker_source": {
            "path": module.name,
            "sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        },
        "runtime_files": [
            {
                "path": str(path),
                "sha256": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
            }
            for path in runtime_files
        ],
    }
    if worker == "blenderproc" and blender_runtime is not None:
        embedded = (
            blender_runtime / "custom-python-packages" / "lib" / "python3.11" / "site-packages"
        )
        metadata = sorted(embedded.glob("*.dist-info/METADATA"))
        embedded_runtime_files = [
            embedded / "physrealx" / "external_workers.py",
            embedded / "physrealx_workers" / "common.py",
        ]
        if any(not path.is_file() for path in embedded_runtime_files):
            raise RuntimeError("Blender embedded worker protocol was not installed")
        record["runtime_files"].extend(
            {
                "path": str(path),
                "sha256": f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}",
            }
            for path in embedded_runtime_files
        )
        record["embedded_runtime"] = {
            "blender": str(blender_runtime / "blender"),
            "packages": [
                {
                    "name": item.parent.name,
                    "sha256": hashlib.sha256(item.read_bytes()).hexdigest(),
                }
                for item in metadata
            ],
        }
    return (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _install(
    worker: str,
    project: Path,
    root: Path,
    workspace: Path,
    blender_runtime: Path | None,
    register_only: bool,
) -> None:
    definition = WORKERS[worker]
    environment = root / worker
    if register_only and not (environment / "bin" / "python").is_file():
        raise RuntimeError(f"worker environment is not installed: {environment}")
    if not register_only and not (environment / "bin" / "python").exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin" / "python"
    requirements = project / "workers" / "requirements" / f"{worker}.txt"
    if not register_only:
        _run(
            [str(python), "-m", "pip", "install", "hatchling==1.27.0", "editables==0.5"],
            project,
        )
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--no-build-isolation",
                "-r",
                str(requirements),
                str(project),
                str(project / "workers"),
            ],
            project,
        )
    _run([str(python), "-m", "pip", "check"], project)
    if worker == "blenderproc":
        if blender_runtime is None or not (blender_runtime / "blender").is_file():
            raise RuntimeError("--blender-runtime must point to an installed Blender directory")
        if not register_only:
            _install_blender_runtime(environment, blender_runtime, project)
    deployments = workspace / "workers.d"
    manifests = deployments / "environments"
    manifests.mkdir(parents=True, exist_ok=True)
    manifest = manifests / f"{worker}.json"
    manifest.write_bytes(_environment_record(environment, project, worker, blender_runtime))
    environment_digest = f"sha256:{hashlib.sha256(manifest.read_bytes()).hexdigest()}"
    descriptor = MethodDescriptor(
        capability=definition.capability,
        name=definition.name,
        version=definition.version,
        api_version=API_VERSION,
        source_uri=definition.source_uri,
        source_revision=definition.source_revision,
        license_spdx=definition.license_spdx,
        maturity=MethodMaturity.STABLE,
        runtime=MethodRuntime(
            mode=MethodExecutionMode.PROCESS,
            accelerator=MethodAccelerator.CPU,
            python="3.11",
            environment_digest=environment_digest,
        ),
        input_contract=definition.input_contract,
        output_contract=definition.output_contract,
    )
    module = project / "workers" / "src" / "physrealx_workers" / f"{worker}_worker.py"
    command = [str(python), str(module)]
    probe_command = [*command, "--probe"]
    if worker == "blenderproc":
        script = project / "workers" / "src" / "physrealx_workers" / "blenderproc_worker.py"
        blenderproc = environment / "bin" / "blenderproc"
        command = [
            str(blenderproc),
            "run",
            "--custom-blender-path",
            str(blender_runtime),
            str(script),
        ]
        probe_command = [*command, "--probe"]
    deployment = {
        "schema_version": "1",
        "worker": {
            "kind": "process",
            "descriptor": descriptor.model_dump(mode="json"),
            "command": command,
            "probe_command": probe_command,
            "environment_manifest": str(manifest),
            "timeout_seconds": 3600.0,
            "max_output_bytes": 8 * 1024 * 1024 * 1024,
            "max_log_bytes": 512 * 1024,
        },
    }
    if worker == "blenderproc":
        deployment["worker"]["inherit_environment"] = [
            "HOME",
            "USER",
            "PATH",
            "CUDA_VISIBLE_DEVICES",
            "NVIDIA_VISIBLE_DEVICES",
        ]
    deployments.mkdir(parents=True, exist_ok=True)
    (deployments / f"{worker}.json").write_text(
        json.dumps(deployment, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--blender-runtime", type=Path)
    parser.add_argument(
        "--register-only",
        action="store_true",
        help="register and attest an existing environment without modifying it",
    )
    parser.add_argument("methods", nargs="*", choices=tuple(WORKERS), default=list(WORKERS))
    arguments = parser.parse_args()
    project = arguments.project.expanduser().resolve()
    root = arguments.root.expanduser().resolve()
    workspace = arguments.workspace.expanduser().resolve()
    blender_runtime = (
        arguments.blender_runtime.expanduser().resolve()
        if arguments.blender_runtime is not None
        else None
    )
    root.mkdir(parents=True, exist_ok=True)
    for worker in arguments.methods:
        _install(
            worker,
            project,
            root,
            workspace,
            blender_runtime,
            arguments.register_only,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
