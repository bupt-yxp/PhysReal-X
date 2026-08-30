#!/usr/bin/env python3
"""Create isolated worker environments and workspace deployment manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import venv
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from physrealx.contracts import API_VERSION
from physrealx.domain import (
    MethodAccelerator,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodMaturity,
    MethodRuntime,
)
from physrealx.external_workers import WORKER_PROTOCOL_VERSION


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


@dataclass(frozen=True)
class WorkerBinding:
    """One fully resolved, machine-local worker installation binding."""

    definition: WorkerDefinition
    worker_arguments: tuple[str, ...] = ()
    environment_assignments: tuple[tuple[str, str], ...] = ()
    external_runtime_files: tuple[Path, ...] = ()
    accelerator: MethodAccelerator = MethodAccelerator.CPU
    maturity: MethodMaturity = MethodMaturity.STABLE


WORKERS = {
    "coacd": WorkerDefinition(
        "coacd",
        MethodCapability.COLLISION_BUILD,
        "1.0.14",
        "https://github.com/SarahWeiii/CoACD",
        "v1.0.14",
        "MIT",
        "VisualMesh/1",
        "CollisionGeometry/1+CollisionBuildReport/1",
    ),
    "colmap": WorkerDefinition(
        "colmap",
        MethodCapability.CAMERA_RECOVER,
        "1.0.0",
        "https://github.com/colmap/colmap",
        "runtime-attested",
        "BSD-3-Clause",
        "ObservationSetManifest/1",
        "CameraRigCalibration/1+SparseReconstructionBundle/1+CameraRecoveryReport/1",
    ),
    "metric_alignment": WorkerDefinition(
        "metric_alignment",
        MethodCapability.CAMERA_ALIGN_METRIC,
        "1.0.0",
        "https://github.com/bupt-yxp/PhysReal-X",
        "adapter-1.0.0",
        "MIT",
        "CameraRigCalibration/1+MetricAlignmentEvidence/1",
        "CameraRigCalibration/1+MetricAlignmentReport/1",
    ),
    "mujoco_sysid": WorkerDefinition(
        "physrealx_mujoco_sysid_adapter",
        MethodCapability.PHYSICS_CALIBRATE,
        "1.0.0",
        "https://github.com/bupt-yxp/PhysReal-X",
        "adapter-1.0.0",
        "MIT",
        "JointSystemIdentificationEvidence/1",
        "IdentifiedJointPhysics/1+PhysicsIdentificationReport/1",
    ),
    "openusd": WorkerDefinition(
        "openusd",
        MethodCapability.ASSET_PACKAGE,
        "1.0.0",
        "https://github.com/bupt-yxp/PhysReal-X",
        "adapter-1.0.0",
        "MIT",
        "AssetAssemblyBundle/1",
        "UsdAssetBundle/1+AssetPackageManifest/1+UsdRoundTripValidationReport/1",
    ),
    "ditto": WorkerDefinition(
        "ditto",
        MethodCapability.ARTICULATION_RECOVER,
        "1.0.0",
        "https://github.com/UT-Austin-RPL/Ditto",
        "f69579f49bbfaadc665fee511ad960f5243e1c1a",
        "MIT",
        "PairedInteractionPointCloudBundle/1",
        "ArticulationModel/1+ArticulationRecoveryReport/1",
    ),
    "omnidc": WorkerDefinition(
        "omni_dc",
        MethodCapability.DEPTH_ENHANCE,
        "1.0.0",
        "https://github.com/princeton-vl/OMNI-DC",
        "installation-manifest",
        "BSD-3-Clause",
        "CalibratedRGBDBundle/1",
        "EnhancedDepthBundle/1+DepthEnhancementReport/1",
    ),
}

_COLMAP_VERSION = re.compile(r"COLMAP\s+([^\s]+)")
_SEMVER_BASE = re.compile(r"^(\d+)\.(\d+)\.(\d+)(.*)$")
_FULL_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_DITTO_REVISION = "f69579f49bbfaadc665fee511ad960f5243e1c1a"
_DITTO_SOURCE_URI = "https://github.com/UT-Austin-RPL/Ditto"
_DITTO_EXPERIMENTS = {"Ditto_s2m", "Ditto_syn"}
_OMNIDC_INSTALLATION_SCHEMA = "physrealx.omnidc-installation/1"
_OMNIDC_SOURCE_URI = "https://github.com/princeton-vl/OMNI-DC"


def _run(command: list[str], cwd: Path) -> None:
    environment = dict(os.environ)
    environment.setdefault("PIP_DISABLE_PIP_VERSION_CHECK", "1")
    environment.setdefault("PIP_DEFAULT_TIMEOUT", "180")
    subprocess.run(command, cwd=cwd, env=environment, check=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _regular_file(
    raw_path: Path,
    label: str,
    *,
    executable: bool = False,
    reject_symlink: bool = True,
) -> Path:
    candidate = raw_path.expanduser()
    if reject_symlink and candidate.is_symlink():
        raise RuntimeError(f"{label} cannot be a symbolic link: {candidate}")
    try:
        path = candidate.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} is unavailable: {candidate}") from error
    if not path.is_file():
        raise RuntimeError(f"{label} must be a regular file: {path}")
    if executable and not os.access(path, os.X_OK):
        raise RuntimeError(f"{label} is not executable: {path}")
    if any(character in str(path) for character in "\x00\r\n"):
        raise RuntimeError(f"{label} path must be a single-line command value")
    return path


def _real_directory(raw_path: Path, label: str) -> Path:
    candidate = raw_path.expanduser()
    if candidate.is_symlink():
        raise RuntimeError(f"{label} cannot be a symbolic link: {candidate}")
    try:
        path = candidate.resolve(strict=True)
    except OSError as error:
        raise RuntimeError(f"{label} is unavailable: {candidate}") from error
    if not path.is_dir():
        raise RuntimeError(f"{label} must be a directory: {path}")
    if any(character in str(path) for character in "\x00\r\n"):
        raise RuntimeError(f"{label} path must be a single-line command value")
    return path


def _python_version(python: Path) -> str:
    try:
        completed = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import json,sys;"
                    "print(json.dumps([sys.version_info.major,sys.version_info.minor]))"
                ),
            ],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(f"cannot execute Python runtime {python}") from error
    try:
        major, minor = json.loads(completed.stdout)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot determine Python version for {python}") from error
    if not isinstance(major, int) or not isinstance(minor, int):
        raise RuntimeError(f"cannot determine Python version for {python}")
    return f"{major}.{minor}"


def _git_text(repository: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        command = " ".join(arguments)
        raise RuntimeError(f"cannot attest Ditto repository with git {command}") from error
    return completed.stdout.strip()


def _normalized_git_origin(value: str) -> str:
    origin = value.strip()
    if origin.endswith(".git"):
        origin = origin[:-4]
    return origin.rstrip("/")


def _ditto_tracked_files(repository: Path) -> tuple[Path, ...]:
    try:
        raw = subprocess.run(
            ["git", "-C", str(repository), "ls-files", "-z"],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError("cannot enumerate Ditto tracked source files") from error
    files: list[Path] = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        try:
            relative = Path(os.fsdecode(encoded))
        except UnicodeDecodeError as error:
            raise RuntimeError("Ditto repository contains a non-decodable tracked path") from error
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Ditto repository contains an unsafe tracked path: {relative}")
        candidate = repository / relative
        if candidate.is_symlink():
            raise RuntimeError(f"Ditto repository contains a tracked symbolic link: {relative}")
        path = candidate.resolve(strict=True)
        if repository not in path.parents or not path.is_file():
            raise RuntimeError(f"Ditto tracked path is not a regular source file: {relative}")
        files.append(path)
    if not files:
        raise RuntimeError("Ditto repository contains no tracked source files")
    return tuple(sorted(files))


def _ditto_binding(
    python_path: Path | None,
    repository_path: Path | None,
    checkpoint_path: Path | None,
    experiment: str,
) -> WorkerBinding:
    if python_path is None or repository_path is None or checkpoint_path is None:
        raise RuntimeError(
            "Ditto registration requires --ditto-python, --ditto-repository and --ditto-checkpoint"
        )
    python = _regular_file(
        python_path,
        "Ditto Python",
        executable=True,
        reject_symlink=False,
    )
    if _python_version(python) != "3.8":
        raise RuntimeError("official Ditto runtime must use Python 3.8")
    repository = _real_directory(repository_path, "Ditto repository")
    head = _git_text(repository, "rev-parse", "HEAD")
    if head != _DITTO_REVISION:
        raise RuntimeError(
            f"Ditto repository must be checked out at pinned revision {_DITTO_REVISION}"
        )
    origin = _normalized_git_origin(_git_text(repository, "remote", "get-url", "origin"))
    if origin not in {_DITTO_SOURCE_URI, "git@github.com:UT-Austin-RPL/Ditto"}:
        raise RuntimeError("Ditto repository origin is not the official UT-Austin-RPL/Ditto")
    if _git_text(repository, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError("Ditto repository contains modified tracked files")
    if experiment not in _DITTO_EXPERIMENTS:
        raise RuntimeError(f"unsupported Ditto experiment: {experiment!r}")
    checkpoint = _regular_file(checkpoint_path, "Ditto checkpoint")
    if checkpoint.stat().st_size == 0:
        raise RuntimeError("Ditto checkpoint cannot be empty")
    checkpoint_digest = _sha256(checkpoint)
    if _SHA256_DIGEST.fullmatch(checkpoint_digest) is None:
        raise RuntimeError("Ditto checkpoint did not produce a valid SHA-256 digest")
    tracked_files = _ditto_tracked_files(repository)
    return WorkerBinding(
        definition=WORKERS["ditto"],
        environment_assignments=(
            ("PHYSREALX_DITTO_PYTHON", str(python)),
            ("PHYSREALX_DITTO_REPOSITORY", str(repository)),
            ("PHYSREALX_DITTO_CHECKPOINT", str(checkpoint)),
            ("PHYSREALX_DITTO_CHECKPOINT_DIGEST", checkpoint_digest),
            ("PHYSREALX_DITTO_EXPERIMENT", experiment),
        ),
        external_runtime_files=(python, checkpoint, *tracked_files),
        accelerator=MethodAccelerator.CUDA,
        maturity=MethodMaturity.EXPERIMENTAL,
    )


def _json_object_without_duplicate_keys(path: Path, label: str) -> dict[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise RuntimeError(f"{label} contains duplicate field {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                RuntimeError(f"{label} contains non-finite number {constant}")
            ),
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be a JSON object")
    return value


def _omnidc_binding(manifest_path: Path | None) -> WorkerBinding:
    if manifest_path is None:
        raise RuntimeError("OMNI-DC registration requires --omnidc-installation-manifest")
    manifest = _regular_file(manifest_path, "OMNI-DC installation manifest")
    document = _json_object_without_duplicate_keys(manifest, "OMNI-DC installation manifest")
    required = {
        "schema",
        "source_uri",
        "source_revision",
        "source_root",
        "source_tree_digest",
        "runtime_files",
        "checkpoint",
        "self_test",
    }
    if set(document) != required:
        missing = sorted(required - set(document))
        unknown = sorted(set(document) - required)
        raise RuntimeError(
            "OMNI-DC installation manifest fields are invalid; "
            f"missing={missing}, unknown={unknown}"
        )
    if (
        document["schema"] != _OMNIDC_INSTALLATION_SCHEMA
        or document["source_uri"] != _OMNIDC_SOURCE_URI
    ):
        raise RuntimeError("OMNI-DC installation manifest identity is unsupported")
    revision = document["source_revision"]
    if not isinstance(revision, str) or _FULL_GIT_REVISION.fullmatch(revision) is None:
        raise RuntimeError("OMNI-DC source_revision must be a full immutable Git commit")
    definition = replace(WORKERS["omnidc"], source_revision=revision)
    return WorkerBinding(
        definition=definition,
        worker_arguments=("--installation-manifest", str(manifest)),
        external_runtime_files=(manifest,),
        accelerator=MethodAccelerator.CUDA,
        maturity=MethodMaturity.EXPERIMENTAL,
    )


def _environment_record(
    environment: Path,
    project: Path,
    worker: str,
    external_runtime_files: tuple[Path, ...] = (),
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
    installed_modules = [Path(value).resolve(strict=True) for value in module_paths]
    candidates = list(installed_modules)
    candidates.append(module.resolve(strict=True))
    for package_root in {path.parent for path in installed_modules}:
        candidates.extend(sorted(package_root.rglob("*.py")))
    if worker == "ditto":
        candidates.append(
            project / "workers" / "src" / "physrealx_workers" / "ditto_inference_bridge.py"
        )
    candidates.append(python.resolve(strict=True))
    candidates.extend(path.resolve(strict=True) for path in external_runtime_files)
    runtime_files: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        if not path.is_file():
            raise RuntimeError(f"worker runtime dependency is not a regular file: {path}")
        seen.add(path)
        runtime_files.append(path)
    protocol_version = subprocess.run(
        [
            str(python),
            "-c",
            (
                "from physrealx.external_workers import WORKER_PROTOCOL_VERSION;"
                "print(WORKER_PROTOCOL_VERSION)"
            ),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    if protocol_version != WORKER_PROTOCOL_VERSION:
        raise RuntimeError(
            "worker environment uses protocol "
            f"{protocol_version!r}; reinstall it for protocol {WORKER_PROTOCOL_VERSION!r}"
        )
    record = {
        "schema": "physrealx.worker-environment/1",
        "protocol_version": protocol_version,
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
                "sha256": _sha256(path),
            }
            for path in runtime_files
        ],
    }
    return (json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def _colmap_definition(binary: Path) -> WorkerDefinition:
    completed = subprocess.run(
        [str(binary), "-h"],
        check=False,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=30,
    )
    match = _COLMAP_VERSION.search(completed.stdout)
    if completed.returncode != 0 or match is None:
        raise RuntimeError(f"cannot determine COLMAP version: {completed.stdout[-4096:]}")
    raw_version = match.group(1)
    version_match = _SEMVER_BASE.fullmatch(raw_version)
    if version_match is None:
        raise RuntimeError(f"COLMAP returned an unsupported version string: {raw_version!r}")
    major, minor, patch, suffix = version_match.groups()
    normalized_suffix = ""
    if suffix:
        normalized = re.sub(r"[^0-9A-Za-z-]+", ".", suffix).strip(".") or "build"
        normalized_suffix = f"-{normalized}"
    digest = _sha256(binary).split(":", 1)[1]
    return replace(
        WORKERS["colmap"],
        version=f"{major}.{minor}.{patch}{normalized_suffix}",
        source_revision=f"binary-{digest[:16]}",
    )


def _worker_binding(
    worker: str,
    *,
    colmap_binary: Path | None,
    ditto_python: Path | None,
    ditto_repository: Path | None,
    ditto_checkpoint: Path | None,
    ditto_experiment: str,
    omnidc_installation_manifest: Path | None,
) -> WorkerBinding:
    if worker == "colmap":
        if colmap_binary is None:
            raise RuntimeError("--colmap-binary must point to an installed COLMAP executable")
        binary = _regular_file(
            colmap_binary,
            "COLMAP binary",
            executable=True,
            reject_symlink=False,
        )
        return WorkerBinding(
            definition=_colmap_definition(binary),
            worker_arguments=("--colmap-binary", str(binary)),
            external_runtime_files=(binary,),
        )
    if worker == "ditto":
        return _ditto_binding(
            ditto_python,
            ditto_repository,
            ditto_checkpoint,
            ditto_experiment,
        )
    if worker == "omnidc":
        return _omnidc_binding(omnidc_installation_manifest)
    return WorkerBinding(definition=WORKERS[worker])


def _install(
    worker: str,
    project: Path,
    root: Path,
    workspace: Path,
    *,
    colmap_binary: Path | None,
    ditto_python: Path | None,
    ditto_repository: Path | None,
    ditto_checkpoint: Path | None,
    ditto_experiment: str,
    omnidc_installation_manifest: Path | None,
    register_only: bool,
) -> None:
    binding = _worker_binding(
        worker,
        colmap_binary=colmap_binary,
        ditto_python=ditto_python,
        ditto_repository=ditto_repository,
        ditto_checkpoint=ditto_checkpoint,
        ditto_experiment=ditto_experiment,
        omnidc_installation_manifest=omnidc_installation_manifest,
    )
    definition = binding.definition
    environment = root / worker
    if register_only and not (environment / "bin" / "python").is_file():
        raise RuntimeError(f"worker environment is not installed: {environment}")
    if not register_only and not (environment / "bin" / "python").exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    python = environment / "bin" / "python"
    requirements = project / "workers" / "requirements" / f"{worker}.txt"
    if not requirements.is_file():
        raise RuntimeError(f"worker requirements are missing: {requirements}")
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

    module = project / "workers" / "src" / "physrealx_workers" / f"{worker}_worker.py"
    if not module.is_file():
        raise RuntimeError(f"worker adapter module is missing: {module}")
    command_prefix: list[str] = []
    additional_runtime_files: tuple[Path, ...] = ()
    if binding.environment_assignments:
        env_candidate = shutil.which("env")
        if env_candidate is None:
            raise RuntimeError("the POSIX env executable is required for pinned Ditto settings")
        env_executable = _regular_file(
            Path(env_candidate),
            "env executable",
            executable=True,
            reject_symlink=False,
        )
        command_prefix = [
            str(env_executable),
            *(f"{name}={value}" for name, value in binding.environment_assignments),
        ]
        additional_runtime_files = (env_executable,)
    worker_command = [
        *command_prefix,
        str(python),
        str(module),
        *binding.worker_arguments,
    ]
    command = worker_command
    probe_command = [*worker_command, "--probe"]

    deployments = workspace / "workers.d"
    manifests = deployments / "environments"
    manifests.mkdir(parents=True, exist_ok=True)
    manifest = manifests / f"{worker}.json"
    manifest.write_bytes(
        _environment_record(
            environment,
            project,
            worker,
            (*binding.external_runtime_files, *additional_runtime_files),
        )
    )
    environment_digest = f"sha256:{hashlib.sha256(manifest.read_bytes()).hexdigest()}"
    descriptor = MethodDescriptor(
        capability=definition.capability,
        name=definition.name,
        version=definition.version,
        api_version=API_VERSION,
        source_uri=definition.source_uri,
        source_revision=definition.source_revision,
        license_spdx=definition.license_spdx,
        maturity=binding.maturity,
        runtime=MethodRuntime(
            mode=MethodExecutionMode.PROCESS,
            accelerator=binding.accelerator,
            python=_python_version(python),
            environment_digest=environment_digest,
        ),
        input_contract=definition.input_contract,
        output_contract=definition.output_contract,
    )
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
            "inherit_environment": [
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
            ],
        },
    }
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
    parser.add_argument(
        "--colmap-binary",
        type=Path,
        help="absolute path to the native COLMAP executable (defaults to PATH lookup)",
    )
    parser.add_argument(
        "--ditto-python",
        type=Path,
        help="Python executable from the separately installed official Ditto Python 3.8 runtime",
    )
    parser.add_argument(
        "--ditto-repository",
        type=Path,
        help=f"official Ditto checkout pinned at {_DITTO_REVISION}",
    )
    parser.add_argument(
        "--ditto-checkpoint",
        type=Path,
        help="official Ditto checkpoint bound by SHA-256 at registration",
    )
    parser.add_argument(
        "--ditto-experiment",
        choices=tuple(sorted(_DITTO_EXPERIMENTS)),
        default="Ditto_s2m",
    )
    parser.add_argument(
        "--omnidc-installation-manifest",
        type=Path,
        help=(
            "pre-existing physrealx.omnidc-installation/1 manifest; the installer "
            "does not download OMNI-DC or its checkpoint"
        ),
    )
    parser.add_argument(
        "--register-only",
        action="store_true",
        help="register and attest an existing environment without modifying it",
    )
    parser.add_argument("methods", nargs="+", choices=tuple(WORKERS))
    arguments = parser.parse_args()
    project = arguments.project.expanduser().resolve()
    root = arguments.root.expanduser().resolve()
    workspace = arguments.workspace.expanduser().resolve()
    colmap_candidate = arguments.colmap_binary
    if colmap_candidate is None:
        discovered_colmap = shutil.which("colmap")
        colmap_candidate = Path(discovered_colmap) if discovered_colmap else None
    root.mkdir(parents=True, exist_ok=True)
    for worker in arguments.methods:
        _install(
            worker,
            project,
            root,
            workspace,
            colmap_binary=colmap_candidate,
            ditto_python=arguments.ditto_python,
            ditto_repository=arguments.ditto_repository,
            ditto_checkpoint=arguments.ditto_checkpoint,
            ditto_experiment=arguments.ditto_experiment,
            omnidc_installation_manifest=arguments.omnidc_installation_manifest,
            register_only=arguments.register_only,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
