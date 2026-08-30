"""Isolated process and OCI-container workers for heavyweight methods.

The core never imports an algorithm framework.  Instead it materializes immutable
inputs, invokes one worker through a versioned JSON/file protocol, validates the
response, and publishes only verified output files to the artifact store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import signal
import stat
import subprocess
import tempfile
import threading
import time
import unicodedata
import zipfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Final, Literal, Union

from pydantic import Field, field_validator, model_validator
from typing_extensions import Annotated

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, WorkerProbe
from physrealx.domain import (
    ArtifactRef,
    DomainModel,
    JSONValue,
    MethodAccelerator,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    OperationOutput,
    OperationOutputRequirement,
    OperationSpec,
    ResourceType,
    canonical_json_bytes,
    sha256_digest,
)
from physrealx.errors import (
    DomainValidationError,
    MethodExecutionError,
    PluginError,
    WorkerUnavailableError,
)

WORKER_PROTOCOL_VERSION: Final[Literal["2"]] = "2"
_DEFAULT_TIMEOUT_SECONDS = 3_600.0
_DEFAULT_MAX_OUTPUT_BYTES = 32 * 1024 * 1024 * 1024
_DEFAULT_MAX_LOG_BYTES = 256 * 1024
_MAX_REQUEST_BYTES = 64 * 1024 * 1024
_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_INHERITED_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
)
_FORBIDDEN_RUNTIME_ARGUMENTS = (
    "--cap-add",
    "--device",
    "--entrypoint",
    "--env",
    "--mount",
    "--network",
    "--privileged",
    "--security-opt",
    "--volume",
)


@dataclass(frozen=True)
class _OutputContractPolicy:
    kind: Literal["binary", "json", "zip"]
    media_types: tuple[str, ...]
    max_size_bytes: int
    schema_discriminators: tuple[str, ...] = ()
    field_types: tuple[
        tuple[str, Literal["array", "boolean", "number", "object", "string"]], ...
    ] = ()
    allowed_statuses: tuple[str, ...] = ()
    zip_manifest: str | None = None
    zip_member_index: Literal["files", "frames"] | None = None
    max_zip_members: int = 100_000
    max_uncompressed_bytes: int = 64 * 1024 * 1024 * 1024
    max_compression_ratio: float = 200.0


_JSON = ("application/json",)
_MIB = 1024 * 1024
_GIB = 1024 * _MIB
_OUTPUT_CONTRACT_POLICIES: Final[dict[str, _OutputContractPolicy]] = {
    "CameraRigCalibration/1": _OutputContractPolicy(
        "json",
        _JSON,
        64 * _MIB,
        ("physrealx.camera-rig-calibration/1",),
        (("method", "object"), ("components", "array"), ("scale_status", "string")),
    ),
    "CameraRecoveryReport/1": _OutputContractPolicy(
        "json",
        _JSON,
        16 * _MIB,
        ("physrealx.camera-recovery-report/1",),
        (("execution", "object"), ("model", "object"), ("scale_status", "string")),
    ),
    "MetricAlignmentReport/1": _OutputContractPolicy(
        "json",
        _JSON,
        16 * _MIB,
        ("physrealx.metric-alignment-report/1",),
        (
            ("status", "string"),
            ("input", "object"),
            ("transform", "object"),
            ("metrics", "object"),
            ("gates", "object"),
            ("provenance", "object"),
        ),
        ("PASSED",),
    ),
    "SparseReconstructionBundle/1": _OutputContractPolicy(
        "zip",
        ("application/zip",),
        8 * _GIB,
        ("physrealx.sparse-reconstruction-bundle/1",),
        (("components", "array"), ("files", "array")),
        zip_manifest="bundle-manifest.json",
        zip_member_index="files",
    ),
    "IdentifiedJointPhysics/1": _OutputContractPolicy(
        "json",
        _JSON,
        4 * _MIB,
        ("physrealx.identified-joint-physics/1",),
        (("status", "string"), ("adapter", "string"), ("evidence_digest", "string")),
        ("IDENTIFIED", "NOT_IDENTIFIABLE"),
    ),
    "PhysicsIdentificationReport/1": _OutputContractPolicy(
        "json",
        _JSON,
        8 * _MIB,
        ("physrealx.physics-identification-report/1",),
        (("status", "string"), ("adapter", "string"), ("diagnostics", "object")),
        ("IDENTIFIED", "NOT_IDENTIFIABLE"),
    ),
    "UsdAssetBundle/1": _OutputContractPolicy(
        "binary",
        ("model/vnd.usd+binary",),
        8 * _GIB,
    ),
    "AssetPackageManifest/1": _OutputContractPolicy(
        "json",
        _JSON,
        8 * _MIB,
        ("physrealx.asset-package-manifest/1",),
        (
            ("resource_id", "string"),
            ("version", "string"),
            ("usd", "object"),
            ("self_contained", "boolean"),
            ("roundtrip_status", "string"),
        ),
    ),
    "UsdRoundTripValidationReport/1": _OutputContractPolicy(
        "json",
        _JSON,
        16 * _MIB,
        ("physrealx.usd-roundtrip-validation-report/1",),
        (("status", "string"), ("checks", "object"), ("failed_checks", "array")),
        ("PASSED",),
    ),
    "EnhancedDepthBundle/1": _OutputContractPolicy(
        "zip",
        ("application/zip",),
        32 * _GIB,
        ("physrealx.enhanced-depth-bundle/1",),
        (("depth_encoding", "object"), ("frames", "array")),
        zip_manifest="manifest.json",
        zip_member_index="frames",
    ),
    "DepthEnhancementReport/1": _OutputContractPolicy(
        "json",
        _JSON,
        32 * _MIB,
        ("physrealx.depth-enhancement-report/1",),
        (("status", "string"), ("metrics", "object"), ("gates", "object")),
        ("PASSED",),
    ),
    "ArticulationModel/1": _OutputContractPolicy(
        "json",
        _JSON,
        32 * _MIB,
        ("physrealx.articulation-model/1",),
        (("status", "string"), ("links", "array"), ("joints", "array")),
        ("IDENTIFIED",),
    ),
    "ArticulationRecoveryReport/1": _OutputContractPolicy(
        "json",
        _JSON,
        16 * _MIB,
        ("physrealx.articulation-recovery-report/1",),
        (("status", "string"), ("gates", "object"), ("diagnostics", "object")),
        ("PASSED",),
    ),
    "CollisionGeometry/1": _OutputContractPolicy(
        "binary",
        ("model/gltf-binary",),
        8 * _GIB,
    ),
    "CollisionBuildReport/1": _OutputContractPolicy(
        "json",
        _JSON,
        16 * _MIB,
        ("physrealx.collision-build-report/1",),
        (
            ("status", "string"),
            ("method", "object"),
            ("unit", "string"),
            ("link", "string"),
            ("source_geometry", "object"),
            ("collision_geometry", "object"),
            ("metrics", "object"),
            ("gates", "object"),
        ),
        ("PASSED",),
    ),
}


class WorkerInputFile(DomainModel):
    """One immutable file materialized for an external worker."""

    name: str
    kind: Literal["resource", "artifact", "checkpoint"]
    digest: str
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1, max_length=255)
    path: str = Field(min_length=1, max_length=255)
    resource_type: ResourceType | None = None

    @model_validator(mode="after")
    def validate_kind(self) -> WorkerInputFile:
        if (self.kind == "resource") != (self.resource_type is not None):
            raise ValueError("only resource inputs may declare resource_type")
        _validate_relative_path(self.path, label="worker input path")
        return self


class WorkerResourceArtifact(DomainModel):
    """One ``ArtifactRef`` discovered inside a resource input manifest."""

    owner: str = Field(min_length=1, max_length=127)
    json_pointer: str = Field(max_length=4_096)
    digest: str
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1, max_length=255)
    path: str = Field(min_length=1, max_length=255)

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return ArtifactRef(
            digest=value,
            size_bytes=0,
            media_type="application/octet-stream",
        ).digest

    @field_validator("json_pointer")
    @classmethod
    def validate_json_pointer(cls, value: str) -> str:
        _validate_json_pointer(value)
        return value

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        _validate_relative_path(value, label="resource artifact path")
        return value


class WorkerRequest(DomainModel):
    """Stable request envelope consumed by every isolated worker."""

    protocol_version: Literal["2"] = WORKER_PROTOCOL_VERSION
    operation: OperationSpec
    inputs: tuple[WorkerInputFile, ...]
    resource_artifacts: tuple[WorkerResourceArtifact, ...]
    checkpoint: WorkerInputFile | None = None
    output_directory: Literal["outputs"] = "outputs"

    @model_validator(mode="after")
    def validate_inputs(self) -> WorkerRequest:
        expected = tuple(item.name for item in self.operation.inputs)
        actual = tuple(item.name for item in self.inputs)
        if actual != expected:
            raise ValueError("worker input files must follow OperationSpec input order")
        resources = {item.name for item in self.inputs if item.kind == "resource"}
        if any(item.owner not in resources for item in self.resource_artifacts):
            raise ValueError("resource artifact owners must name a declared resource input")
        owner_pointers = [(item.owner, item.json_pointer) for item in self.resource_artifacts]
        if len(owner_pointers) != len(set(owner_pointers)):
            raise ValueError("resource artifact owner and JSON pointer pairs must be unique")
        owner_digests = [(item.owner, item.digest) for item in self.resource_artifacts]
        if len(owner_digests) != len(set(owner_digests)):
            raise ValueError("resource artifact digests must be unique within each owner")
        paths = [item.path for item in self.inputs]
        paths.extend(item.path for item in self.resource_artifacts)
        if self.checkpoint is not None:
            paths.append(self.checkpoint.path)
        if len(paths) != len(set(paths)):
            raise ValueError("all materialized worker paths must be unique")
        return self


class WorkerProbeSelfTest(DomainModel):
    """Bounded, machine-readable result of a worker's installation self-test."""

    status: Literal["passed", "failed"]
    checks: dict[str, JSONValue] = Field(default_factory=dict)


class WorkerProbeCheckpoint(DomainModel):
    """Immutable checkpoint identity used by an installed method, when applicable."""

    digest: str
    source_revision: str = Field(min_length=1, max_length=255)

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return ArtifactRef(
            digest=value,
            size_bytes=0,
            media_type="application/octet-stream",
        ).digest


class WorkerProbeEnvironment(DomainModel):
    """Runtime identity observed by the worker process itself."""

    python: str = Field(min_length=1, max_length=127)
    executable_digest: str
    dependencies: dict[str, str] = Field(default_factory=dict)

    @field_validator("executable_digest")
    @classmethod
    def validate_executable_digest(cls, value: str) -> str:
        return ArtifactRef(
            digest=value,
            size_bytes=0,
            media_type="application/octet-stream",
        ).digest

    @field_validator("dependencies")
    @classmethod
    def validate_dependencies(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not name or not version for name, version in value.items()):
            raise ValueError("probe dependency names and versions must be non-empty")
        return value


class WorkerProbeRecord(DomainModel):
    """Authenticated identity and readiness envelope emitted by protocol-v2 workers."""

    schema_name: Literal["physrealx.worker-probe/1"] = Field(
        default="physrealx.worker-probe/1",
        alias="schema",
        serialization_alias="schema",
    )
    protocol_version: Literal["2"] = WORKER_PROTOCOL_VERSION
    capability: MethodCapability
    name: str = Field(min_length=1, max_length=127)
    version: str = Field(min_length=1, max_length=127)
    source_revision: str = Field(min_length=1, max_length=255)
    ready: bool
    self_test: WorkerProbeSelfTest
    checkpoint: WorkerProbeCheckpoint | None = None
    environment: WorkerProbeEnvironment
    reason: str | None = Field(default=None, min_length=1, max_length=4_096)
    details: dict[str, JSONValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_readiness(self) -> WorkerProbeRecord:
        if self.ready and self.self_test.status != "passed":
            raise ValueError("a ready worker must pass its self-test")
        if self.ready and self.reason is not None:
            raise ValueError("a ready worker cannot include a failure reason")
        if not self.ready and self.self_test.status != "failed":
            raise ValueError("an unavailable worker must report a failed self-test")
        if not self.ready and self.reason is None:
            raise ValueError("an unavailable worker must include a failure reason")
        return self


class WorkerProducedFile(DomainModel):
    """One output file declared by a completed external worker."""

    name: str
    contract: str = Field(min_length=1, max_length=127)
    path: str = Field(min_length=1, max_length=255)
    media_type: str = Field(min_length=1, max_length=255)

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        _validate_relative_path(value, label="worker output path")
        return value


class WorkerResponse(DomainModel):
    """Stable response envelope emitted by every isolated worker."""

    protocol_version: Literal["2"] = WORKER_PROTOCOL_VERSION
    request_digest: str | None = None
    descriptor_digest: str | None = None
    outputs: tuple[WorkerProducedFile, ...]
    metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_response(self) -> WorkerResponse:
        for digest in (self.request_digest, self.descriptor_digest):
            if digest is not None:
                ArtifactRef(
                    digest=digest,
                    size_bytes=0,
                    media_type="application/octet-stream",
                )
        if len({item.name for item in self.outputs}) != len(self.outputs):
            raise ValueError("worker response output names must be unique")
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("worker response metrics must be finite")
        return self


@dataclass(frozen=True)
class ProcessLaunch:
    """Local executable from an independently managed algorithm environment."""

    command: tuple[str, ...]
    environment_manifest: Path
    probe_command: tuple[str, ...] = ()
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES
    max_log_bytes: int = _DEFAULT_MAX_LOG_BYTES
    inherit_environment: tuple[str, ...] = _INHERITED_ENVIRONMENT

    def __post_init__(self) -> None:
        _validate_command(self.command)
        executable = Path(self.command[0]).expanduser()
        if not executable.is_absolute():
            raise PluginError("process worker executable must use an absolute path")
        manifest = self.environment_manifest.expanduser()
        if not manifest.is_absolute():
            raise PluginError("process worker environment_manifest must use an absolute path")
        _validate_limits(self.timeout_seconds, self.max_output_bytes, self.max_log_bytes)
        _validate_command(self.probe_command, allow_empty=True)
        if self.probe_command and not Path(self.probe_command[0]).expanduser().is_absolute():
            raise PluginError("process worker probe executable must use an absolute path")
        _validate_environment_names(self.inherit_environment)
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "environment_manifest", manifest)
        object.__setattr__(self, "probe_command", tuple(self.probe_command))
        object.__setattr__(self, "inherit_environment", tuple(self.inherit_environment))


@dataclass(frozen=True)
class ContainerLaunch:
    """OCI image from an independently installed, digest-pinned worker package."""

    image: str
    runtime: str = "docker"
    command: tuple[str, ...] = ()
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES
    max_log_bytes: int = _DEFAULT_MAX_LOG_BYTES
    runtime_arguments: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if "@sha256:" not in self.image or self.image.endswith("@sha256:"):
            raise PluginError("container worker image must be pinned with @sha256:<digest>")
        if not self.runtime or any(character in self.runtime for character in "\x00\r\n"):
            raise PluginError("container runtime must be a non-empty executable name")
        _validate_command(self.command, allow_empty=True)
        _validate_command(self.runtime_arguments, allow_empty=True)
        if any(
            argument == forbidden or argument.startswith(f"{forbidden}=")
            for argument in self.runtime_arguments
            for forbidden in _FORBIDDEN_RUNTIME_ARGUMENTS
        ):
            raise PluginError("container runtime_arguments cannot weaken worker isolation")
        _validate_limits(self.timeout_seconds, self.max_output_bytes, self.max_log_bytes)
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "runtime_arguments", tuple(self.runtime_arguments))


ExternalLaunch = Union[ProcessLaunch, ContainerLaunch]
WorkerHandler = Callable[[WorkerRequest, Path], WorkerResponse]


class ProcessWorkerDeployment(DomainModel):
    """Machine-local binding from an immutable descriptor to one process environment."""

    kind: Literal["process"]
    descriptor: MethodDescriptor
    command: tuple[str, ...]
    environment_manifest: Path
    probe_command: tuple[str, ...] = ()
    inherit_environment: tuple[str, ...] = _INHERITED_ENVIRONMENT
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES
    max_log_bytes: int = _DEFAULT_MAX_LOG_BYTES


class ContainerWorkerDeployment(DomainModel):
    """Machine-local binding from an immutable descriptor to one pinned OCI image."""

    kind: Literal["container"]
    descriptor: MethodDescriptor
    image: str
    runtime: str = "docker"
    command: tuple[str, ...] = ()
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS
    max_output_bytes: int = _DEFAULT_MAX_OUTPUT_BYTES
    max_log_bytes: int = _DEFAULT_MAX_LOG_BYTES
    runtime_arguments: tuple[str, ...] = ()


WorkerDeployment = Annotated[
    Union[ProcessWorkerDeployment, ContainerWorkerDeployment],
    Field(discriminator="kind"),
]


class WorkerDeploymentFile(DomainModel):
    """Versioned deployment file loaded from a workspace ``workers.d`` directory."""

    schema_version: Literal["1"] = "1"
    worker: WorkerDeployment


def load_worker_deployment(path: Path) -> ExternalMethodWorker:
    """Load one deployment without importing its algorithm environment."""

    candidate = path.expanduser()
    if candidate.is_symlink() or not candidate.is_file():
        raise PluginError(f"worker deployment must be a regular file: {candidate}")
    try:
        deployment = WorkerDeploymentFile.model_validate_json(candidate.read_bytes()).worker
    except (OSError, ValueError) as error:
        raise PluginError(f"invalid worker deployment: {candidate}") from error
    launch: ExternalLaunch
    if isinstance(deployment, ProcessWorkerDeployment):
        launch = ProcessLaunch(
            command=deployment.command,
            environment_manifest=deployment.environment_manifest,
            probe_command=deployment.probe_command,
            inherit_environment=deployment.inherit_environment,
            timeout_seconds=deployment.timeout_seconds,
            max_output_bytes=deployment.max_output_bytes,
            max_log_bytes=deployment.max_log_bytes,
        )
    else:
        launch = ContainerLaunch(
            image=deployment.image,
            runtime=deployment.runtime,
            command=deployment.command,
            timeout_seconds=deployment.timeout_seconds,
            max_output_bytes=deployment.max_output_bytes,
            max_log_bytes=deployment.max_log_bytes,
            runtime_arguments=deployment.runtime_arguments,
        )
    return ExternalMethodWorker(deployment.descriptor, launch)


def run_worker(handler: WorkerHandler, argv: Sequence[str] | None = None) -> int:
    """Run a worker-package handler against the stable file protocol.

    External packages use this as their console-script entry point.  The helper
    validates all materialized inputs before algorithm code runs and atomically
    writes the response manifest after the handler returns.
    """

    parser = argparse.ArgumentParser(prog="physrealx-worker")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    arguments = parser.parse_args(argv)
    raw_request_path = arguments.request.expanduser()
    if raw_request_path.is_symlink():
        raise DomainValidationError("worker request cannot be a symbolic link")
    request_path = raw_request_path.resolve(strict=True)
    if not request_path.is_file():
        raise DomainValidationError("worker request must be a regular file")
    if request_path.stat().st_size > _MAX_REQUEST_BYTES:
        raise DomainValidationError("worker request exceeds the protocol size limit")
    request = WorkerRequest.model_validate_json(request_path.read_bytes())
    request_digest, _ = _digest_file(request_path)
    descriptor_digest = _descriptor_digest(request.operation.method.descriptor)
    root = request_path.parent
    _validate_worker_materialization(request, root)

    raw_output_directory = root / request.output_directory
    if raw_output_directory.is_symlink():
        raise DomainValidationError("worker output directory cannot be a symbolic link")
    output_directory = raw_output_directory.resolve()
    if root not in output_directory.parents:
        raise DomainValidationError("worker output directory escapes its sandbox")
    output_directory.mkdir(mode=0o700, exist_ok=True)
    response_path = arguments.response.expanduser().resolve()
    if response_path.parent != output_directory or response_path.name != "response.json":
        raise DomainValidationError("worker response must be outputs/response.json")
    response = handler(request, output_directory)
    if not isinstance(response, WorkerResponse):
        raise DomainValidationError("worker handler returned an incompatible response")
    response = response.model_copy(
        update={
            "request_digest": request_digest,
            "descriptor_digest": descriptor_digest,
        }
    )
    expected = tuple((item.name, item.contract) for item in request.operation.outputs)
    actual = tuple((item.name, item.contract) for item in response.outputs)
    if actual != expected:
        raise DomainValidationError("worker handler returned unexpected output contracts")
    for output_item in response.outputs:
        if output_item.path == "response.json":
            raise DomainValidationError("response.json is reserved for the worker protocol")
        raw_output = output_directory / output_item.path
        if raw_output.is_symlink():
            raise DomainValidationError(
                f"worker output cannot be a symbolic link: {output_item.path}"
            )
        output = raw_output.resolve(strict=True)
        if output_directory not in output.parents or not output.is_file():
            raise DomainValidationError(
                f"worker output is not a safe regular file: {output_item.path}"
            )

    descriptor, temporary_name = tempfile.mkstemp(
        dir=output_directory,
        prefix=".response.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(response))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, response_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return 0


class ExternalMethodWorker:
    """Method worker that delegates execution without importing algorithm code."""

    def __init__(self, descriptor: MethodDescriptor, launch: ExternalLaunch) -> None:
        expected_mode = (
            MethodExecutionMode.PROCESS
            if isinstance(launch, ProcessLaunch)
            else MethodExecutionMode.CONTAINER
        )
        if descriptor.runtime.mode is not expected_mode:
            raise PluginError(
                f"method runtime mode {descriptor.runtime.mode.value!r} does not match "
                f"{expected_mode.value!r} launch configuration"
            )
        if isinstance(launch, ContainerLaunch):
            image_digest = descriptor.runtime.image_digest
            if image_digest is None or launch.image.rsplit("@", 1)[1] != image_digest:
                raise PluginError(
                    "container image reference does not match descriptor image_digest"
                )
        else:
            environment_digest = descriptor.runtime.environment_digest
            if environment_digest is None:
                raise PluginError("process descriptor is missing environment_digest")
            manifest = launch.environment_manifest
            if manifest.is_symlink() or not manifest.is_file():
                raise PluginError(f"process environment manifest is unavailable: {manifest}")
            actual_digest, _ = _digest_file(manifest)
            if actual_digest != environment_digest:
                raise PluginError(
                    "process environment manifest does not match descriptor environment_digest"
                )
        self.descriptor = descriptor
        self._launch = launch

    def probe(self) -> WorkerProbe:
        """Check only local installation state; never download images or dependencies."""

        if isinstance(self._launch, ProcessLaunch):
            integrity = _probe_process_environment(
                self._launch.environment_manifest,
                self.descriptor.runtime.environment_digest,
                commands=(self._launch.command, self._launch.probe_command),
            )
            if integrity is not None:
                return WorkerProbe(False, integrity)
            executable = Path(self._launch.command[0]).expanduser()
            if not executable.is_file():
                return WorkerProbe(False, f"executable is not installed: {executable}")
            if not os.access(executable, os.X_OK):
                return WorkerProbe(False, f"executable is not runnable: {executable}")
            if self._launch.probe_command:
                environment = {
                    name: os.environ[name]
                    for name in self._launch.inherit_environment
                    if name in os.environ
                }
                try:
                    completed = subprocess.run(
                        self._launch.probe_command,
                        check=False,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        env=environment,
                        timeout=30,
                    )
                except (OSError, subprocess.TimeoutExpired) as error:
                    return WorkerProbe(False, f"worker probe failed: {error}")
                detail = completed.stdout[-4_096:].decode("utf-8", errors="replace").strip()
                if detail:
                    record, protocol_error = _parse_machine_probe(
                        detail,
                        self.descriptor,
                        attested_runtime_digests=_environment_runtime_digests(
                            self._launch.environment_manifest
                        ),
                    )
                    if protocol_error is not None:
                        return WorkerProbe(False, protocol_error)
                    if record is not None:
                        if completed.returncode != 0 and record.ready:
                            return WorkerProbe(
                                False,
                                "worker probe returned success data with a failing exit code",
                            )
                        if not record.ready:
                            return WorkerProbe(False, record.reason or "worker self-test failed")
                        return WorkerProbe(True, detail)
                if completed.returncode != 0:
                    return WorkerProbe(False, detail or "worker probe returned a failure")
                if detail:
                    return WorkerProbe(True, detail)
            return WorkerProbe(True, f"process executable: {executable}")

        runtime = shutil.which(self._launch.runtime)
        if runtime is None:
            return WorkerProbe(False, f"OCI runtime is not installed: {self._launch.runtime}")
        try:
            completed = subprocess.run(
                [runtime, "image", "inspect", self._launch.image],
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return WorkerProbe(False, f"cannot inspect pinned OCI image: {error}")
        if completed.returncode != 0:
            return WorkerProbe(False, f"pinned OCI image is not installed: {self._launch.image}")
        return WorkerProbe(True, f"pinned OCI image: {self._launch.image}")

    def execute(
        self,
        spec: OperationSpec,
        artifacts: ArtifactStore,
    ) -> MethodExecutionOutput:
        probe = self.probe()
        if not probe.ready:
            raise WorkerUnavailableError(
                "worker "
                f"{self.descriptor.capability.value}:{self.descriptor.name!r} is unavailable",
                details={"reason": probe.detail},
            )
        prefix = f"physrealx-{self.descriptor.name[:32]}-"
        with tempfile.TemporaryDirectory(prefix=prefix) as temporary:
            root = Path(temporary).resolve()
            inputs_directory = root / "inputs"
            outputs_directory = root / "outputs"
            inputs_directory.mkdir(mode=0o700)
            outputs_directory.mkdir(mode=0o700)
            request = self._materialize_request(spec, artifacts, root)
            request_path = root / "request.json"
            # Omit newly added defaulted constraints so protocol-v2 workers
            # remain wire-compatible; non-default constraints are explicit and
            # therefore require an updated worker-side core package.
            request_payload = canonical_json_bytes(
                request.model_dump(mode="json", exclude_defaults=True)
            )
            if len(request_payload) > _MAX_REQUEST_BYTES:
                raise DomainValidationError("worker request exceeds the protocol size limit")
            request_path.write_bytes(request_payload)
            request_path.chmod(0o400)
            request_digest, request_size = _digest_file(request_path)
            response_path = outputs_directory / "response.json"
            command, environment = self._command(root, request_path, response_path)
            stdout, stderr = _run_bounded(
                command,
                cwd=root,
                environment=environment,
                timeout_seconds=self._launch.timeout_seconds,
                max_log_bytes=self._launch.max_log_bytes,
                output_directory=outputs_directory,
                max_output_bytes=self._launch.max_output_bytes,
            )
            if request_path.is_symlink() or not request_path.is_file():
                raise MethodExecutionError("worker removed or replaced its request manifest")
            actual_request_digest, actual_request_size = _digest_file(request_path)
            if (actual_request_digest, actual_request_size) != (request_digest, request_size):
                raise MethodExecutionError("worker modified its immutable request manifest")
            _verify_materialized_inputs(request, root)
            response = _load_response(response_path, stdout=stdout, stderr=stderr)
            if response.request_digest != request_digest:
                raise MethodExecutionError(
                    "worker response is not bound to the immutable request manifest"
                )
            expected_descriptor_digest = _descriptor_digest(self.descriptor)
            if response.descriptor_digest != expected_descriptor_digest:
                raise MethodExecutionError(
                    "worker response is not bound to the selected method descriptor"
                )
            return self._publish_response(spec, response, outputs_directory, artifacts)

    def _materialize_request(
        self,
        spec: OperationSpec,
        artifacts: ArtifactStore,
        root: Path,
    ) -> WorkerRequest:
        inputs: list[WorkerInputFile] = []
        resource_artifacts: list[WorkerResourceArtifact] = []
        for index, item in enumerate(spec.inputs):
            reference = item.resource if item.resource is not None else item.artifact
            if reference is None:
                raise DomainValidationError(f"operation input {item.name!r} has no reference")
            digest = reference.digest
            if digest is None:
                raise DomainValidationError(f"operation input {item.name!r} is unresolved")
            relative = Path("inputs") / f"{index:03d}-{item.name}"
            destination = root / relative
            artifacts.materialize(digest, destination)
            destination.chmod(0o400)
            actual_digest, size = _digest_file(destination)
            if actual_digest != digest:
                raise DomainValidationError(
                    f"materialized operation input {item.name!r} failed digest validation"
                )
            if item.resource is not None:
                media_type = f"application/vnd.physrealx.{item.resource.resource_type.value}+json"
                kind: Literal["resource", "artifact", "checkpoint"] = "resource"
                resource_type = item.resource.resource_type
                dependencies = _resource_artifact_refs(destination, owner=item.name)
                for dependency_index, (json_pointer, dependency) in enumerate(dependencies):
                    dependency_relative = _resource_artifact_path(
                        input_index=index,
                        owner=item.name,
                        artifact_index=dependency_index,
                        media_type=dependency.media_type,
                    )
                    dependency_destination = root / dependency_relative
                    dependency_destination.parent.mkdir(mode=0o700, exist_ok=True)
                    artifacts.materialize(dependency, dependency_destination)
                    dependency_destination.chmod(0o400)
                    dependency_digest, dependency_size = _digest_file(dependency_destination)
                    if (
                        dependency_digest != dependency.digest
                        or dependency_size != dependency.size_bytes
                    ):
                        raise DomainValidationError(
                            "materialized resource artifact "
                            f"{item.name!r}{json_pointer} failed integrity validation"
                        )
                    resource_artifacts.append(
                        WorkerResourceArtifact(
                            owner=item.name,
                            json_pointer=json_pointer,
                            digest=dependency.digest,
                            size_bytes=dependency.size_bytes,
                            media_type=dependency.media_type,
                            path=dependency_relative.as_posix(),
                        )
                    )
            else:
                artifact = item.artifact
                if artifact is None:
                    raise DomainValidationError(
                        f"operation input {item.name!r} has no artifact reference"
                    )
                media_type = artifact.media_type
                kind = "artifact"
                resource_type = None
            inputs.append(
                WorkerInputFile(
                    name=item.name,
                    kind=kind,
                    digest=digest,
                    size_bytes=size,
                    media_type=media_type,
                    path=relative.as_posix(),
                    resource_type=resource_type,
                )
            )

        checkpoint_file = None
        checkpoint = spec.method.checkpoint
        if checkpoint is not None:
            relative = Path("inputs") / "checkpoint"
            destination = root / relative
            artifacts.materialize(checkpoint, destination)
            destination.chmod(0o400)
            checkpoint_file = WorkerInputFile(
                name="checkpoint",
                kind="checkpoint",
                digest=checkpoint.digest,
                size_bytes=checkpoint.size_bytes,
                media_type=checkpoint.media_type,
                path=relative.as_posix(),
            )
        return WorkerRequest(
            operation=spec,
            inputs=tuple(inputs),
            resource_artifacts=tuple(resource_artifacts),
            checkpoint=checkpoint_file,
        )

    def _command(
        self,
        root: Path,
        request_path: Path,
        response_path: Path,
    ) -> tuple[tuple[str, ...], Mapping[str, str]]:
        if isinstance(self._launch, ProcessLaunch):
            environment = {
                name: os.environ[name]
                for name in self._launch.inherit_environment
                if name in os.environ
            }
            environment["PYTHONNOUSERSITE"] = "1"
            return (
                *self._launch.command,
                "--request",
                str(request_path),
                "--response",
                str(response_path),
            ), environment

        runtime = shutil.which(self._launch.runtime)
        if runtime is None:
            raise WorkerUnavailableError(f"OCI runtime is not installed: {self._launch.runtime}")
        request_mount = f"type=bind,src={request_path},dst=/work/request.json,readonly"
        inputs_mount = f"type=bind,src={root / 'inputs'},dst=/work/inputs,readonly"
        outputs_mount = f"type=bind,src={root / 'outputs'},dst=/work/outputs"
        accelerator_arguments: tuple[str, ...] = ()
        if self.descriptor.runtime.accelerator is MethodAccelerator.CUDA:
            accelerator_arguments = ("--gpus", "all")
        return (
            runtime,
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=4096",
            "--workdir=/work",
            "--mount",
            request_mount,
            "--mount",
            inputs_mount,
            "--mount",
            outputs_mount,
            *accelerator_arguments,
            *self._launch.runtime_arguments,
            self._launch.image,
            *self._launch.command,
            "--request",
            "/work/request.json",
            "--response",
            "/work/outputs/response.json",
        ), {}

    def _publish_response(
        self,
        spec: OperationSpec,
        response: WorkerResponse,
        output_directory: Path,
        artifacts: ArtifactStore,
    ) -> MethodExecutionOutput:
        actual_contracts = tuple((item.name, item.contract) for item in response.outputs)
        expected_contracts = tuple((item.name, item.contract) for item in spec.outputs)
        if actual_contracts != expected_contracts:
            raise MethodExecutionError(
                "worker output names or contracts do not match the operation specification",
                details={
                    "expected": [f"{name}:{contract}" for name, contract in expected_contracts],
                    "actual": [f"{name}:{contract}" for name, contract in actual_contracts],
                },
            )

        validated: list[tuple[WorkerProducedFile, Path]] = []
        total_size = 0
        base = output_directory.resolve()
        for requirement, item in zip(spec.outputs, response.outputs):
            if item.path == "response.json":
                raise MethodExecutionError("response.json cannot be published as a worker output")
            raw_candidate = base / item.path
            if raw_candidate.is_symlink():
                raise MethodExecutionError(f"worker output cannot be a symbolic link: {item.path}")
            candidate = raw_candidate.resolve(strict=True)
            if base not in candidate.parents or candidate == base:
                raise MethodExecutionError(
                    f"worker output escapes its output directory: {item.path}"
                )
            if not candidate.is_file():
                raise MethodExecutionError(f"worker output is not a regular file: {item.path}")
            size = candidate.stat().st_size
            total_size += size
            if total_size > self._launch.max_output_bytes:
                raise MethodExecutionError(
                    "worker outputs exceed the configured size limit",
                    details={"max_output_bytes": self._launch.max_output_bytes},
                )
            _validate_output_contract(
                requirement,
                item,
                candidate,
                execution_size_limit=self._launch.max_output_bytes,
            )
            validated.append((item, candidate))

        # Validation is intentionally a complete first pass.  No artifact is
        # published when any sibling output violates its contract.
        _validate_output_set_closure(validated, operation=spec)
        published: list[OperationOutput] = []
        for item, candidate in validated:
            published.append(
                OperationOutput(
                    name=item.name,
                    contract=item.contract,
                    artifact=artifacts.put_file(candidate, media_type=item.media_type),
                )
            )
        return MethodExecutionOutput(outputs=tuple(published), metrics=response.metrics)


def _validate_output_contract(
    requirement: OperationOutputRequirement,
    produced: WorkerProducedFile,
    path: Path,
    *,
    execution_size_limit: int,
) -> None:
    """Validate one output payload before any sibling output is published."""

    catalog = _OUTPUT_CONTRACT_POLICIES.get(requirement.contract)
    catalog_kind = catalog.kind if catalog is not None else None
    inferred_kind: Literal["binary", "json", "zip"] = "binary"
    if produced.media_type == "application/json":
        inferred_kind = "json"
    elif produced.media_type in {"application/zip", "application/x-zip-compressed"}:
        inferred_kind = "zip"
    requested_kind = None if requirement.content_kind == "auto" else requirement.content_kind
    kind = requested_kind or catalog_kind or inferred_kind
    if catalog_kind is not None and requested_kind is not None and requested_kind != catalog_kind:
        raise MethodExecutionError(
            f"output {produced.name!r} requirement changes the registered content kind"
        )

    catalog_media = catalog.media_types if catalog is not None else ()
    required_media = requirement.media_types
    if catalog_media and required_media:
        allowed_media = tuple(item for item in catalog_media if item in required_media)
        if not allowed_media:
            raise MethodExecutionError(
                f"output {produced.name!r} requirement has no media type allowed by its contract"
            )
    else:
        allowed_media = required_media or catalog_media
    if allowed_media and produced.media_type not in allowed_media:
        raise MethodExecutionError(
            f"worker output {produced.name!r} declared media type {produced.media_type!r} "
            f"but {requirement.contract!r} requires one of {allowed_media!r}"
        )

    limits = [execution_size_limit]
    if catalog is not None:
        limits.append(catalog.max_size_bytes)
    if requirement.max_size_bytes is not None:
        limits.append(requirement.max_size_bytes)
    size_limit = min(limits)
    size = path.stat().st_size
    if size <= 0:
        raise MethodExecutionError(f"worker output {produced.name!r} is empty")
    if size > size_limit:
        raise MethodExecutionError(
            f"worker output {produced.name!r} exceeds its contract size limit",
            details={"size_bytes": size, "max_size_bytes": size_limit},
        )

    catalog_schemas = catalog.schema_discriminators if catalog is not None else ()
    explicit_schema = requirement.schema_discriminator
    if explicit_schema is not None and catalog_schemas and explicit_schema not in catalog_schemas:
        raise MethodExecutionError(
            f"output {produced.name!r} requirement changes the registered schema discriminator"
        )
    schemas = (explicit_schema,) if explicit_schema is not None else catalog_schemas
    policy = catalog or _OutputContractPolicy(
        kind=kind,
        media_types=allowed_media,
        max_size_bytes=size_limit,
        schema_discriminators=schemas,
        zip_manifest=("manifest.json" if kind == "zip" else None),
    )
    if kind == "json":
        _validate_json_file(path, policy, schemas, label=produced.name)
    elif kind == "zip":
        _validate_zip_file(path, policy, schemas, label=produced.name)


def _validated_json_output(path: Path, *, label: str) -> dict[str, JSONValue]:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise MethodExecutionError(f"cannot read worker JSON output {label!r}") from error
    document = _load_json_bytes(payload, label=f"worker output {label!r}")
    if not isinstance(document, dict):
        raise MethodExecutionError(f"worker output {label!r} must contain a JSON object")
    return document


def _validate_output_set_closure(
    outputs: list[tuple[WorkerProducedFile, Path]],
    *,
    operation: OperationSpec | None = None,
) -> None:
    """Validate relationships that cannot be proven from one output alone."""

    by_contract = {item.contract: (item, path) for item, path in outputs}
    collision_entry = by_contract.get("CollisionGeometry/1")
    collision_report_entry = by_contract.get("CollisionBuildReport/1")
    if collision_entry is not None and collision_report_entry is not None:
        collision_item, collision_path = collision_entry
        report = _validated_json_output(
            collision_report_entry[1],
            label=collision_report_entry[0].name,
        )
        source_geometry = report.get("source_geometry")
        collision_geometry = report.get("collision_geometry")
        try:
            source_reference = ArtifactRef.model_validate(source_geometry)
            collision_reference = ArtifactRef.model_validate(collision_geometry)
        except ValueError as error:
            raise MethodExecutionError(
                "collision build report contains an invalid artifact reference"
            ) from error
        actual_digest, actual_size = _digest_file(collision_path)
        gates = report.get("gates")
        metrics = report.get("metrics")
        link = report.get("link")
        method = report.get("method")
        method_identity = (
            tuple(method.get(field) for field in ("name", "version", "source_revision"))
            if isinstance(method, dict)
            else ()
        )
        operation_source = None
        if operation is not None:
            operation_input = next(
                (item for item in operation.inputs if item.name == "asset"),
                None,
            )
            operation_source = operation_input.artifact if operation_input is not None else None
        if (
            report.get("status") != "PASSED"
            or report.get("unit") != "m"
            or not isinstance(link, str)
            or not link.strip()
            or source_reference.size_bytes <= 0
            or not source_reference.media_type.startswith("model/")
            or (operation is not None and source_reference != operation_source)
            or collision_reference.digest != actual_digest
            or collision_reference.size_bytes != actual_size
            or collision_reference.media_type != collision_item.media_type
            or not isinstance(method, dict)
            or len(method_identity) != 3
            or any(not isinstance(value, str) or not value.strip() for value in method_identity)
            or not isinstance(metrics, dict)
            or not metrics
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in metrics.values()
            )
            or not isinstance(gates, dict)
            or not gates
            or any(value is not True for value in gates.values())
        ):
            raise MethodExecutionError(
                "collision build report does not close over one metric source and produced GLB"
            )

    camera_entry = by_contract.get("CameraRigCalibration/1")
    alignment_entry = by_contract.get("MetricAlignmentReport/1")
    if camera_entry is not None and alignment_entry is not None:
        camera = _validated_json_output(camera_entry[1], label=camera_entry[0].name)
        report_path = alignment_entry[1]
        report = _validated_json_output(report_path, label=alignment_entry[0].name)
        source_camera = camera.get("source_camera_rig")
        source_evidence = camera.get("metric_alignment_evidence")
        alignment = camera.get("metric_alignment")
        report_input = report.get("input")
        gates = report.get("gates")
        if not isinstance(source_camera, dict) or not isinstance(source_evidence, dict):
            raise MethodExecutionError(
                "metric camera calibration must identify its camera and evidence inputs"
            )
        try:
            source_camera_ref = ArtifactRef.model_validate(source_camera)
            source_evidence_ref = ArtifactRef.model_validate(source_evidence)
        except ValueError as error:
            raise MethodExecutionError(
                "metric camera calibration contains an invalid source artifact reference"
            ) from error
        if (
            camera.get("scale_status") != "resolved"
            or not isinstance(alignment, dict)
            or not isinstance(report_input, dict)
            or report.get("status") != "PASSED"
            or not isinstance(gates, dict)
            or not gates
            or any(value is not True for value in gates.values())
        ):
            raise MethodExecutionError(
                "metric alignment outputs do not prove a resolved, gate-passed calibration"
            )
        report_digest, _ = _digest_file(report_path)
        transform = report.get("transform")
        if not isinstance(transform, dict):
            raise MethodExecutionError("metric alignment report transform must be an object")
        _validate_metric_sim3(transform)
        convention = camera.get("coordinate_convention")
        components = camera.get("components")
        if (
            report_input.get("camera_rig_digest") != source_camera_ref.digest
            or report_input.get("evidence_digest") != source_evidence_ref.digest
            or alignment.get("report_digest") != report_digest
            or alignment.get("transform") != transform
            or not isinstance(convention, dict)
            or convention.get("translation_unit") != "m"
            or convention.get("frame") != transform.get("target_frame")
            or not isinstance(components, list)
            or not components
            or any(
                not isinstance(component, dict)
                or component.get("coordinate_frame") != transform.get("target_frame")
                for component in components
            )
        ):
            raise MethodExecutionError(
                "metric alignment outputs are not closed over the same inputs and transform"
            )

    usd_entry = by_contract.get("UsdAssetBundle/1")
    manifest_entry = by_contract.get("AssetPackageManifest/1")
    report_entry = by_contract.get("UsdRoundTripValidationReport/1")
    if usd_entry is not None and manifest_entry is not None:
        usd_item, usd_path = usd_entry
        manifest = _validated_json_output(manifest_entry[1], label=manifest_entry[0].name)
        usd_reference = manifest.get("usd")
        if not isinstance(usd_reference, dict):
            raise MethodExecutionError("asset package manifest usd must be an artifact object")
        try:
            declared = ArtifactRef.model_validate(usd_reference)
        except ValueError as error:
            raise MethodExecutionError(
                "asset package manifest contains an invalid USD artifact reference"
            ) from error
        actual_digest, actual_size = _digest_file(usd_path)
        if (
            declared.digest != actual_digest
            or declared.size_bytes != actual_size
            or declared.media_type != usd_item.media_type
        ):
            raise MethodExecutionError(
                "asset package manifest does not identify the produced USD bytes"
            )
        if (
            manifest.get("self_contained") is not True
            or manifest.get("external_dependencies") != []
            or manifest.get("roundtrip_status") != "PASSED"
        ):
            raise MethodExecutionError(
                "asset package manifest must prove a self-contained, round-trip-passed asset"
            )
        if report_entry is not None:
            report = _validated_json_output(report_entry[1], label=report_entry[0].name)
            checks = report.get("checks")
            if (
                report.get("usd_digest") != actual_digest
                or report.get("status") != "PASSED"
                or report.get("failed_checks") != []
                or not isinstance(checks, dict)
                or not checks
                or any(value is not True for value in checks.values())
            ):
                raise MethodExecutionError(
                    "USD round-trip report does not prove every check for the produced USD"
                )

    physics_entry = by_contract.get("IdentifiedJointPhysics/1")
    identification_entry = by_contract.get("PhysicsIdentificationReport/1")
    if physics_entry is not None and identification_entry is not None:
        physics = _validated_json_output(physics_entry[1], label=physics_entry[0].name)
        report = _validated_json_output(identification_entry[1], label=identification_entry[0].name)
        if physics.get("status") != report.get("status") or physics.get("adapter") != report.get(
            "adapter"
        ):
            raise MethodExecutionError(
                "identified joint physics and identification report disagree on status or adapter"
            )


def _validate_metric_sim3(transform: dict[str, JSONValue]) -> None:
    scale = transform.get("scale_m_per_source_unit")
    translation = transform.get("translation_target_m")
    rotation = transform.get("rotation_target_from_source")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or scale <= 0.0
        or not isinstance(translation, list)
        or len(translation) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in translation
        )
        or not isinstance(rotation, list)
        or len(rotation) != 3
        or any(
            not isinstance(row, list)
            or len(row) != 3
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in row)
            for row in rotation
        )
        or not isinstance(transform.get("source_frame"), str)
        or not isinstance(transform.get("target_frame"), str)
        or transform.get("source_frame") == transform.get("target_frame")
    ):
        raise MethodExecutionError("metric alignment transform is not a finite metric Sim(3)")
    numeric_rotation: list[list[float]] = []
    for row in rotation:
        assert isinstance(row, list)
        numeric_row: list[float] = []
        for value in row:
            assert isinstance(value, (int, float)) and not isinstance(value, bool)
            numeric_row.append(float(value))
        numeric_rotation.append(numeric_row)
    for left in range(3):
        for right in range(3):
            product = sum(
                numeric_rotation[left][index] * numeric_rotation[right][index] for index in range(3)
            )
            expected = 1.0 if left == right else 0.0
            if not math.isclose(product, expected, rel_tol=0.0, abs_tol=1e-6):
                raise MethodExecutionError("metric alignment rotation is not orthonormal")
    determinant = (
        numeric_rotation[0][0]
        * (
            numeric_rotation[1][1] * numeric_rotation[2][2]
            - numeric_rotation[1][2] * numeric_rotation[2][1]
        )
        - numeric_rotation[0][1]
        * (
            numeric_rotation[1][0] * numeric_rotation[2][2]
            - numeric_rotation[1][2] * numeric_rotation[2][0]
        )
        + numeric_rotation[0][2]
        * (
            numeric_rotation[1][0] * numeric_rotation[2][1]
            - numeric_rotation[1][1] * numeric_rotation[2][0]
        )
    )
    if not math.isclose(determinant, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise MethodExecutionError("metric alignment rotation must be proper")


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _load_json_bytes(payload: bytes, *, label: str) -> JSONValue:
    try:
        document: JSONValue = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise MethodExecutionError(f"{label} is not unambiguous finite UTF-8 JSON") from error
    pending: list[JSONValue] = [document]
    while pending:
        value = pending.pop()
        if isinstance(value, float) and not math.isfinite(value):
            raise MethodExecutionError(f"{label} contains a non-finite number")
        if isinstance(value, dict):
            pending.extend(value.values())
        elif isinstance(value, list):
            pending.extend(value)
    return document


def _json_kind(value: JSONValue) -> str:
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    return "null"


def _validate_json_document(
    document: JSONValue,
    policy: _OutputContractPolicy,
    schemas: tuple[str, ...],
    *,
    label: str,
) -> dict[str, JSONValue]:
    if not isinstance(document, dict):
        raise MethodExecutionError(f"{label} must contain a JSON object")
    if schemas:
        discriminator = document.get("schema")
        if discriminator not in schemas:
            raise MethodExecutionError(
                f"{label} schema discriminator must be one of {schemas!r}, not {discriminator!r}"
            )
    for field_name, expected_kind in policy.field_types:
        if field_name not in document:
            raise MethodExecutionError(f"{label} is missing required field {field_name!r}")
        actual_kind = _json_kind(document[field_name])
        if actual_kind != expected_kind:
            raise MethodExecutionError(
                f"{label}.{field_name} must be {expected_kind}, not {actual_kind}"
            )
    if policy.allowed_statuses:
        status_value = document.get("status")
        if status_value not in policy.allowed_statuses:
            raise MethodExecutionError(f"{label}.status must be one of {policy.allowed_statuses!r}")
    return document


def _validate_json_file(
    path: Path,
    policy: _OutputContractPolicy,
    schemas: tuple[str, ...],
    *,
    label: str,
) -> None:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise MethodExecutionError(f"cannot read worker JSON output {label!r}") from error
    document = _load_json_bytes(payload, label=f"worker output {label!r}")
    _validate_json_document(document, policy, schemas, label=f"worker output {label!r}")


def _safe_zip_member_name(value: str) -> str:
    if not value or "\\" in value or "\x00" in value:
        raise MethodExecutionError("ZIP member path is empty or non-portable")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise MethodExecutionError(f"ZIP member path is unsafe: {value!r}")
    return path.as_posix()


def _zip_manifest_records(
    document: dict[str, JSONValue],
    index_kind: Literal["files", "frames"],
) -> tuple[tuple[str, str, int], ...]:
    raw_records: object
    if index_kind == "files":
        raw_records = document.get("files")
        if not isinstance(raw_records, list):
            raise MethodExecutionError("ZIP bundle manifest files must be an array")
        candidates = raw_records
    else:
        frames = document.get("frames")
        if not isinstance(frames, list):
            raise MethodExecutionError("ZIP bundle manifest frames must be an array")
        candidates = []
        for frame in frames:
            if not isinstance(frame, dict) or not isinstance(frame.get("depth"), dict):
                raise MethodExecutionError("ZIP bundle frame must contain a depth object")
            candidates.append(frame["depth"])

    records: list[tuple[str, str, int]] = []
    for raw in candidates:
        if not isinstance(raw, dict):
            raise MethodExecutionError("ZIP bundle member index entries must be objects")
        member_path = raw.get("path")
        digest = raw.get("digest", raw.get("sha256"))
        size = raw.get("size_bytes")
        if not isinstance(member_path, str):
            raise MethodExecutionError("ZIP bundle member path must be a string")
        member_path = _safe_zip_member_name(member_path)
        if not isinstance(digest, str):
            raise MethodExecutionError("ZIP bundle member digest must be a string")
        try:
            ArtifactRef(digest=digest, size_bytes=0, media_type="application/octet-stream")
        except ValueError as error:
            raise MethodExecutionError("ZIP bundle member digest is invalid") from error
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise MethodExecutionError("ZIP bundle member size_bytes must be non-negative")
        records.append((member_path, digest, size))
    if not records:
        raise MethodExecutionError("ZIP bundle manifest must index at least one payload")
    names = [item[0] for item in records]
    if len(names) != len(set(names)):
        raise MethodExecutionError("ZIP bundle manifest contains duplicate member paths")
    return tuple(records)


def _validate_zip_file(
    path: Path,
    policy: _OutputContractPolicy,
    schemas: tuple[str, ...],
    *,
    label: str,
) -> None:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.infolist()
            if not members or len(members) > policy.max_zip_members:
                raise MethodExecutionError(
                    f"worker output {label!r} has an invalid ZIP member count"
                )
            normalized_names: list[str] = []
            portable_names: set[str] = set()
            total_uncompressed = 0
            total_compressed = 0
            by_name: dict[str, zipfile.ZipInfo] = {}
            for member in members:
                name = _safe_zip_member_name(member.filename)
                portable = unicodedata.normalize("NFC", name).casefold()
                if name in by_name or portable in portable_names:
                    raise MethodExecutionError("ZIP output contains duplicate member paths")
                if member.is_dir():
                    raise MethodExecutionError("ZIP output may not contain directory entries")
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise MethodExecutionError("ZIP output may not contain symbolic links")
                if member.flag_bits & 0x1:
                    raise MethodExecutionError("ZIP output may not contain encrypted members")
                if member.file_size > 0 and member.compress_size == 0:
                    raise MethodExecutionError("ZIP output has an impossible compression size")
                ratio = member.file_size / max(member.compress_size, 1)
                if ratio > policy.max_compression_ratio:
                    raise MethodExecutionError("ZIP output exceeds the compression-ratio limit")
                total_uncompressed += member.file_size
                total_compressed += member.compress_size
                normalized_names.append(name)
                portable_names.add(portable)
                by_name[name] = member
            if total_uncompressed > policy.max_uncompressed_bytes:
                raise MethodExecutionError("ZIP output exceeds the uncompressed-size limit")
            if total_uncompressed / max(total_compressed, 1) > policy.max_compression_ratio:
                raise MethodExecutionError("ZIP output exceeds the aggregate compression ratio")

            manifest_name = policy.zip_manifest
            if manifest_name is None:
                for candidate in ("bundle-manifest.json", "manifest.json"):
                    if candidate in by_name:
                        manifest_name = candidate
                        break
            if manifest_name is None or manifest_name not in by_name:
                raise MethodExecutionError("ZIP output is missing its required bundle manifest")
            manifest_info = by_name[manifest_name]
            if manifest_info.file_size > 16 * _MIB:
                raise MethodExecutionError("ZIP bundle manifest exceeds its size limit")
            document = _load_json_bytes(
                archive.read(manifest_info),
                label=f"worker output {label!r} ZIP manifest",
            )
            manifest = _validate_json_document(
                document,
                policy,
                schemas,
                label=f"worker output {label!r} ZIP manifest",
            )
            index_kind = policy.zip_member_index
            if index_kind is None:
                if isinstance(manifest.get("files"), list):
                    index_kind = "files"
                elif isinstance(manifest.get("frames"), list):
                    index_kind = "frames"
                else:
                    raise MethodExecutionError("ZIP bundle manifest has no payload index")
            records = _zip_manifest_records(manifest, index_kind)
            expected_names = {manifest_name, *(item[0] for item in records)}
            if set(normalized_names) != expected_names:
                raise MethodExecutionError(
                    "ZIP bundle members do not exactly match the manifest payload index"
                )
            for member_name, declared_digest, declared_size in records:
                info = by_name[member_name]
                if info.file_size != declared_size:
                    raise MethodExecutionError(
                        f"ZIP member {member_name!r} size does not match its manifest"
                    )
                digest = hashlib.sha256()
                size = 0
                with archive.open(info, "r") as source:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                actual_digest = f"sha256:{digest.hexdigest()}"
                if size != declared_size or actual_digest != declared_digest:
                    raise MethodExecutionError(
                        f"ZIP member {member_name!r} failed manifest integrity validation"
                    )
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise MethodExecutionError(f"worker output {label!r} is not a valid ZIP bundle") from error


def _validate_relative_path(value: str, *, label: str) -> None:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a normalized relative path without traversal")


def _validate_json_pointer(value: str) -> None:
    if value and not value.startswith("/"):
        raise ValueError("resource artifact json_pointer must be an RFC 6901 JSON pointer")
    index = 0
    while index < len(value):
        if value[index] != "~":
            index += 1
            continue
        if index + 1 >= len(value) or value[index + 1] not in {"0", "1"}:
            raise ValueError("resource artifact json_pointer contains an invalid escape")
        index += 2


def _reject_duplicate_json_keys(pairs: list[tuple[str, JSONValue]]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"resource manifest contains duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _json_pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _resource_artifact_refs(
    manifest: Path,
    *,
    owner: str,
) -> tuple[tuple[str, ArtifactRef], ...]:
    if manifest.stat().st_size > _MAX_REQUEST_BYTES:
        raise DomainValidationError(f"resource input {owner!r} exceeds the manifest size limit")
    try:
        document: JSONValue = json.loads(
            manifest.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise DomainValidationError(
            f"resource input {owner!r} is not an unambiguous UTF-8 JSON manifest"
        ) from error
    if not isinstance(document, dict):
        raise DomainValidationError(f"resource input {owner!r} must contain a JSON object")

    discovered: list[tuple[str, ArtifactRef]] = []
    pending: list[tuple[str, JSONValue]] = [("", document)]
    while pending:
        pointer, value = pending.pop()
        if isinstance(value, dict):
            if set(value) == {"digest", "size_bytes", "media_type"}:
                try:
                    discovered.append((pointer, ArtifactRef.model_validate(value)))
                except ValueError as error:
                    raise DomainValidationError(
                        f"resource input {owner!r} contains an invalid ArtifactRef at "
                        f"{pointer or '/'}"
                    ) from error
                continue
            for key in reversed(sorted(value)):
                pending.append((f"{pointer}/{_json_pointer_token(key)}", value[key]))
        elif isinstance(value, list):
            for index in range(len(value) - 1, -1, -1):
                pending.append((f"{pointer}/{index}", value[index]))
    digests = [reference.digest for _, reference in discovered]
    if len(digests) != len(set(digests)):
        raise DomainValidationError(
            f"resource input {owner!r} contains duplicate ArtifactRef digests"
        )
    return tuple(discovered)


def _resource_artifact_path(
    *,
    input_index: int,
    owner: str,
    artifact_index: int,
    media_type: str,
) -> Path:
    suffixes = {
        "application/json": ".json",
        "application/octet-stream": ".bin",
        "application/x-npy": ".npy",
        "application/x-npz": ".npz",
        "application/zip": ".zip",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/tiff": ".tiff",
        "model/gltf-binary": ".glb",
        "model/obj": ".obj",
        "model/ply": ".ply",
        "model/stl": ".stl",
        "model/vnd.usd": ".usd",
        "text/plain": ".txt",
    }
    suffix = suffixes.get(media_type, ".bin")
    return Path("inputs") / f"{input_index:03d}-{owner}.artifacts" / f"{artifact_index:06d}{suffix}"


def _validate_command(command: Sequence[str], *, allow_empty: bool = False) -> None:
    if not command and not allow_empty:
        raise PluginError("worker command cannot be empty")
    if any(not value or any(character in value for character in "\x00\r\n") for value in command):
        raise PluginError("worker command arguments must be non-empty single-line strings")


def _validate_limits(timeout_seconds: float, max_output_bytes: int, max_log_bytes: int) -> None:
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise PluginError("worker timeout_seconds must be finite and positive")
    if max_output_bytes <= 0 or max_log_bytes <= 0:
        raise PluginError("worker byte limits must be positive")


def _validate_environment_names(names: Sequence[str]) -> None:
    if any(
        not name or "=" in name or any(character in name for character in "\x00\r\n")
        for name in names
    ):
        raise PluginError("worker environment names must be valid single-line names")


def _digest_file(path: Path) -> tuple[str, int]:
    hasher = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
            size += len(chunk)
    return f"sha256:{hasher.hexdigest()}", size


def _descriptor_digest(descriptor: MethodDescriptor) -> str:
    payload = canonical_json_bytes(descriptor.model_dump(mode="json", exclude_defaults=True))
    return sha256_digest(payload)


def _probe_process_environment(
    manifest: Path,
    expected_digest: str | None,
    *,
    commands: tuple[tuple[str, ...], ...] = (),
) -> str | None:
    if manifest.is_symlink() or not manifest.is_file():
        return f"environment manifest is unavailable: {manifest}"
    digest, _ = _digest_file(manifest)
    if digest != expected_digest:
        return "environment manifest digest changed after installation"
    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return "environment manifest is not valid UTF-8 JSON"
    if not isinstance(record, dict) or record.get("schema") != "physrealx.worker-environment/1":
        return "environment manifest uses an unsupported schema"
    if record.get("protocol_version") != WORKER_PROTOCOL_VERSION:
        return (
            "environment manifest targets an incompatible worker protocol; "
            "reinstall or re-register the worker"
        )
    runtime_files = record.get("runtime_files")
    if not isinstance(runtime_files, list) or not runtime_files:
        return "environment manifest has no runtime file integrity records"
    attested_files: set[Path] = set()
    for item in runtime_files:
        if not isinstance(item, dict):
            return "environment manifest contains an invalid runtime file record"
        raw_path = item.get("path")
        expected = item.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected, str):
            return "environment manifest contains an invalid runtime file record"
        path = Path(raw_path).expanduser()
        if not path.is_absolute() or path.is_symlink() or not path.is_file():
            return f"worker runtime file is unavailable: {path}"
        actual, _ = _digest_file(path)
        if actual != expected:
            return f"worker runtime file changed after installation: {path}"
        attested_files.add(path.resolve())
    for command in commands:
        for argument in command:
            candidate_value = argument.split("=", 1)[1] if "=" in argument else argument
            candidate = Path(candidate_value).expanduser()
            if not candidate.is_absolute() or not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved not in attested_files:
                return f"worker command file is not attested by its environment: {resolved}"
    return None


def _parse_machine_probe(
    detail: str,
    descriptor: MethodDescriptor,
    *,
    attested_runtime_digests: set[str],
) -> tuple[WorkerProbeRecord | None, str | None]:
    """Validate authenticated JSON probes while retaining explicit plain-text probes."""

    final_line = detail.rsplit("\n", 1)[-1]
    if not final_line.lstrip().startswith("{"):
        return None, None
    try:
        raw_record = json.loads(
            final_line,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (json.JSONDecodeError, ValueError):
        return None, "worker probe emitted malformed machine-readable output"
    if not isinstance(raw_record, dict):
        return None, "worker probe emitted an invalid machine-readable record"
    protocol_version = raw_record.get("protocol_version")
    if protocol_version != WORKER_PROTOCOL_VERSION:
        return None, (
            f"worker probe targets protocol {protocol_version!r}; "
            f"protocol {WORKER_PROTOCOL_VERSION!r} is required"
        )
    try:
        record = WorkerProbeRecord.model_validate(raw_record)
    except ValueError as error:
        return None, f"worker probe violates WorkerProbe/1: {error}"
    expected_identity = (
        descriptor.capability,
        descriptor.name,
        descriptor.version,
        descriptor.source_revision,
    )
    actual_identity = (
        record.capability,
        record.name,
        record.version,
        record.source_revision,
    )
    if actual_identity != expected_identity:
        return None, (
            "worker probe identity does not match its deployment descriptor: "
            f"expected={tuple(str(value) for value in expected_identity)!r}, "
            f"actual={tuple(str(value) for value in actual_identity)!r}"
        )
    if record.environment.executable_digest not in attested_runtime_digests:
        return None, (
            "worker probe interpreter is not attested by its environment manifest: "
            f"{record.environment.executable_digest}"
        )
    expected_python = descriptor.runtime.python
    if re.fullmatch(r"\d+\.\d+", expected_python) is not None:
        actual_python = record.environment.python.split(".")
        if len(actual_python) < 2 or ".".join(actual_python[:2]) != expected_python:
            return None, (
                "worker probe Python identity does not match its deployment descriptor: "
                f"expected={expected_python!r}, actual={record.environment.python!r}"
            )
    return record, None


def _environment_runtime_digests(manifest: Path) -> set[str]:
    """Return only digests from an environment manifest already integrity-checked above."""

    try:
        record = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return set()
    runtime_files = record.get("runtime_files") if isinstance(record, dict) else None
    if not isinstance(runtime_files, list):
        return set()
    return {
        digest
        for item in runtime_files
        if isinstance(item, dict) and isinstance((digest := item.get("sha256")), str)
    }


def _safe_materialized_file(root: Path, relative_path: str, *, label: str) -> Path:
    raw_path = root / relative_path
    if raw_path.is_symlink():
        raise DomainValidationError(f"{label} cannot be a symbolic link: {relative_path}")
    try:
        path = raw_path.resolve(strict=True)
    except OSError as error:
        raise DomainValidationError(f"{label} is missing: {relative_path}") from error
    if root not in path.parents or not path.is_file():
        raise DomainValidationError(f"{label} is not a safe regular file: {relative_path}")
    return path


def _validate_resource_artifact_index(request: WorkerRequest, root: Path) -> None:
    """Prove the declared resource closure is complete and deterministic."""

    for input_index, resource_input in enumerate(request.inputs):
        if resource_input.kind != "resource":
            continue
        manifest = _safe_materialized_file(
            root,
            resource_input.path,
            label=f"resource input {resource_input.name!r}",
        )
        discovered = _resource_artifact_refs(manifest, owner=resource_input.name)
        indexed = tuple(
            item for item in request.resource_artifacts if item.owner == resource_input.name
        )
        if len(indexed) != len(discovered):
            raise DomainValidationError(
                f"resource artifact index for {resource_input.name!r} is incomplete: "
                f"manifest declares {len(discovered)}, request indexes {len(indexed)}"
            )
        for artifact_index, ((json_pointer, reference), declared) in enumerate(
            zip(discovered, indexed)
        ):
            expected_path = _resource_artifact_path(
                input_index=input_index,
                owner=resource_input.name,
                artifact_index=artifact_index,
                media_type=reference.media_type,
            ).as_posix()
            expected = (
                resource_input.name,
                json_pointer,
                reference.digest,
                reference.size_bytes,
                reference.media_type,
                expected_path,
            )
            actual = (
                declared.owner,
                declared.json_pointer,
                declared.digest,
                declared.size_bytes,
                declared.media_type,
                declared.path,
            )
            if actual != expected:
                raise DomainValidationError(
                    f"resource artifact index mismatch for {resource_input.name!r} at "
                    f"{json_pointer or '/'}"
                )


def _validate_worker_materialization(request: WorkerRequest, root: Path) -> None:
    for item in request.inputs:
        source = _safe_materialized_file(root, item.path, label="worker input")
        digest, size = _digest_file(source)
        if size != item.size_bytes or digest != item.digest:
            raise DomainValidationError(f"worker input failed integrity validation: {item.name}")
    if request.checkpoint is not None:
        checkpoint = _safe_materialized_file(
            root,
            request.checkpoint.path,
            label="worker checkpoint",
        )
        digest, size = _digest_file(checkpoint)
        if size != request.checkpoint.size_bytes or digest != request.checkpoint.digest:
            raise DomainValidationError("worker checkpoint failed integrity validation")
    for dependency_item in request.resource_artifacts:
        dependency = _safe_materialized_file(
            root,
            dependency_item.path,
            label=(f"resource artifact {dependency_item.owner!r}{dependency_item.json_pointer}"),
        )
        digest, size = _digest_file(dependency)
        if size != dependency_item.size_bytes or digest != dependency_item.digest:
            raise DomainValidationError(
                "resource artifact failed integrity validation: "
                f"{dependency_item.owner}{dependency_item.json_pointer}"
            )
    _validate_resource_artifact_index(request, root)


def _verify_materialized_inputs(request: WorkerRequest, root: Path) -> None:
    materialized = request.inputs + ((request.checkpoint,) if request.checkpoint else ())
    for item in materialized:
        raw_path = root / item.path
        if raw_path.is_symlink():
            raise MethodExecutionError(f"worker modified input into a symbolic link: {item.name}")
        try:
            path = raw_path.resolve(strict=True)
        except OSError as error:
            raise MethodExecutionError(f"worker removed an immutable input: {item.name}") from error
        if root not in path.parents or not path.is_file():
            raise MethodExecutionError(f"worker moved an input outside its sandbox: {item.name}")
        digest, size = _digest_file(path)
        if digest != item.digest or size != item.size_bytes:
            raise MethodExecutionError(f"worker modified an immutable input: {item.name}")
    for dependency_item in request.resource_artifacts:
        raw_path = root / dependency_item.path
        label = f"{dependency_item.owner}{dependency_item.json_pointer}"
        if raw_path.is_symlink():
            raise MethodExecutionError(
                f"worker modified resource artifact into a symbolic link: {label}"
            )
        try:
            path = raw_path.resolve(strict=True)
        except OSError as error:
            raise MethodExecutionError(
                f"worker removed an immutable resource artifact: {label}"
            ) from error
        if root not in path.parents or not path.is_file():
            raise MethodExecutionError(
                f"worker moved a resource artifact outside its sandbox: {label}"
            )
        digest, size = _digest_file(path)
        if digest != dependency_item.digest or size != dependency_item.size_bytes:
            raise MethodExecutionError(f"worker modified an immutable resource artifact: {label}")
    try:
        _validate_resource_artifact_index(request, root)
    except DomainValidationError as error:
        raise MethodExecutionError(
            "worker resource artifact closure failed post-execution validation"
        ) from error


def _run_bounded(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    timeout_seconds: float,
    max_log_bytes: int,
    output_directory: Path,
    max_output_bytes: int,
) -> tuple[str, str]:
    """Run without a shell, drain logs concurrently, and retain bounded tails."""

    try:
        process = subprocess.Popen(
            tuple(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        raise WorkerUnavailableError(f"cannot start worker executable: {error}") from error

    stdout_buffer = bytearray()
    stderr_buffer = bytearray()

    def drain(stream: object, target: bytearray) -> None:
        reader = getattr(stream, "read", None)
        if reader is None:
            return
        while True:
            chunk = reader(64 * 1024)
            if not chunk:
                return
            target.extend(chunk)
            if len(target) > max_log_bytes:
                del target[:-max_log_bytes]

    stdout_thread = threading.Thread(
        target=drain, args=(process.stdout, stdout_buffer), daemon=True
    )
    stderr_thread = threading.Thread(
        target=drain, args=(process.stderr, stderr_buffer), daemon=True
    )
    stdout_thread.start()
    stderr_thread.start()
    deadline = time.monotonic() + timeout_seconds
    failure: MethodExecutionError | None = None
    while process.poll() is None:
        if time.monotonic() >= deadline:
            failure = MethodExecutionError(
                "worker execution exceeded its timeout",
                details={"timeout_seconds": timeout_seconds},
            )
        elif _directory_size(output_directory) > max_output_bytes:
            failure = MethodExecutionError(
                "worker outputs exceeded the configured size limit during execution",
                details={"max_output_bytes": max_output_bytes},
            )
        if failure is None:
            time.sleep(0.1)
            continue
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        break
    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    stdout = stdout_buffer.decode("utf-8", errors="replace")
    stderr = stderr_buffer.decode("utf-8", errors="replace")
    if failure is not None:
        raise failure
    if return_code != 0:
        raise MethodExecutionError(
            "worker process exited unsuccessfully",
            details={"exit_code": return_code, "stdout_tail": stdout, "stderr_tail": stderr},
        )
    return stdout, stderr


def _directory_size(directory: Path) -> int:
    total = 0
    for root, _, files in os.walk(directory, followlinks=False):
        for name in files:
            try:
                total += (Path(root) / name).lstat().st_size
            except FileNotFoundError:
                continue
    return total


def _load_response(path: Path, *, stdout: str, stderr: str) -> WorkerResponse:
    if path.is_symlink() or not path.is_file():
        raise MethodExecutionError(
            "worker completed without a response manifest",
            details={"stdout_tail": stdout, "stderr_tail": stderr},
        )
    if path.stat().st_size > _MAX_RESPONSE_BYTES:
        raise MethodExecutionError("worker response manifest exceeds the protocol size limit")
    try:
        value: JSONValue = json.loads(path.read_text(encoding="utf-8"))
        return WorkerResponse.model_validate(value)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise MethodExecutionError("worker response manifest is invalid") from error


__all__ = [
    "WORKER_PROTOCOL_VERSION",
    "ContainerLaunch",
    "ExternalMethodWorker",
    "ProcessLaunch",
    "WorkerHandler",
    "WorkerInputFile",
    "WorkerProducedFile",
    "WorkerRequest",
    "WorkerResourceArtifact",
    "WorkerResponse",
    "run_worker",
]
