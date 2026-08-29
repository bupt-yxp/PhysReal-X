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
import shutil
import signal
import subprocess
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Final, Literal, Union

from pydantic import Field, field_validator, model_validator

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, WorkerProbe
from physrealx.domain import (
    DomainModel,
    JSONValue,
    MethodAccelerator,
    MethodDescriptor,
    MethodExecutionMode,
    OperationOutput,
    OperationSpec,
    ResourceType,
    canonical_json_bytes,
)
from physrealx.errors import (
    DomainValidationError,
    MethodExecutionError,
    PluginError,
    WorkerUnavailableError,
)

WORKER_PROTOCOL_VERSION: Final[Literal["1"]] = "1"
_DEFAULT_TIMEOUT_SECONDS = 3_600.0
_DEFAULT_MAX_OUTPUT_BYTES = 32 * 1024 * 1024 * 1024
_DEFAULT_MAX_LOG_BYTES = 256 * 1024
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


class WorkerRequest(DomainModel):
    """Stable request envelope consumed by every isolated worker."""

    protocol_version: Literal["1"] = WORKER_PROTOCOL_VERSION
    operation: OperationSpec
    inputs: tuple[WorkerInputFile, ...]
    checkpoint: WorkerInputFile | None = None
    output_directory: Literal["outputs"] = "outputs"

    @model_validator(mode="after")
    def validate_inputs(self) -> WorkerRequest:
        expected = tuple(item.name for item in self.operation.inputs)
        actual = tuple(item.name for item in self.inputs)
        if actual != expected:
            raise ValueError("worker input files must follow OperationSpec input order")
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

    protocol_version: Literal["1"] = WORKER_PROTOCOL_VERSION
    outputs: tuple[WorkerProducedFile, ...]
    metrics: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_response(self) -> WorkerResponse:
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
        _validate_environment_names(self.inherit_environment)
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "environment_manifest", manifest)
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
    if request_path.stat().st_size > _MAX_RESPONSE_BYTES:
        raise DomainValidationError("worker request exceeds the protocol size limit")
    request = WorkerRequest.model_validate_json(request_path.read_bytes())
    root = request_path.parent
    materialized = request.inputs + ((request.checkpoint,) if request.checkpoint else ())
    for item in materialized:
        raw_source = root / item.path
        if raw_source.is_symlink():
            raise DomainValidationError(f"worker input cannot be a symbolic link: {item.path}")
        source = raw_source.resolve(strict=True)
        if root not in source.parents or not source.is_file():
            raise DomainValidationError(f"worker input is not a safe regular file: {item.path}")
        digest, size = _digest_file(source)
        if size != item.size_bytes or digest != item.digest:
            raise DomainValidationError(f"worker input failed integrity validation: {item.name}")

    output_directory = (root / request.output_directory).resolve()
    output_directory.mkdir(mode=0o700, exist_ok=True)
    response_path = arguments.response.expanduser().resolve()
    if response_path.parent != output_directory or response_path.name != "response.json":
        raise DomainValidationError("worker response must be outputs/response.json")
    response = handler(request, output_directory)
    if not isinstance(response, WorkerResponse):
        raise DomainValidationError("worker handler returned an incompatible response")
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
            manifest = self._launch.environment_manifest
            if manifest.is_symlink() or not manifest.is_file():
                return WorkerProbe(False, f"environment manifest is unavailable: {manifest}")
            digest, _ = _digest_file(manifest)
            if digest != self.descriptor.runtime.environment_digest:
                return WorkerProbe(False, "environment manifest digest changed after installation")
            executable = Path(self._launch.command[0]).expanduser()
            if not executable.is_file():
                return WorkerProbe(False, f"executable is not installed: {executable}")
            if not os.access(executable, os.X_OK):
                return WorkerProbe(False, f"executable is not runnable: {executable}")
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
            request_path.write_bytes(canonical_json_bytes(request))
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
            _verify_materialized_inputs(request, root)
            response = _load_response(response_path, stdout=stdout, stderr=stderr)
            return self._publish_response(spec, response, outputs_directory, artifacts)

    def _materialize_request(
        self,
        spec: OperationSpec,
        artifacts: ArtifactStore,
        root: Path,
    ) -> WorkerRequest:
        inputs: list[WorkerInputFile] = []
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

        published: list[OperationOutput] = []
        total_size = 0
        base = output_directory.resolve()
        for item in response.outputs:
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
            published.append(
                OperationOutput(
                    name=item.name,
                    contract=item.contract,
                    artifact=artifacts.put_file(candidate, media_type=item.media_type),
                )
            )
        return MethodExecutionOutput(outputs=tuple(published), metrics=response.metrics)


def _validate_relative_path(value: str, *, label: str) -> None:
    path = Path(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} must be a normalized relative path without traversal")


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
    "WorkerResponse",
    "run_worker",
]
