from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from physrealx.contracts import ArtifactStore, MethodExecutionOutput
from physrealx.domain import (
    ArtifactRef,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodRuntime,
    OperationInput,
    OperationManifest,
    OperationOutputRequirement,
    OperationSpec,
    ResourceType,
    VersionedRef,
    sha256_digest,
)
from physrealx.errors import (
    DomainValidationError,
    MethodExecutionError,
    PluginError,
    ResourceNotFoundError,
    WorkerUnavailableError,
)
from physrealx.external_workers import (
    ContainerLaunch,
    ExternalMethodWorker,
    ProcessLaunch,
    WorkerDeploymentFile,
    WorkerInputFile,
    WorkerRequest,
    WorkerResourceArtifact,
    run_worker,
)
from physrealx.platform import PhysRealX
from physrealx.plugins import PluginRegistry


def _descriptor(
    name: str,
    *,
    mode: MethodExecutionMode = MethodExecutionMode.PROCESS,
    image_digest: str | None = None,
    environment_digest: str | None = None,
) -> MethodDescriptor:
    return MethodDescriptor(
        capability=MethodCapability.COLLISION_BUILD,
        name=name,
        version="1.0.0",
        api_version="1",
        source_uri="https://example.com/physrealx-test-worker",
        source_revision="v1.0.0",
        license_spdx="MIT",
        runtime=MethodRuntime(
            mode=mode,
            python=">=3.8,<3.13",
            image_digest=image_digest,
            environment_digest=environment_digest,
        ),
        input_contract="AssetSpec|TriangleMesh",
        output_contract="CollisionGeometry",
    )


def _process_worker(
    tmp_path: Path,
    name: str,
    command: tuple[str, ...],
    *,
    timeout_seconds: float = 3_600.0,
) -> tuple[MethodDescriptor, ExternalMethodWorker]:
    manifest = tmp_path / f"{name}.environment.lock"
    runtime_paths = [Path(sys.executable).resolve()] + [
        Path(value).resolve() for value in command[1:] if Path(value).is_file()
    ]
    payload = json.dumps(
        {
            "schema": "physrealx.worker-environment/1",
            "protocol_version": "2",
            "runtime_files": [
                {
                    "path": str(path),
                    "sha256": sha256_digest(path.read_bytes()),
                }
                for path in runtime_paths
            ],
        }
    ).encode()
    manifest.write_bytes(payload)
    descriptor = _descriptor(name, environment_digest=sha256_digest(payload))
    return descriptor, ExternalMethodWorker(
        descriptor,
        ProcessLaunch(
            command=command,
            environment_manifest=manifest,
            timeout_seconds=timeout_seconds,
        ),
    )


def _write_worker(path: Path) -> None:
    path.write_text(
        """\
import json
from pathlib import Path

from physrealx.external_workers import WorkerProducedFile, WorkerResponse, run_worker


def handler(request, output_directory: Path):
    source = output_directory.parent / request.inputs[0].path
    output = output_directory / "collision.json"
    output.write_text(
        json.dumps({"source_size": len(source.read_bytes()), "parts": 4}),
        encoding="utf-8",
    )
    requirement = request.operation.outputs[0]
    return WorkerResponse(
        outputs=(
            WorkerProducedFile(
                name=requirement.name,
                contract=requirement.contract,
                path=output.name,
                media_type="application/json",
            ),
        ),
        metrics={"part_count": 4.0},
    )


raise SystemExit(run_worker(handler))
""",
        encoding="utf-8",
    )


def _write_resource_worker(path: Path) -> None:
    workers_source = Path(__file__).parents[2] / "workers" / "src"
    path.write_text(
        f"""\
import json
import sys
from pathlib import Path

sys.path.insert(0, {str(workers_source)!r})

from physrealx.external_workers import WorkerProducedFile, WorkerResponse, run_worker
from physrealx_workers.common import dependency_path, resource_json


def handler(request, output_directory: Path):
    manifest = resource_json(request, "asset")
    digest = manifest["geometry"]["artifact"]["digest"]
    source = dependency_path(request, "asset", digest)
    dependency = request.resource_artifacts[0]
    output = output_directory / "collision.json"
    output.write_text(
        json.dumps(
            {{
                "resource_id": manifest["resource_id"],
                "source_size": len(source.read_bytes()),
                "owner": dependency.owner,
                "json_pointer": dependency.json_pointer,
                "path": dependency.path,
            }}
        ),
        encoding="utf-8",
    )
    requirement = request.operation.outputs[0]
    return WorkerResponse(
        outputs=(
            WorkerProducedFile(
                name=requirement.name,
                contract=requirement.contract,
                path=output.name,
                media_type="application/json",
            ),
        ),
        metrics={{"dependency_count": float(len(request.resource_artifacts))}},
    )


raise SystemExit(run_worker(handler))
""",
        encoding="utf-8",
    )


def _write_fake_oci_runtime(path: Path) -> None:
    path.write_text(
        f"""\
#!{sys.executable}
import hashlib
import json
import sys
from pathlib import Path

arguments = sys.argv[1:]
if arguments[:2] == ["image", "inspect"]:
    raise SystemExit(0)
if not arguments or arguments[0] != "run":
    raise SystemExit(64)
mounts = {{}}
for index, argument in enumerate(arguments):
    if argument != "--mount":
        continue
    fields = dict(field.split("=", 1) for field in arguments[index + 1].split(",") if "=" in field)
    mounts[fields["dst"]] = Path(fields["src"])
request_bytes = mounts["/work/request.json"].read_bytes()
request = json.loads(request_bytes)
inputs = mounts["/work/inputs"]
outputs = mounts["/work/outputs"]
source = inputs / Path(request["inputs"][0]["path"]).name
output = outputs / "collision.json"
output.write_text(
    json.dumps({{"source_size": len(source.read_bytes()), "parts": 5}}),
    encoding="utf-8",
)
requirement = request["operation"]["outputs"][0]
(outputs / "response.json").write_text(
    json.dumps(
        {{
            "protocol_version": "2",
            "request_digest": "sha256:" + hashlib.sha256(request_bytes).hexdigest(),
            "descriptor_digest": "sha256:" + hashlib.sha256(
                json.dumps(
                    request["operation"]["method"]["descriptor"],
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
            "outputs": [
                {{
                    "name": requirement["name"],
                    "contract": requirement["contract"],
                    "path": output.name,
                    "media_type": "application/json",
                }}
            ],
            "metrics": {{"part_count": 5.0}},
        }}
    ),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )
    path.chmod(0o700)


def _resource_reference(
    platform: PhysRealX,
    document: dict[str, object],
) -> VersionedRef:
    manifest = platform.artifacts.put_json(
        document,
        media_type="application/vnd.physrealx.asset+json",
    )
    reference = VersionedRef(
        resource_type=ResourceType.ASSET,
        resource_id="cabinet.resource",
        version="1.0.0",
        digest=manifest.digest,
    )
    return platform.repository.register_resource(reference)


def _resource_document(reference: ArtifactRef) -> dict[str, object]:
    return {
        "schema_version": 1,
        "resource_id": "cabinet.resource",
        "geometry": {"artifact": reference.model_dump(mode="json")},
    }


def _execute_protocol_fixture(
    platform: PhysRealX,
    source: VersionedRef | ArtifactRef,
    method: str,
) -> OperationManifest:
    """Exercise the generic worker protocol without a domain service contract."""

    selected = platform.methods.bind(MethodCapability.COLLISION_BUILD, method)
    source_digest = source.digest
    if source_digest is None:
        raise AssertionError("protocol fixture requires an immutable input")
    return platform._execute_operation(
        OperationSpec(
            resource_id=(
                "operation.protocol."
                + sha256_digest(
                    {
                        "method": selected.model_dump(mode="json"),
                        "source_digest": source_digest,
                    }
                )[-16:]
            ),
            version="1.0.0",
            method=selected,
            inputs=(
                OperationInput(
                    name="asset",
                    resource=source if isinstance(source, VersionedRef) else None,
                    artifact=source if isinstance(source, ArtifactRef) else None,
                ),
            ),
            outputs=(
                OperationOutputRequirement(
                    name="collision_geometry",
                    contract="CollisionGeometry",
                ),
            ),
        )
    )


def test_process_worker_executes_through_versioned_file_protocol(tmp_path: Path) -> None:
    executable = tmp_path / "worker.py"
    _write_worker(executable)
    descriptor, worker = _process_worker(
        tmp_path,
        "isolated_coacd",
        (sys.executable, str(executable)),
    )
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    platform.plugins.register_worker(worker)
    source = platform.artifacts.put_json({"mesh": "cabinet"})

    installation = platform.methods.inspect(
        MethodCapability.COLLISION_BUILD,
        descriptor.name,
    )
    result = _execute_protocol_fixture(platform, source, descriptor.name)

    assert installation.executable and installation.ready
    assert installation.execution_kind == "method_worker"
    assert result.metrics == {"part_count": 4.0}
    assert result.outputs[0].contract == "CollisionGeometry"
    assert platform.artifacts.get_bytes(result.outputs[0].artifact) == (
        b'{"source_size": 18, "parts": 4}'
    )


def test_resource_input_materializes_complete_artifact_closure(tmp_path: Path) -> None:
    executable = tmp_path / "resource-worker.py"
    _write_resource_worker(executable)
    descriptor, worker = _process_worker(
        tmp_path,
        "resource_coacd",
        (sys.executable, str(executable)),
    )
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    platform.plugins.register_worker(worker)
    geometry = platform.artifacts.put_bytes(b"ply-payload", media_type="model/ply")
    resource = _resource_reference(platform, _resource_document(geometry))

    result = _execute_protocol_fixture(platform, resource, descriptor.name)

    assert result.metrics == {"dependency_count": 1.0}
    output = json.loads(platform.artifacts.get_bytes(result.outputs[0].artifact))
    assert output == {
        "resource_id": "cabinet.resource",
        "source_size": 11,
        "owner": "asset",
        "json_pointer": "/geometry/artifact",
        "path": "inputs/000-asset.artifacts/000000.ply",
    }


def test_resource_input_rejects_missing_artifact_dependency(tmp_path: Path) -> None:
    executable = tmp_path / "resource-worker.py"
    _write_resource_worker(executable)
    descriptor, worker = _process_worker(
        tmp_path,
        "missing_resource_artifact",
        (sys.executable, str(executable)),
    )
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    platform.plugins.register_worker(worker)
    missing = ArtifactRef(
        digest=f"sha256:{'f' * 64}",
        size_bytes=7,
        media_type="model/ply",
    )
    resource = _resource_reference(platform, _resource_document(missing))

    with pytest.raises(ResourceNotFoundError, match="artifact not found"):
        _execute_protocol_fixture(platform, resource, descriptor.name)
    assert platform.operations.list() == []


def test_resource_input_rejects_duplicate_artifact_dependency(tmp_path: Path) -> None:
    executable = tmp_path / "resource-worker.py"
    _write_resource_worker(executable)
    descriptor, worker = _process_worker(
        tmp_path,
        "duplicate_resource_artifact",
        (sys.executable, str(executable)),
    )
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    platform.plugins.register_worker(worker)
    geometry = platform.artifacts.put_bytes(b"ply-payload", media_type="model/ply")
    document = _resource_document(geometry)
    document["duplicate"] = geometry.model_dump(mode="json")
    resource = _resource_reference(platform, document)

    with pytest.raises(DomainValidationError, match="duplicate ArtifactRef digests"):
        _execute_protocol_fixture(platform, resource, descriptor.name)
    assert platform.operations.list() == []


def test_process_worker_cannot_mutate_resource_artifact_dependency(tmp_path: Path) -> None:
    executable = tmp_path / "resource-tamper-worker.py"
    executable.write_text(
        """\
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request", type=Path, required=True)
parser.add_argument("--response", type=Path, required=True)
arguments = parser.parse_args()
request = json.loads(arguments.request.read_text(encoding="utf-8"))
dependency = arguments.request.parent / request["resource_artifacts"][0]["path"]
dependency.chmod(0o600)
dependency.write_bytes(b"tampered")
""",
        encoding="utf-8",
    )
    descriptor, worker = _process_worker(
        tmp_path,
        "tampering_resource_artifact",
        (sys.executable, str(executable)),
    )
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    platform.plugins.register_worker(worker)
    geometry = platform.artifacts.put_bytes(b"ply-payload", media_type="model/ply")
    resource = _resource_reference(platform, _resource_document(geometry))

    with pytest.raises(MethodExecutionError, match="modified an immutable resource artifact"):
        _execute_protocol_fixture(platform, resource, descriptor.name)
    assert platform.operations.list() == []


def test_worker_protocol_rejects_incomplete_resource_artifact_index(tmp_path: Path) -> None:
    payload = b"dependency"
    dependency = ArtifactRef(
        digest=sha256_digest(payload),
        size_bytes=len(payload),
        media_type="application/octet-stream",
    )
    manifest_payload = json.dumps(
        _resource_document(dependency), separators=(",", ":"), sort_keys=True
    ).encode()
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    manifest_path = inputs / "000-asset"
    manifest_path.write_bytes(manifest_payload)
    descriptor = _descriptor(
        "protocol_resource_coacd",
        environment_digest=f"sha256:{'e' * 64}",
    )
    operation = OperationSpec(
        resource_id="operation.resource_closure",
        version="1.0.0",
        method=descriptor.bind(),
        inputs=(
            OperationInput(
                name="asset",
                resource=VersionedRef(
                    resource_type=ResourceType.ASSET,
                    resource_id="cabinet.resource",
                    version="1.0.0",
                    digest=sha256_digest(manifest_payload),
                ),
            ),
        ),
        outputs=(
            OperationOutputRequirement(
                name="collision_geometry",
                contract="CollisionGeometry",
            ),
        ),
    )
    request = WorkerRequest(
        operation=operation,
        inputs=(
            WorkerInputFile(
                name="asset",
                kind="resource",
                digest=sha256_digest(manifest_payload),
                size_bytes=len(manifest_payload),
                media_type="application/vnd.physrealx.asset+json",
                path="inputs/000-asset",
                resource_type=ResourceType.ASSET,
            ),
        ),
        resource_artifacts=(),
    )
    request_path = tmp_path / "request.json"
    request_path.write_bytes(request.model_dump_json().encode())

    def handler(_request: WorkerRequest, _output: Path) -> object:
        raise AssertionError("handler must not run")

    with pytest.raises(DomainValidationError, match=r"index.*incomplete"):
        run_worker(
            handler,  # type: ignore[arg-type]
            argv=[
                "--request",
                str(request_path),
                "--response",
                str(tmp_path / "outputs" / "response.json"),
            ],
        )


def test_worker_request_rejects_duplicate_resource_artifact_index() -> None:
    descriptor = _descriptor(
        "duplicate_index_coacd",
        environment_digest=f"sha256:{'e' * 64}",
    )
    operation = OperationSpec(
        resource_id="operation.duplicate_index",
        version="1.0.0",
        method=descriptor.bind(),
        inputs=(
            OperationInput(
                name="asset",
                resource=VersionedRef(
                    resource_type=ResourceType.ASSET,
                    resource_id="cabinet.resource",
                    version="1.0.0",
                    digest=f"sha256:{'a' * 64}",
                ),
            ),
        ),
        outputs=(
            OperationOutputRequirement(
                name="collision_geometry",
                contract="CollisionGeometry",
            ),
        ),
    )
    resource_input = WorkerInputFile(
        name="asset",
        kind="resource",
        digest=f"sha256:{'a' * 64}",
        size_bytes=1,
        media_type="application/vnd.physrealx.asset+json",
        path="inputs/000-asset",
        resource_type=ResourceType.ASSET,
    )
    dependency = WorkerResourceArtifact(
        owner="asset",
        json_pointer="/geometry/artifact",
        digest=f"sha256:{'b' * 64}",
        size_bytes=1,
        media_type="model/ply",
        path="inputs/000-asset.artifacts/000000.ply",
    )

    with pytest.raises(ValidationError, match="owner and JSON pointer pairs must be unique"):
        WorkerRequest(
            operation=operation,
            inputs=(resource_input,),
            resource_artifacts=(dependency, dependency),
        )


def test_missing_process_worker_is_reported_before_execution(tmp_path: Path) -> None:
    descriptor, worker = _process_worker(
        tmp_path,
        "missing_coacd",
        (str(tmp_path / "not-installed"),),
    )
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    platform.plugins.register_worker(worker)
    source = platform.artifacts.put_json({"mesh": "cabinet"})

    installation = platform.methods.inspect(MethodCapability.COLLISION_BUILD, descriptor.name)
    assert installation.executable and not installation.ready
    assert "not installed" in installation.detail
    with pytest.raises(WorkerUnavailableError, match="installed but unavailable"):
        _execute_protocol_fixture(platform, source, descriptor.name)


def test_process_worker_cannot_mutate_materialized_inputs(tmp_path: Path) -> None:
    executable = tmp_path / "tampering-worker.py"
    executable.write_text(
        """\
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request", type=Path, required=True)
parser.add_argument("--response", type=Path, required=True)
args = parser.parse_args()
request = json.loads(args.request.read_text(encoding="utf-8"))
source = args.request.parent / request["inputs"][0]["path"]
source.chmod(0o600)
source.write_bytes(b"tampered")
""",
        encoding="utf-8",
    )
    descriptor, worker = _process_worker(
        tmp_path,
        "tampering_coacd",
        (sys.executable, str(executable)),
    )
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    platform.plugins.register_worker(worker)
    source = platform.artifacts.put_json({"mesh": "cabinet"})

    with pytest.raises(MethodExecutionError, match="modified an immutable input"):
        _execute_protocol_fixture(platform, source, descriptor.name)
    assert platform.operations.list() == []


def test_process_worker_timeout_is_structured_and_does_not_publish(tmp_path: Path) -> None:
    executable = tmp_path / "slow-worker.py"
    executable.write_text("import time\ntime.sleep(10)\n", encoding="utf-8")
    descriptor, worker = _process_worker(
        tmp_path,
        "slow_coacd",
        (sys.executable, str(executable)),
        timeout_seconds=0.1,
    )
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    platform.plugins.register_worker(worker)
    source = platform.artifacts.put_json({"mesh": "cabinet"})

    with pytest.raises(MethodExecutionError, match="exceeded its timeout") as captured:
        _execute_protocol_fixture(platform, source, descriptor.name)
    assert captured.value.details == {"timeout_seconds": 0.1}
    assert platform.operations.list() == []


def test_container_worker_requires_matching_pinned_image() -> None:
    digest = f"sha256:{'a' * 64}"
    descriptor = _descriptor(
        "container_coacd",
        mode=MethodExecutionMode.CONTAINER,
        image_digest=digest,
    )

    with pytest.raises(PluginError, match="does not match"):
        ExternalMethodWorker(
            descriptor,
            ContainerLaunch(image=f"ghcr.io/example/coacd@sha256:{'b' * 64}"),
        )
    with pytest.raises(PluginError, match="cannot weaken"):
        ContainerLaunch(
            image=f"ghcr.io/example/coacd@{digest}",
            runtime_arguments=("--privileged",),
        )


def test_container_worker_executes_pinned_image_with_isolated_mounts(tmp_path: Path) -> None:
    runtime = tmp_path / "fake-oci-runtime"
    _write_fake_oci_runtime(runtime)
    digest = f"sha256:{'d' * 64}"
    image = f"ghcr.io/example/coacd@{digest}"
    descriptor = _descriptor(
        "oci_coacd",
        mode=MethodExecutionMode.CONTAINER,
        image_digest=digest,
    )
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    platform.plugins.register_worker(
        ExternalMethodWorker(
            descriptor,
            ContainerLaunch(image=image, runtime=str(runtime)),
        )
    )
    source = platform.artifacts.put_json({"mesh": "cabinet"})

    result = _execute_protocol_fixture(platform, source, descriptor.name)

    assert result.metrics == {"part_count": 5.0}
    assert platform.artifacts.get_bytes(result.outputs[0].artifact) == (
        b'{"source_size": 18, "parts": 5}'
    )


def test_non_in_process_descriptor_cannot_use_an_in_process_worker() -> None:
    environment_digest = f"sha256:{'c' * 64}"

    class MisrepresentedWorker:
        descriptor = _descriptor(
            "misrepresented_coacd",
            environment_digest=environment_digest,
        )

        def execute(
            self,
            spec: OperationSpec,
            artifacts: ArtifactStore,
        ) -> MethodExecutionOutput:
            del spec, artifacts
            raise AssertionError("must not execute")

    with pytest.raises(PluginError, match="require ExternalMethodWorker"):
        PluginRegistry().register_worker(MisrepresentedWorker())


def test_workspace_deployment_discovers_and_probes_process_worker(tmp_path: Path) -> None:
    executable = tmp_path / "worker"
    executable.write_text("#!/bin/sh\necho ready\n", encoding="utf-8")
    executable.chmod(0o755)
    environment = tmp_path / "environment-lock.json"
    environment.write_text(
        json.dumps(
            {
                "schema": "physrealx.worker-environment/1",
                "protocol_version": "2",
                "runtime_files": [
                    {
                        "path": str(executable),
                        "sha256": sha256_digest(executable.read_bytes()),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    descriptor = _descriptor(
        "deployed_coacd",
        environment_digest=sha256_digest(environment.read_bytes()),
    )
    deployment = WorkerDeploymentFile.model_validate(
        {
            "worker": {
                "kind": "process",
                "descriptor": descriptor,
                "command": [str(executable)],
                "probe_command": [str(executable)],
                "environment_manifest": str(environment),
            }
        }
    )
    workspace = tmp_path / "workspace"
    deployment_directory = workspace / "workers.d"
    deployment_directory.mkdir(parents=True)
    (deployment_directory / "coacd.json").write_text(
        deployment.model_dump_json(indent=2), encoding="utf-8"
    )

    platform = PhysRealX.open(workspace)

    installation = platform.methods.inspect(MethodCapability.COLLISION_BUILD, descriptor.name)
    assert installation.ready is True
    assert installation.detail == "ready"


def test_process_probe_rejects_mutated_runtime_file(tmp_path: Path) -> None:
    executable = tmp_path / "worker"
    executable.write_text("#!/bin/sh\necho ready\n", encoding="utf-8")
    executable.chmod(0o755)
    environment = tmp_path / "environment-lock.json"
    environment.write_text(
        json.dumps(
            {
                "schema": "physrealx.worker-environment/1",
                "protocol_version": "2",
                "runtime_files": [
                    {
                        "path": str(executable),
                        "sha256": sha256_digest(executable.read_bytes()),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    descriptor = _descriptor(
        "integrity_coacd",
        environment_digest=sha256_digest(environment.read_bytes()),
    )
    worker = ExternalMethodWorker(
        descriptor,
        ProcessLaunch(
            command=(str(executable),),
            probe_command=(str(executable),),
            environment_manifest=environment,
        ),
    )

    executable.write_text("#!/bin/sh\necho changed\n", encoding="utf-8")

    probe = worker.probe()
    assert probe.ready is False
    assert "changed after installation" in probe.detail


def test_process_probe_rejects_previous_worker_protocol(tmp_path: Path) -> None:
    executable = tmp_path / "worker"
    executable.write_text("#!/bin/sh\necho ready\n", encoding="utf-8")
    executable.chmod(0o755)
    environment = tmp_path / "environment-lock.json"
    environment.write_text(
        json.dumps(
            {
                "schema": "physrealx.worker-environment/1",
                "protocol_version": "1",
                "runtime_files": [
                    {
                        "path": str(executable),
                        "sha256": sha256_digest(executable.read_bytes()),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    descriptor = _descriptor(
        "old_protocol_coacd",
        environment_digest=sha256_digest(environment.read_bytes()),
    )
    worker = ExternalMethodWorker(
        descriptor,
        ProcessLaunch(
            command=(str(executable),),
            environment_manifest=environment,
        ),
    )

    probe = worker.probe()

    assert probe.ready is False
    assert "incompatible worker protocol" in probe.detail


def test_process_probe_rejects_runtime_protocol_mismatch(tmp_path: Path) -> None:
    executable = tmp_path / "worker"
    executable.write_text(
        '#!/bin/sh\necho \'{"worker":"old","ready":true,"protocol_version":"1"}\'\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    environment = tmp_path / "environment-lock.json"
    environment.write_text(
        json.dumps(
            {
                "schema": "physrealx.worker-environment/1",
                "protocol_version": "2",
                "runtime_files": [
                    {
                        "path": str(executable),
                        "sha256": sha256_digest(executable.read_bytes()),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    descriptor = _descriptor(
        "runtime_protocol_coacd",
        environment_digest=sha256_digest(environment.read_bytes()),
    )
    worker = ExternalMethodWorker(
        descriptor,
        ProcessLaunch(
            command=(str(executable),),
            probe_command=(str(executable),),
            environment_manifest=environment,
        ),
    )

    probe = worker.probe()

    assert probe.ready is False
    assert "protocol '1'" in probe.detail


def _machine_probe_record(
    descriptor: MethodDescriptor,
    *,
    ready: bool = True,
    **updates: object,
) -> dict[str, object]:
    record: dict[str, object] = {
        "schema": "physrealx.worker-probe/1",
        "protocol_version": "2",
        "capability": descriptor.capability.value,
        "name": descriptor.name,
        "version": descriptor.version,
        "source_revision": descriptor.source_revision,
        "ready": ready,
        "self_test": {
            "status": "passed" if ready else "failed",
            "checks": {"fixture": ready},
        },
        "checkpoint": None,
        "environment": {
            "python": "3.11.9",
            "executable_digest": sha256_digest(Path(sys.executable).resolve().read_bytes()),
            "dependencies": {"fixture": "1.0.0"},
        },
        "reason": None if ready else "fixture self-test failed",
        "details": {},
    }
    record.update(updates)
    return record


def _worker_with_probe_record(
    tmp_path: Path,
    record_factory: Callable[[MethodDescriptor], dict[str, object]],
) -> tuple[MethodDescriptor, ExternalMethodWorker]:
    executable = tmp_path / "identity-probe"
    executable.write_text("#!/bin/sh\necho placeholder\n", encoding="utf-8")
    executable.chmod(0o755)
    environment = tmp_path / "identity-environment.json"
    environment.write_text(
        json.dumps(
            {
                "schema": "physrealx.worker-environment/1",
                "protocol_version": "2",
                "runtime_files": [
                    {
                        "path": str(executable),
                        "sha256": sha256_digest(executable.read_bytes()),
                    },
                    {
                        "path": str(Path(sys.executable).resolve()),
                        "sha256": sha256_digest(Path(sys.executable).resolve().read_bytes()),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    descriptor = _descriptor(
        "identity_coacd",
        environment_digest=sha256_digest(environment.read_bytes()),
    )
    record = record_factory(descriptor)
    executable.write_text(
        "#!/bin/sh\necho '" + json.dumps(record, separators=(",", ":")) + "'\n",
        encoding="utf-8",
    )
    environment.write_text(
        json.dumps(
            {
                "schema": "physrealx.worker-environment/1",
                "protocol_version": "2",
                "runtime_files": [
                    {
                        "path": str(executable),
                        "sha256": sha256_digest(executable.read_bytes()),
                    },
                    {
                        "path": str(Path(sys.executable).resolve()),
                        "sha256": sha256_digest(Path(sys.executable).resolve().read_bytes()),
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    descriptor = descriptor.model_copy(
        update={
            "runtime": descriptor.runtime.model_copy(
                update={"environment_digest": sha256_digest(environment.read_bytes())}
            )
        }
    )
    return descriptor, ExternalMethodWorker(
        descriptor,
        ProcessLaunch(
            command=(str(executable),),
            probe_command=(str(executable),),
            environment_manifest=environment,
        ),
    )


@pytest.mark.parametrize(
    ("record_factory", "expected"),
    [
        (
            lambda descriptor: _machine_probe_record(descriptor, name="wrong_worker"),
            "identity does not match",
        ),
        (
            lambda descriptor: _machine_probe_record(descriptor, source_revision="v9.9.9"),
            "identity does not match",
        ),
        (
            lambda descriptor: _machine_probe_record(descriptor, ready=False),
            "fixture self-test failed",
        ),
    ],
)
def test_machine_probe_requires_exact_identity_and_readiness(
    tmp_path: Path,
    record_factory: Callable[[MethodDescriptor], dict[str, object]],
    expected: str,
) -> None:
    _, worker = _worker_with_probe_record(tmp_path, record_factory)

    probe = worker.probe()

    assert probe.ready is False
    assert expected in probe.detail


def test_machine_probe_accepts_exact_deployment_identity(tmp_path: Path) -> None:
    _, worker = _worker_with_probe_record(
        tmp_path,
        lambda descriptor: _machine_probe_record(descriptor),
    )

    probe = worker.probe()

    assert probe.ready is True
    assert "physrealx.worker-probe/1" in probe.detail


def test_process_probe_command_must_be_environment_attested(tmp_path: Path) -> None:
    command = tmp_path / "worker"
    command.write_text("#!/bin/sh\necho ready\n", encoding="utf-8")
    command.chmod(0o755)
    unpinned_probe = tmp_path / "unpinned-probe"
    unpinned_probe.write_text("#!/bin/sh\necho ready\n", encoding="utf-8")
    unpinned_probe.chmod(0o755)
    environment = tmp_path / "environment.json"
    environment.write_text(
        json.dumps(
            {
                "schema": "physrealx.worker-environment/1",
                "protocol_version": "2",
                "runtime_files": [
                    {
                        "path": str(command),
                        "sha256": sha256_digest(command.read_bytes()),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    descriptor = _descriptor(
        "unpinned_probe_coacd",
        environment_digest=sha256_digest(environment.read_bytes()),
    )
    worker = ExternalMethodWorker(
        descriptor,
        ProcessLaunch(
            command=(str(command),),
            probe_command=(str(unpinned_probe),),
            environment_manifest=environment,
        ),
    )

    probe = worker.probe()

    assert probe.ready is False
    assert "not attested" in probe.detail


@pytest.mark.parametrize(
    ("mutated_field", "expected"),
    [
        ("request_digest", "immutable request manifest"),
        ("descriptor_digest", "selected method descriptor"),
    ],
)
def test_worker_response_must_echo_request_and_descriptor_identity(
    tmp_path: Path,
    mutated_field: str,
    expected: str,
) -> None:
    script = tmp_path / f"wrong-{mutated_field}.py"
    script.write_text(
        f"""\
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--request", required=True)
parser.add_argument("--response", required=True)
arguments = parser.parse_args()
request_path = Path(arguments.request)
request_bytes = request_path.read_bytes()
request = json.loads(request_bytes)
descriptor_bytes = json.dumps(
    request["operation"]["method"]["descriptor"],
    allow_nan=False,
    ensure_ascii=False,
    separators=(",", ":"),
    sort_keys=True,
).encode("utf-8")
output = Path(arguments.response).parent / "collision.json"
output.write_text('{{"parts": 1}}', encoding="utf-8")
requirement = request["operation"]["outputs"][0]
response = {{
    "protocol_version": "2",
    "request_digest": "sha256:" + hashlib.sha256(request_bytes).hexdigest(),
    "descriptor_digest": "sha256:" + hashlib.sha256(descriptor_bytes).hexdigest(),
    "outputs": [{{
        "name": requirement["name"],
        "contract": requirement["contract"],
        "path": output.name,
        "media_type": "application/json",
    }}],
    "metrics": {{}},
}}
response[{mutated_field!r}] = "sha256:" + "f" * 64
Path(arguments.response).write_text(json.dumps(response), encoding="utf-8")
""",
        encoding="utf-8",
    )
    descriptor, worker = _process_worker(
        tmp_path,
        f"wrong_{mutated_field}_coacd",
        (sys.executable, str(script)),
    )
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    platform.plugins.register_worker(worker)
    source = platform.artifacts.put_json({"mesh": "cabinet"})

    with pytest.raises(MethodExecutionError, match=expected):
        _execute_protocol_fixture(platform, source, descriptor.name)
