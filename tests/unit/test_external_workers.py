from __future__ import annotations

import sys
from pathlib import Path

import pytest

from physrealx.contracts import ArtifactStore, MethodExecutionOutput
from physrealx.domain import (
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodRuntime,
    OperationSpec,
    sha256_digest,
)
from physrealx.errors import MethodExecutionError, PluginError, WorkerUnavailableError
from physrealx.external_workers import ContainerLaunch, ExternalMethodWorker, ProcessLaunch
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
    payload = f"{name}=1.0.0\n".encode()
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


def _write_fake_oci_runtime(path: Path) -> None:
    path.write_text(
        f"""\
#!{sys.executable}
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
request = json.loads(mounts["/work/request.json"].read_text(encoding="utf-8"))
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
            "protocol_version": "1",
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
    result = platform.assets.build_collision(source, method=descriptor.name)

    assert installation.executable and installation.ready
    assert installation.execution_kind == "method_worker"
    assert result.metrics == {"part_count": 4.0}
    assert result.outputs[0].contract == "CollisionGeometry"
    assert platform.artifacts.get_bytes(result.outputs[0].artifact) == (
        b'{"source_size": 18, "parts": 4}'
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
        platform.assets.build_collision(source, method=descriptor.name)


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
        platform.assets.build_collision(source, method=descriptor.name)
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
        platform.assets.build_collision(source, method=descriptor.name)
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

    result = platform.assets.build_collision(source, method=descriptor.name)

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
