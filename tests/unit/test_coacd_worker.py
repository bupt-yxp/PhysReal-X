from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
PROJECT_SOURCE = PROJECT_ROOT / "src"
WORKERS_SOURCE = PROJECT_ROOT / "workers" / "src"
sys.path.insert(0, str(PROJECT_SOURCE))
sys.path.insert(0, str(WORKERS_SOURCE))

from physrealx_workers.coacd_worker import _validate_request  # noqa: E402

from physrealx.domain import (  # noqa: E402
    ArtifactRef,
    MethodAccelerator,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodRuntime,
    OperationInput,
    OperationOutputRequirement,
    OperationSpec,
    sha256_digest,
)
from physrealx.external_workers import WorkerInputFile, WorkerRequest  # noqa: E402


def _request(config: dict[str, object]) -> WorkerRequest:
    source = ArtifactRef(
        digest=sha256_digest(b"visual mesh fixture"),
        size_bytes=len(b"visual mesh fixture"),
        media_type="model/obj",
    )
    descriptor = MethodDescriptor(
        capability=MethodCapability.COLLISION_BUILD,
        name="coacd",
        version="1.0.14",
        api_version="2",
        source_uri="https://github.com/SarahWeiii/CoACD",
        source_revision="v1.0.14",
        license_spdx="MIT",
        runtime=MethodRuntime(
            mode=MethodExecutionMode.PROCESS,
            accelerator=MethodAccelerator.CPU,
            python=">=3.8,<3.13",
            environment_digest=sha256_digest(b"coacd environment"),
        ),
        input_contract="VisualMesh/1",
        output_contract="CollisionGeometry/1+CollisionBuildReport/1",
    )
    operation = OperationSpec(
        resource_id="operation.coacd-fixture",
        version="1.0.0",
        method=descriptor.bind(config=config),
        inputs=(OperationInput(name="asset", artifact=source),),
        outputs=(
            OperationOutputRequirement(
                name="collision_geometry",
                contract="CollisionGeometry/1",
            ),
            OperationOutputRequirement(
                name="collision_build_report",
                contract="CollisionBuildReport/1",
            ),
        ),
    )
    return WorkerRequest(
        operation=operation,
        inputs=(
            WorkerInputFile(
                name="asset",
                kind="artifact",
                digest=source.digest,
                size_bytes=source.size_bytes,
                media_type=source.media_type,
                path="inputs/000-asset.obj",
            ),
        ),
        resource_artifacts=(),
    )


def test_coacd_requires_explicit_metric_unit_and_link_identity() -> None:
    assert _validate_request(_request({"source_unit": "m", "link": "drawer"}))["link"] == "drawer"

    with pytest.raises(ValueError, match="source_unit"):
        _validate_request(_request({"link": "drawer"}))
    with pytest.raises(ValueError, match="source_unit"):
        _validate_request(_request({"source_unit": "mm", "link": "drawer"}))
    with pytest.raises(ValueError, match="link"):
        _validate_request(_request({"source_unit": "m", "link": ""}))


def test_coacd_rejects_unknown_configuration() -> None:
    with pytest.raises(ValueError, match="unsupported CoACD options"):
        _validate_request(_request({"source_unit": "m", "link": "drawer", "pretend_metric": True}))
