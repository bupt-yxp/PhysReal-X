from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from physrealx.digital_twin import DigitalTwinStage, DigitalTwinStep
from physrealx.domain import (
    ArtifactRef,
    MethodAccelerator,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodRuntime,
)
from physrealx.errors import DomainValidationError, ResourceConflictError
from physrealx.methods import builtin_method
from physrealx.platform import PhysRealX


def test_container_method_requires_immutable_image() -> None:
    with pytest.raises(ValidationError, match="immutable image_digest"):
        MethodRuntime(
            mode=MethodExecutionMode.CONTAINER,
            accelerator=MethodAccelerator.CUDA,
            python=">=3.10,<3.13",
            minimum_vram_gb=12,
        )


def test_method_descriptor_rejects_mutable_source_revision() -> None:
    runtime = MethodRuntime(
        mode=MethodExecutionMode.IN_PROCESS,
        python=">=3.8,<3.13",
    )
    with pytest.raises(ValidationError, match="immutable release"):
        MethodDescriptor(
            capability=MethodCapability.COLLISION_BUILD,
            name="coacd",
            version="1.0.0",
            api_version="1",
            source_uri="https://github.com/SarahWeiii/CoACD",
            source_revision="main",
            license_spdx="MIT",
            runtime=runtime,
            input_contract="TriangleMesh",
            output_contract="CollisionGeometry",
        )


def test_digital_twin_stage_rejects_wrong_method_capability() -> None:
    wrong = builtin_method(
        MethodCapability.COLLISION_BUILD,
        "coacd",
        input_contract="TriangleMesh",
        output_contract="CollisionGeometry",
    ).bind()
    with pytest.raises(ValidationError, match=r"geometry\.reconstruct"):
        DigitalTwinStep(
            stage=DigitalTwinStage.GEOMETRY_RECONSTRUCTION,
            method=wrong,
        )


def test_workspace_method_catalog_binds_exact_registered_descriptor(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    descriptor = builtin_method(
        MethodCapability.COLLISION_BUILD,
        "coacd",
        input_contract="TriangleMesh",
        output_contract="CollisionGeometry",
    )
    platform.methods.register(descriptor)

    selected = platform.methods.bind(
        MethodCapability.COLLISION_BUILD,
        "coacd",
        config={"threshold": 0.05},
    )
    assert selected.descriptor == descriptor
    assert selected.config == {"threshold": 0.05}
    with pytest.raises(ResourceConflictError, match="already registered"):
        platform.methods.register(descriptor)


def test_workspace_rejects_unpublished_method_checkpoint(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    descriptor = builtin_method(
        MethodCapability.DEPTH_ENHANCE,
        "omni_dc",
        input_contract="RGBDFrames",
        output_contract="EnhancedDepthFrames",
    )
    platform.methods.register(descriptor)
    missing = ArtifactRef(
        digest=f"sha256:{'f' * 64}",
        size_bytes=1,
        media_type="application/octet-stream",
    )

    with pytest.raises(DomainValidationError, match="not present"):
        platform.methods.bind(
            MethodCapability.DEPTH_ENHANCE,
            "omni_dc",
            checkpoint=missing,
        )
