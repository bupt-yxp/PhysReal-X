from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from physrealx import platform as platform_module
from physrealx import plugins as plugins_module
from physrealx.contracts import ArtifactStore, MethodExecutionOutput
from physrealx.domain import (
    ArtifactRef,
    MethodAccelerator,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodRuntime,
    OperationOutput,
    OperationSpec,
)
from physrealx.errors import (
    DomainValidationError,
    PluginError,
    ResourceConflictError,
    ResourceNotFoundError,
)
from physrealx.methods import builtin_method
from physrealx.platform import PhysRealX
from physrealx.plugins import PluginRegistry
from physrealx.reference import reference_asset


def test_method_runtime_only_accepts_the_project_environment() -> None:
    with pytest.raises(ValidationError, match="Input should be 'in_process'"):
        MethodRuntime.model_validate(
            {
                "mode": "container",
                "accelerator": MethodAccelerator.CUDA,
                "python": ">=3.10,<3.13",
                "minimum_vram_gb": 12,
            }
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


def test_workspace_method_catalog_binds_exact_registered_descriptor(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    descriptor = builtin_method(
        MethodCapability.COLLISION_BUILD,
        "custom_collision",
        input_contract="TriangleMesh",
        output_contract="CollisionGeometry",
    )
    platform.methods.register(descriptor)

    selected = platform.methods.bind(
        MethodCapability.COLLISION_BUILD,
        "custom_collision",
        config={"threshold": 0.05},
    )
    assert selected.descriptor == descriptor
    assert selected.config == {"threshold": 0.05}
    with pytest.raises(ResourceConflictError, match="already registered"):
        platform.methods.register(descriptor)


def test_optional_training_method_is_not_advertised_without_dependencies(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform_module, "find_spec", lambda name: None)

    platform = PhysRealX.open(tmp_path, discover_plugins=False)

    assert "torch_bc" not in {item.name for item in platform.methods.list()}


def test_default_catalog_does_not_advertise_candidate_adapters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platform_module, "create_colmap_adapter", lambda: None)
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    installations = platform.methods.installations()
    names = {item.descriptor.name for item in installations}

    assert {"artmodeling", "artgs", "reference_drawer", "scripted_drawer"} <= names
    assert names.isdisjoint(
        {
            "blenderproc",
            "colmap",
            "ditto",
            "lerobot_v3",
            "metric_alignment",
            "mujoco_sysid",
            "omni_dc",
            "ros2_control",
        }
    )
    assert {
        "ransac_umeyama",
        "rgb_guided_fill",
        "rigid_pair_fit",
        "coacd",
        "mujoco",
        "openusd",
    } <= names
    assert all(item.executable for item in installations)


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
        artifact_id=f"art_{'f' * 32}",
        size_bytes=1,
        media_type="application/octet-stream",
    )

    with pytest.raises(DomainValidationError, match="not present"):
        platform.methods.bind(
            MethodCapability.DEPTH_ENHANCE,
            "omni_dc",
            checkpoint=missing,
        )


class _CollisionAdapter:
    descriptor = builtin_method(
        MethodCapability.COLLISION_BUILD,
        "test_coacd",
        input_contract="VisualMesh/1",
        output_contract="CollisionGeometry/1+CollisionBuildReport/1",
    )

    def execute(
        self,
        spec: OperationSpec,
        artifacts: ArtifactStore,
    ) -> MethodExecutionOutput:
        source = spec.inputs[0].artifact
        assert source is not None
        collision = artifacts.put_bytes(
            b"glTF fixture",
            media_type="model/gltf-binary",
        )
        report = artifacts.put_json(
            {
                "schema": "physrealx.collision-build-report/1",
                "status": "PASSED",
                "method": {
                    "name": "test_coacd",
                    "version": "1.0.0",
                    "source_revision": "builtin-reference",
                },
                "unit": "m",
                "link": "fixture",
                "source_geometry": source.model_dump(mode="json"),
                "collision_geometry": collision.model_dump(mode="json"),
                "metrics": {"convex_parts": 1.0},
                "gates": {"decomposition_nonempty": True},
            }
        )
        artifacts_by_contract = {
            "CollisionGeometry/1": collision,
            "CollisionBuildReport/1": report,
        }
        outputs = tuple(
            OperationOutput(
                name=requirement.name,
                contract=requirement.contract,
                artifact=artifacts_by_contract[requirement.contract],
            )
            for requirement in spec.outputs
        )
        return MethodExecutionOutput(outputs=outputs, metrics={"part_count": 3.0})


def test_capability_service_executes_and_publishes_immutable_evidence(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    visual_mesh = platform.artifacts.put_bytes(
        b"glTF fixture input",
        media_type="model/gltf-binary",
    )
    platform.methods.register(_CollisionAdapter.descriptor)
    platform.plugins.register_adapter(_CollisionAdapter())

    first = platform.assets.build_collision(visual_mesh, method="test_coacd")
    second = platform.assets.build_collision(visual_mesh, method="test_coacd")

    assert first == second
    assert first.spec.method.capability is MethodCapability.COLLISION_BUILD
    assert first.outputs[0].name == "collision_geometry"
    assert first.metrics == {"part_count": 3.0}
    operation_ref = platform.operations.list()[0]
    assert platform.operations.get(operation_ref) == first
    assert operation_ref == first.reference(operation_ref.artifact_id)


def test_metadata_only_method_cannot_be_executed(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    descriptor = builtin_method(
        MethodCapability.COLLISION_BUILD,
        "custom_collision",
        input_contract="VisualMesh/1",
        output_contract="CollisionGeometry/1+CollisionBuildReport/1",
    )
    platform.methods.register(descriptor)
    source = platform.artifacts.put_bytes(
        b"glTF fixture",
        media_type="model/gltf-binary",
    )

    with pytest.raises(ResourceNotFoundError, match="metadata only"):
        platform.assets.build_collision(source, method="custom_collision")

    installation = platform.methods.inspect(MethodCapability.COLLISION_BUILD, "custom_collision")
    assert not installation.executable
    assert not installation.ready


def test_failed_entry_point_registration_is_atomic(monkeypatch: pytest.MonkeyPatch) -> None:
    descriptor = builtin_method(
        MethodCapability.COLLISION_BUILD,
        "broken_coacd",
        input_contract="TriangleMesh",
        output_contract="CollisionGeometry",
    )

    class BrokenEntryPoint:
        name = "broken"
        value = "broken_adapter:register"

        @staticmethod
        def load():
            def register(registry: PluginRegistry) -> None:
                registry.register_method(descriptor)
                raise RuntimeError("registration failed after mutation")

            return register

    class FakeEntryPoints:
        items = (BrokenEntryPoint(),)

        def __iter__(self):
            return iter(self.items)

        def get(self, group: str, default: object = ()):
            del group, default
            return self.items

    monkeypatch.setattr(
        plugins_module,
        "entry_points",
        lambda **kwargs: FakeEntryPoints(),
    )
    registry = PluginRegistry()

    with pytest.raises(PluginError, match="failed during registration"):
        registry.discover()

    assert registry.descriptors() == ()


def test_trainable_asset_profile_requires_validated_physical_evidence(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)

    report = platform.assets.validate(
        reference_asset(),
        profile="trainable",
    )

    assert not report.passed
    assert "asset_not_validated" in {issue.code for issue in report.issues}
