from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest

from physrealx.contracts import ArtifactStore, MethodExecutionOutput
from physrealx.domain import (
    ArtifactRef,
    MethodCapability,
    MethodDescriptor,
    OperationManifest,
    OperationOutput,
    OperationSpec,
    VersionedRef,
)
from physrealx.errors import DomainValidationError, MethodExecutionError, ResourceConflictError
from physrealx.methods import builtin_method
from physrealx.platform import PhysRealX
from physrealx.reference import reference_asset


class _CanonicalAdapter:
    def __init__(self, descriptor: MethodDescriptor) -> None:
        self.descriptor = descriptor
        self.calls = 0

    def execute(
        self,
        spec: OperationSpec,
        artifacts: ArtifactStore,
    ) -> MethodExecutionOutput:
        self.calls += 1
        outputs = []
        for requirement in spec.outputs:
            media_type = (
                requirement.media_types[0] if requirement.media_types else "application/json"
            )
            payload = f"fixture:{requirement.name}".encode()
            if requirement.content_kind == "json":
                document = {"schema": requirement.schema_discriminator or "fixture"}
                if requirement.schema_discriminator == "physrealx.geometry-reconstruction-report/1":
                    document.update(
                        unit="m",
                        coordinate_frame="capture",
                        prediction={
                            "parts": {"whole": {}},
                            "segmented": False,
                            "states": {
                                "observed": {
                                    "whole": artifacts.put_bytes(b"mesh").model_dump(mode="json")
                                }
                            },
                        },
                    )
                payload = json.dumps(document).encode()
            elif requirement.content_kind == "zip":
                stream = io.BytesIO()
                with zipfile.ZipFile(stream, "w") as archive:
                    archive.writestr("manifest.json", "{}")
                payload = stream.getvalue()
            outputs.append(
                OperationOutput(
                    name=requirement.name,
                    contract=requirement.contract,
                    artifact=artifacts.put_bytes(
                        payload,
                        media_type=media_type,
                    ),
                )
            )
        return MethodExecutionOutput(outputs=tuple(outputs), metrics={"executed": 1.0})


class _WrongMediaCollisionAdapter:
    descriptor = builtin_method(
        MethodCapability.COLLISION_BUILD,
        "wrong_media_collision",
        input_contract="VisualMesh/1",
        output_contract="CollisionGeometry/1+CollisionBuildReport/1",
    )

    def execute(
        self,
        spec: OperationSpec,
        artifacts: ArtifactStore,
    ) -> MethodExecutionOutput:
        collision = artifacts.put_json({"not": "a GLB collision mesh"})
        report = artifacts.put_json(
            {
                "schema": "physrealx.collision-build-report/1",
                "status": "PASSED",
            }
        )
        return MethodExecutionOutput(
            outputs=(
                OperationOutput(
                    name=spec.outputs[0].name,
                    contract=spec.outputs[0].contract,
                    artifact=collision,
                ),
                OperationOutput(
                    name=spec.outputs[1].name,
                    contract=spec.outputs[1].contract,
                    artifact=report,
                ),
            ),
            metrics={},
        )


def _install(
    platform: PhysRealX,
    capability: MethodCapability,
    *,
    input_contract: str,
    output_contract: str,
) -> str:
    name = f"test_{capability.value.replace('.', '_')}"
    descriptor = builtin_method(
        capability,
        name,
        input_contract=input_contract,
        output_contract=output_contract,
    )
    platform.plugins.register_adapter(_CanonicalAdapter(descriptor))
    return name


AssetInvocation = Callable[[PhysRealX, str, VersionedRef, ArtifactRef], OperationManifest]


def test_capability_and_explicit_operation_share_validated_execution(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    adapter = _CanonicalAdapter(
        builtin_method(
            MethodCapability.PHYSICS_CALIBRATE,
            "counted_identification",
            input_contract="JointSystemIdentificationEvidence/1",
            output_contract="IdentifiedJointPhysics/1+PhysicsIdentificationReport/1",
        )
    )
    platform.plugins.register_adapter(adapter)
    evidence = platform.artifacts.put_json({"fixture": "identification"})

    original = platform.assets.calibrate_physics(evidence, method=adapter.descriptor.name)
    assert platform.operations.execute(original.spec) == original
    assert platform.assets.calibrate_physics(evidence, method=adapter.descriptor.name) == original
    assert adapter.calls == 1

    changed = original.spec.model_copy(update={"arguments": {"different": True}})
    with pytest.raises(ResourceConflictError, match="different specification"):
        platform.operations.execute(changed)
    assert adapter.calls == 1

    missing = original.outputs[0].artifact
    next((tmp_path / "artifacts").rglob(missing.artifact_id)).unlink()
    with pytest.raises(DomainValidationError, match="unavailable or corrupt output"):
        platform.operations.execute(original.spec)
    assert adapter.calls == 1


@pytest.mark.parametrize(
    ("capability", "input_contract", "output_contract", "invoke", "expected"),
    (
        (
            MethodCapability.DEPTH_ENHANCE,
            "CalibratedRGBDBundle/1",
            "EnhancedDepthBundle/1+DepthEnhancementReport/1",
            lambda px, method, resource, artifact: px.assets.enhance_depth(
                artifact,
                method=method,
            ),
            (
                ("enhanced_depth", "EnhancedDepthBundle/1", "zip", "application/zip"),
                (
                    "enhancement_report",
                    "DepthEnhancementReport/1",
                    "json",
                    "application/json",
                ),
            ),
        ),
        (
            MethodCapability.GEOMETRY_RECONSTRUCT,
            "MetricMultiStateRGBDBundle/1",
            "VisualGeometryBundle/1+GeometryReconstructionReport/1",
            lambda px, method, resource, artifact: px.assets.reconstruct_geometry(
                artifact,
                method=method,
            ),
            (
                (
                    "visual_geometry_bundle",
                    "VisualGeometryBundle/1",
                    "zip",
                    "application/zip",
                ),
                (
                    "geometry_reconstruction_report",
                    "GeometryReconstructionReport/1",
                    "json",
                    "application/json",
                ),
            ),
        ),
        (
            MethodCapability.ARTICULATION_RECOVER,
            "PairedInteractionPointCloudBundle/1",
            "ArticulationModel/1+ArticulationRecoveryReport/1",
            lambda px, method, resource, artifact: px.assets.articulate(
                artifact,
                method=method,
            ),
            (
                ("articulation_model", "ArticulationModel/1", "json", "application/json"),
                (
                    "articulation_recovery_report",
                    "ArticulationRecoveryReport/1",
                    "json",
                    "application/json",
                ),
            ),
        ),
        (
            MethodCapability.COLLISION_BUILD,
            "VisualMesh/1",
            "CollisionGeometry/1+CollisionBuildReport/1",
            lambda px, method, resource, artifact: px.assets.build_collision(
                artifact,
                method=method,
            ),
            (
                (
                    "collision_geometry",
                    "CollisionGeometry/1",
                    "binary",
                    "model/gltf-binary",
                ),
                (
                    "collision_build_report",
                    "CollisionBuildReport/1",
                    "json",
                    "application/json",
                ),
            ),
        ),
        (
            MethodCapability.PHYSICS_CALIBRATE,
            "JointSystemIdentificationEvidence/1",
            "IdentifiedJointPhysics/1+PhysicsIdentificationReport/1",
            lambda px, method, resource, artifact: px.assets.calibrate_physics(
                artifact,
                method=method,
            ),
            (
                (
                    "identified_joint_physics",
                    "IdentifiedJointPhysics/1",
                    "json",
                    "application/json",
                ),
                (
                    "identification_report",
                    "PhysicsIdentificationReport/1",
                    "json",
                    "application/json",
                ),
            ),
        ),
        (
            MethodCapability.ASSET_PACKAGE,
            "AssetAssemblyBundle/1",
            "UsdAssetBundle/1+AssetPackageManifest/1+UsdRoundTripValidationReport/1",
            lambda px, method, resource, artifact: px.assets.package(
                artifact,
                method=method,
            ),
            (
                ("usd_asset_bundle", "UsdAssetBundle/1", "binary", "model/vnd.usdz+zip"),
                (
                    "asset_package_manifest",
                    "AssetPackageManifest/1",
                    "json",
                    "application/json",
                ),
                (
                    "roundtrip_validation_report",
                    "UsdRoundTripValidationReport/1",
                    "json",
                    "application/json",
                ),
            ),
        ),
    ),
)
def test_asset_services_own_exact_versioned_adapter_contracts(
    tmp_path: Path,
    capability: MethodCapability,
    input_contract: str,
    output_contract: str,
    invoke: AssetInvocation,
    expected: tuple[tuple[str, str, str, str], ...],
) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    resource = platform.assets.register(reference_asset())
    artifact = platform.artifacts.put_bytes(
        b"glTF fixture input",
        media_type="model/gltf-binary",
    )
    method = _install(
        platform,
        capability,
        input_contract=input_contract,
        output_contract=output_contract,
    )

    manifest = invoke(platform, method, resource, artifact)
    requirements = manifest.spec.outputs

    assert (
        tuple(
            (
                item.name,
                item.contract,
                item.content_kind,
                item.media_types[0],
            )
            for item in requirements
        )
        == expected
    )
    assert manifest.spec.method.descriptor.input_contract == input_contract
    assert tuple(item.contract for item in manifest.outputs) == tuple(
        item.contract for item in requirements
    )


def test_service_rejects_a_method_that_redefines_its_schema(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    resource = platform.assets.register(reference_asset())
    descriptor = builtin_method(
        MethodCapability.ARTICULATION_RECOVER,
        "legacy_ditto_shape",
        input_contract="PointCloudPair",
        output_contract="ArticulationModel",
    )
    platform.methods.register(descriptor)

    with pytest.raises(DomainValidationError, match="service requires"):
        platform.assets.articulate(resource, method=descriptor.name)
    assert platform.operations.list() == []


def test_service_rejects_resource_when_adapter_requires_artifact(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    descriptor = builtin_method(
        MethodCapability.ASSET_PACKAGE,
        "openusd_fixture",
        input_contract="AssetAssemblyBundle/1",
        output_contract=("UsdAssetBundle/1+AssetPackageManifest/1+UsdRoundTripValidationReport/1"),
    )
    platform.methods.register(descriptor)
    artifact = platform.assets.register(reference_asset())

    with pytest.raises(DomainValidationError, match="must be an immutable artifact"):
        platform.methods.execute(
            MethodCapability.ASSET_PACKAGE,
            descriptor.name,
            inputs={"asset": artifact},
        )
    assert platform.operations.list() == []


def test_in_process_adapter_cannot_bypass_service_media_contract(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    platform.plugins.register_adapter(_WrongMediaCollisionAdapter())
    mesh = platform.artifacts.put_bytes(b"glTF fixture", media_type="model/gltf-binary")

    with pytest.raises(MethodExecutionError, match="media type"):
        platform.assets.build_collision(mesh, method="wrong_media_collision")
    assert platform.operations.list() == []


def test_service_canonicalizes_input_order_before_building_operation_identity(
    tmp_path: Path,
) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    method = _install(
        platform,
        MethodCapability.WORLD_MODEL_ROLLOUT,
        input_contract="ModelBundle/1+ContextWindow/1+ActionSequence/1",
        output_contract="WorldModelRollout/1+UncertaintyReport/1",
    )
    model = platform.artifacts.put_json({"kind": "model"})
    context = platform.artifacts.put_json({"kind": "context"})
    actions = platform.artifacts.put_json({"kind": "actions"})

    manifest = platform.methods.execute(
        MethodCapability.WORLD_MODEL_ROLLOUT,
        method,
        inputs={"actions": actions, "model": model, "context": context},
    )

    assert tuple(item.name for item in manifest.spec.inputs) == (
        "model",
        "context",
        "actions",
    )
