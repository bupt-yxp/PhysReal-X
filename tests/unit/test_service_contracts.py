from __future__ import annotations

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
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.methods import builtin_method
from physrealx.platform import PhysRealX
from physrealx.reference import reference_asset


class _CanonicalWorker:
    def __init__(self, descriptor: MethodDescriptor) -> None:
        self.descriptor = descriptor

    def execute(
        self,
        spec: OperationSpec,
        artifacts: ArtifactStore,
    ) -> MethodExecutionOutput:
        outputs = []
        for requirement in spec.outputs:
            media_type = (
                requirement.media_types[0] if requirement.media_types else "application/json"
            )
            outputs.append(
                OperationOutput(
                    name=requirement.name,
                    contract=requirement.contract,
                    artifact=artifacts.put_bytes(
                        f"fixture:{requirement.name}".encode(),
                        media_type=media_type,
                    ),
                )
            )
        return MethodExecutionOutput(outputs=tuple(outputs), metrics={"executed": 1.0})


class _WrongMediaCollisionWorker:
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
    platform.plugins.register_worker(_CanonicalWorker(descriptor))
    return name


AssetInvocation = Callable[[PhysRealX, str, VersionedRef, ArtifactRef], OperationManifest]


@pytest.mark.parametrize(
    ("capability", "input_contract", "output_contract", "invoke", "expected"),
    (
        (
            MethodCapability.DEPTH_ENHANCE,
            "CalibratedRGBDBundle/1",
            "EnhancedDepthBundle/1+DepthEnhancementReport/1",
            lambda px, method, resource, artifact: px.assets.enhance_depth(
                resource,
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
            MethodCapability.ARTICULATION_RECOVER,
            "PairedInteractionPointCloudBundle/1",
            "ArticulationModel/1+ArticulationRecoveryReport/1",
            lambda px, method, resource, artifact: px.assets.articulate(
                resource,
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
                resource,
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
                resource,
                method=method,
            ),
            (
                ("usd_asset_bundle", "UsdAssetBundle/1", "binary", "model/vnd.usd+binary"),
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
def test_asset_services_own_exact_versioned_worker_contracts(
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


def test_service_rejects_artifact_when_worker_requires_resource_closure(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    descriptor = builtin_method(
        MethodCapability.ASSET_PACKAGE,
        "openusd_fixture",
        input_contract="AssetAssemblyBundle/1",
        output_contract=("UsdAssetBundle/1+AssetPackageManifest/1+UsdRoundTripValidationReport/1"),
    )
    platform.methods.register(descriptor)
    artifact = platform.artifacts.put_json({"schema": "physrealx.asset-assembly-bundle/1"})

    with pytest.raises(DomainValidationError, match="must be an immutable resource"):
        platform.methods.execute(
            MethodCapability.ASSET_PACKAGE,
            descriptor.name,
            inputs={"asset": artifact},
        )
    assert platform.operations.list() == []


def test_in_process_worker_cannot_bypass_service_media_contract(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path, discover_plugins=False)
    platform.plugins.register_worker(_WrongMediaCollisionWorker())
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
