from __future__ import annotations

import hashlib
import json
import warnings
import zipfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from physrealx.domain import (
    ArtifactRef,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodRuntime,
    OperationInput,
    OperationOutputRequirement,
    OperationSpec,
    sha256_digest,
)
from physrealx.errors import MethodExecutionError
from physrealx.external_workers import (
    ExternalMethodWorker,
    ProcessLaunch,
    WorkerProducedFile,
    WorkerResponse,
    _validate_output_contract,
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _produced(
    contract: str,
    *,
    name: str = "result",
    path: str = "result.json",
    media_type: str = "application/json",
) -> WorkerProducedFile:
    return WorkerProducedFile(
        name=name,
        contract=contract,
        path=path,
        media_type=media_type,
    )


def _validate(
    path: Path,
    contract: str,
    *,
    media_type: str = "application/json",
    requirement: OperationOutputRequirement | None = None,
) -> None:
    selected = requirement or OperationOutputRequirement(name="result", contract=contract)
    _validate_output_contract(
        selected,
        _produced(contract, media_type=media_type),
        path,
        execution_size_limit=64 * 1024 * 1024 * 1024,
    )


def test_output_requirement_can_declare_extensible_payload_constraints() -> None:
    requirement = OperationOutputRequirement(
        name="future_depth",
        contract="FutureDepthBundle/1",
        content_kind="json",
        media_types=("application/vnd.example.depth+json",),
        max_size_bytes=4096,
        schema_discriminator="example.future-depth/1",
    )

    assert requirement.content_kind == "json"
    assert requirement.max_size_bytes == 4096
    with pytest.raises(ValidationError, match="unique"):
        requirement.model_copy(update={"media_types": ("application/json", "application/json")})
    with pytest.raises(ValidationError, match="only valid for JSON"):
        requirement.model_copy(update={"content_kind": "zip", "schema_discriminator": "example/1"})


@pytest.mark.parametrize(
    ("contract", "document"),
    [
        (
            "CameraRigCalibration/1",
            {
                "schema": "physrealx.camera-rig-calibration/1",
                "method": {},
                "components": [],
                "scale_status": "unresolved",
            },
        ),
        (
            "CameraRecoveryReport/1",
            {
                "schema": "physrealx.camera-recovery-report/1",
                "execution": {},
                "model": {},
                "scale_status": "unresolved",
            },
        ),
        (
            "IdentifiedJointPhysics/1",
            {
                "schema": "physrealx.identified-joint-physics/1",
                "status": "IDENTIFIED",
                "adapter": "physrealx_mujoco_sysid_adapter",
                "evidence_digest": f"sha256:{'a' * 64}",
            },
        ),
        (
            "PhysicsIdentificationReport/1",
            {
                "schema": "physrealx.physics-identification-report/1",
                "status": "NOT_IDENTIFIABLE",
                "adapter": "physrealx_mujoco_sysid_adapter",
                "diagnostics": {},
            },
        ),
        (
            "AssetPackageManifest/1",
            {
                "schema": "physrealx.asset-package-manifest/1",
                "resource_id": "cabinet2",
                "version": "1.0.0",
                "usd": {},
                "self_contained": True,
                "roundtrip_status": "PASSED",
            },
        ),
        (
            "UsdRoundTripValidationReport/1",
            {
                "schema": "physrealx.usd-roundtrip-validation-report/1",
                "status": "PASSED",
                "checks": {},
                "failed_checks": [],
            },
        ),
        (
            "DepthEnhancementReport/1",
            {
                "schema": "physrealx.depth-enhancement-report/1",
                "status": "PASSED",
                "metrics": {},
                "gates": {},
            },
        ),
        (
            "ArticulationModel/1",
            {
                "schema": "physrealx.articulation-model/1",
                "status": "IDENTIFIED",
                "links": [],
                "joints": [],
            },
        ),
        (
            "ArticulationRecoveryReport/1",
            {
                "schema": "physrealx.articulation-recovery-report/1",
                "status": "PASSED",
                "gates": {},
                "diagnostics": {},
            },
        ),
        (
            "CollisionBuildReport/1",
            {
                "schema": "physrealx.collision-build-report/1",
                "status": "PASSED",
                "method": {
                    "name": "coacd",
                    "version": "1.0.14",
                    "source_revision": "v1.0.14",
                },
                "unit": "m",
                "link": "drawer",
                "source_geometry": {},
                "collision_geometry": {},
                "metrics": {},
                "gates": {},
            },
        ),
    ],
)
def test_known_json_contracts_require_schema_types_and_status(
    tmp_path: Path,
    contract: str,
    document: dict[str, object],
) -> None:
    path = tmp_path / "result.json"
    _write_json(path, document)

    _validate(path, contract)


def test_json_output_rejects_media_schema_type_duplicate_keys_and_nonfinite(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    valid = {
        "schema": "physrealx.camera-recovery-report/1",
        "execution": {},
        "model": {},
        "scale_status": "unresolved",
    }
    _write_json(path, valid)
    with pytest.raises(MethodExecutionError, match="declared media type"):
        _validate(path, "CameraRecoveryReport/1", media_type="text/plain")

    _write_json(path, {**valid, "schema": "wrong/1"})
    with pytest.raises(MethodExecutionError, match="schema discriminator"):
        _validate(path, "CameraRecoveryReport/1")

    _write_json(path, {**valid, "model": []})
    with pytest.raises(MethodExecutionError, match=r"model must be object"):
        _validate(path, "CameraRecoveryReport/1")

    path.write_text(
        '{"schema":"physrealx.camera-recovery-report/1",'
        '"schema":"physrealx.camera-recovery-report/1",'
        '"execution":{},"model":{},"scale_status":"unresolved"}',
        encoding="utf-8",
    )
    with pytest.raises(MethodExecutionError, match="unambiguous"):
        _validate(path, "CameraRecoveryReport/1")

    path.write_text(
        '{"schema":"physrealx.camera-recovery-report/1",'
        '"execution":{},"model":{"score":NaN},"scale_status":"unresolved"}',
        encoding="utf-8",
    )
    with pytest.raises(MethodExecutionError, match="finite UTF-8 JSON"):
        _validate(path, "CameraRecoveryReport/1")


def test_unknown_contract_can_enforce_custom_json_media_size_and_discriminator(
    tmp_path: Path,
) -> None:
    path = tmp_path / "result.json"
    _write_json(path, {"schema": "example.future-depth/1", "values": [1.0]})
    requirement = OperationOutputRequirement(
        name="result",
        contract="FutureDepthBundle/1",
        content_kind="json",
        media_types=("application/vnd.example.depth+json",),
        max_size_bytes=1024,
        schema_discriminator="example.future-depth/1",
    )

    _validate(
        path,
        requirement.contract,
        media_type="application/vnd.example.depth+json",
        requirement=requirement,
    )

    path.write_bytes(b"x" * 1025)
    with pytest.raises(MethodExecutionError, match="size limit"):
        _validate(
            path,
            requirement.contract,
            media_type="application/vnd.example.depth+json",
            requirement=requirement,
        )


def _zip_info(name: str, *, mode: int = 0o100400) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = mode << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _sparse_bundle(path: Path, *, payload: bytes = b"camera-data") -> None:
    name = "components/0000/text/cameras.txt"
    manifest = {
        "schema": "physrealx.sparse-reconstruction-bundle/1",
        "components": ["0000"],
        "files": [
            {
                "path": name,
                "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
                "size_bytes": len(payload),
            }
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(_zip_info(name), payload)
        archive.writestr(
            _zip_info("bundle-manifest.json"),
            json.dumps(manifest, allow_nan=False).encode(),
        )


def test_sparse_zip_requires_safe_exact_manifest_and_member_digests(tmp_path: Path) -> None:
    path = tmp_path / "bundle.zip"
    _sparse_bundle(path)

    _validate(path, "SparseReconstructionBundle/1", media_type="application/zip")

    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr(_zip_info("unlisted.txt"), b"unlisted")
    with pytest.raises(MethodExecutionError, match="exactly match"):
        _validate(path, "SparseReconstructionBundle/1", media_type="application/zip")


@pytest.mark.parametrize("unsafe_name", ("../escape", "/absolute", "folder\\portable"))
def test_zip_rejects_traversal_and_nonportable_paths(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(_zip_info(unsafe_name), b"bad")
        archive.writestr(_zip_info("bundle-manifest.json"), b"{}")

    with pytest.raises(MethodExecutionError, match=r"unsafe|non-portable"):
        _validate(path, "SparseReconstructionBundle/1", media_type="application/zip")


def test_zip_rejects_duplicate_symlink_and_compression_bomb_members(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.zip"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(duplicate, "w") as archive:
            archive.writestr(_zip_info("same.txt"), b"one")
            archive.writestr(_zip_info("same.txt"), b"two")
    with pytest.raises(MethodExecutionError, match="duplicate"):
        _validate(duplicate, "SparseReconstructionBundle/1", media_type="application/zip")

    symlink = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink, "w") as archive:
        archive.writestr(_zip_info("link", mode=0o120777), b"target")
    with pytest.raises(MethodExecutionError, match="symbolic links"):
        _validate(symlink, "SparseReconstructionBundle/1", media_type="application/zip")

    bomb = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb, "w") as archive:
        archive.writestr(_zip_info("zeros.bin"), b"\0" * (2 * 1024 * 1024))
    with pytest.raises(MethodExecutionError, match="compression-ratio"):
        _validate(bomb, "SparseReconstructionBundle/1", media_type="application/zip")


def test_enhanced_depth_zip_uses_frame_payload_index(tmp_path: Path) -> None:
    path = tmp_path / "depth.zip"
    payload = b"NUMPY"
    member = "depth/frame-000.npy"
    manifest = {
        "schema": "physrealx.enhanced-depth-bundle/1",
        "depth_encoding": {"unit": "m"},
        "frames": [
            {
                "frame_id": "frame-000",
                "depth": {
                    "path": member,
                    "digest": f"sha256:{hashlib.sha256(payload).hexdigest()}",
                    "size_bytes": len(payload),
                },
            }
        ],
    }
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(_zip_info(member), payload)
        archive.writestr(_zip_info("manifest.json"), json.dumps(manifest).encode())

    _validate(path, "EnhancedDepthBundle/1", media_type="application/zip")


class _RecordingStore:
    def __init__(self) -> None:
        self.published: list[Path] = []

    def put_file(self, source: Path, *, media_type: str) -> ArtifactRef:
        self.published.append(source)
        payload = source.read_bytes()
        return ArtifactRef(
            digest=sha256_digest(payload),
            size_bytes=len(payload),
            media_type=media_type,
        )


def _worker(
    tmp_path: Path,
    *,
    capability: MethodCapability = MethodCapability.CAMERA_RECOVER,
    output_contract: str = "CameraRigCalibration/1+CameraRecoveryReport/1",
) -> tuple[MethodDescriptor, ExternalMethodWorker]:
    manifest = tmp_path / "environment.lock"
    manifest.write_bytes(b"locked-environment")
    descriptor = MethodDescriptor(
        capability=capability,
        name="contract-validation-worker",
        version="1.0.0",
        api_version="2",
        source_uri="https://example.test/worker",
        source_revision="v1.0.0",
        license_spdx="MIT",
        runtime=MethodRuntime(
            mode=MethodExecutionMode.PROCESS,
            python=">=3.8,<3.13",
            environment_digest=sha256_digest(manifest.read_bytes()),
        ),
        input_contract="ObservationSet/1",
        output_contract=output_contract,
    )
    return descriptor, ExternalMethodWorker(
        descriptor,
        ProcessLaunch(command=("/bin/true",), environment_manifest=manifest),
    )


def test_all_sibling_outputs_are_validated_before_any_artifact_is_published(
    tmp_path: Path,
) -> None:
    descriptor, worker = _worker(tmp_path)
    output = tmp_path / "outputs"
    output.mkdir()
    _write_json(
        output / "rig.json",
        {
            "schema": "physrealx.camera-rig-calibration/1",
            "method": {},
            "components": [],
            "scale_status": "unresolved",
        },
    )
    _write_json(output / "report.json", {"schema": "wrong/1"})
    spec = OperationSpec(
        resource_id="operation.contract-validation",
        version="1.0.0",
        method=descriptor.bind(),
        outputs=(
            OperationOutputRequirement(name="rig", contract="CameraRigCalibration/1"),
            OperationOutputRequirement(name="report", contract="CameraRecoveryReport/1"),
        ),
    )
    response = WorkerResponse(
        outputs=(
            _produced("CameraRigCalibration/1", name="rig", path="rig.json"),
            _produced("CameraRecoveryReport/1", name="report", path="report.json"),
        )
    )
    store = _RecordingStore()

    with pytest.raises(MethodExecutionError, match="schema discriminator"):
        worker._publish_response(  # type: ignore[arg-type]
            spec,
            response,
            output,
            store,
        )

    assert store.published == []


def test_joint_physics_and_identification_report_must_describe_the_same_result(
    tmp_path: Path,
) -> None:
    descriptor, worker = _worker(
        tmp_path,
        capability=MethodCapability.PHYSICS_CALIBRATE,
        output_contract="IdentifiedJointPhysics/1+PhysicsIdentificationReport/1",
    )
    output = tmp_path / "outputs"
    output.mkdir()
    physics_path = output / "joint-physics.json"
    report_path = output / "report.json"
    _write_json(
        physics_path,
        {
            "schema": "physrealx.identified-joint-physics/1",
            "status": "IDENTIFIED",
            "adapter": "physrealx_mujoco_sysid_adapter",
            "evidence_digest": f"sha256:{'a' * 64}",
        },
    )
    _write_json(
        report_path,
        {
            "schema": "physrealx.physics-identification-report/1",
            "status": "NOT_IDENTIFIABLE",
            "adapter": "physrealx_mujoco_sysid_adapter",
            "diagnostics": {},
        },
    )
    spec = OperationSpec(
        resource_id="operation.joint-physics-validation",
        version="1.0.0",
        method=descriptor.bind(),
        outputs=(
            OperationOutputRequirement(
                name="identified_joint_physics",
                contract="IdentifiedJointPhysics/1",
            ),
            OperationOutputRequirement(
                name="identification_report",
                contract="PhysicsIdentificationReport/1",
            ),
        ),
    )
    response = WorkerResponse(
        outputs=(
            _produced(
                "IdentifiedJointPhysics/1",
                name="identified_joint_physics",
                path=physics_path.name,
            ),
            _produced(
                "PhysicsIdentificationReport/1",
                name="identification_report",
                path=report_path.name,
            ),
        )
    )
    store = _RecordingStore()

    with pytest.raises(MethodExecutionError, match="disagree on status or adapter"):
        worker._publish_response(  # type: ignore[arg-type]
            spec,
            response,
            output,
            store,
        )
    assert store.published == []


def test_collision_report_binds_metric_source_and_exact_produced_glb(tmp_path: Path) -> None:
    descriptor, worker = _worker(
        tmp_path,
        capability=MethodCapability.COLLISION_BUILD,
        output_contract="CollisionGeometry/1+CollisionBuildReport/1",
    )
    output = tmp_path / "outputs"
    output.mkdir()
    collision_path = output / "collision.glb"
    collision_path.write_bytes(b"glTF verified collision fixture")
    collision = ArtifactRef(
        digest=sha256_digest(collision_path.read_bytes()),
        size_bytes=collision_path.stat().st_size,
        media_type="model/gltf-binary",
    )
    source = ArtifactRef(
        digest=sha256_digest(b"visual-source"),
        size_bytes=len(b"visual-source"),
        media_type="model/obj",
    )
    report_path = output / "collision-build-report.json"
    report = {
        "schema": "physrealx.collision-build-report/1",
        "status": "PASSED",
        "method": {
            "name": "coacd",
            "version": "1.0.14",
            "source_revision": "v1.0.14",
        },
        "unit": "m",
        "link": "drawer",
        "source_geometry": source.model_dump(mode="json"),
        "collision_geometry": collision.model_dump(mode="json"),
        "metrics": {"convex_parts": 2.0},
        "gates": {"decomposition_nonempty": True, "metric_unit_declared": True},
    }
    _write_json(report_path, report)
    spec = OperationSpec(
        resource_id="operation.collision-validation",
        version="1.0.0",
        method=descriptor.bind(),
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
    response = WorkerResponse(
        outputs=(
            _produced(
                "CollisionGeometry/1",
                name="collision_geometry",
                path=collision_path.name,
                media_type="model/gltf-binary",
            ),
            _produced(
                "CollisionBuildReport/1",
                name="collision_build_report",
                path=report_path.name,
            ),
        )
    )
    store = _RecordingStore()

    report["source_geometry"] = {
        **source.model_dump(mode="json"),
        "digest": sha256_digest(b"different-source"),
    }
    _write_json(report_path, report)
    with pytest.raises(MethodExecutionError, match="does not close over"):
        worker._publish_response(  # type: ignore[arg-type]
            spec,
            response,
            output,
            store,
        )
    assert store.published == []

    report["source_geometry"] = source.model_dump(mode="json")
    _write_json(report_path, report)
    result = worker._publish_response(  # type: ignore[arg-type]
        spec,
        response,
        output,
        store,
    )
    assert len(result.outputs) == 2
    assert len(store.published) == 2


def test_asset_package_manifest_and_roundtrip_report_bind_the_exact_usd_bytes(
    tmp_path: Path,
) -> None:
    descriptor, worker = _worker(
        tmp_path,
        capability=MethodCapability.ASSET_PACKAGE,
        output_contract=("UsdAssetBundle/1+AssetPackageManifest/1+UsdRoundTripValidationReport/1"),
    )
    output = tmp_path / "outputs"
    output.mkdir()
    usd = output / "asset.usdc"
    usd.write_bytes(b"PXR-USDC\x00verified-fixture")
    usd_digest = sha256_digest(usd.read_bytes())
    manifest_path = output / "manifest.json"
    report_path = output / "report.json"
    manifest = {
        "schema": "physrealx.asset-package-manifest/1",
        "resource_id": "cabinet2",
        "version": "1.0.0",
        "usd": {
            "digest": sha256_digest(b"wrong"),
            "size_bytes": usd.stat().st_size,
            "media_type": "model/vnd.usd+binary",
        },
        "self_contained": True,
        "external_dependencies": [],
        "roundtrip_status": "PASSED",
    }
    _write_json(manifest_path, manifest)
    _write_json(
        report_path,
        {
            "schema": "physrealx.usd-roundtrip-validation-report/1",
            "status": "PASSED",
            "checks": {"stage_opens": True, "joint_limits": True},
            "failed_checks": [],
            "usd_digest": usd_digest,
        },
    )
    spec = OperationSpec(
        resource_id="operation.asset-package-validation",
        version="1.0.0",
        method=descriptor.bind(),
        outputs=(
            OperationOutputRequirement(name="usd", contract="UsdAssetBundle/1"),
            OperationOutputRequirement(name="manifest", contract="AssetPackageManifest/1"),
            OperationOutputRequirement(name="report", contract="UsdRoundTripValidationReport/1"),
        ),
    )
    response = WorkerResponse(
        outputs=(
            _produced(
                "UsdAssetBundle/1",
                name="usd",
                path=usd.name,
                media_type="model/vnd.usd+binary",
            ),
            _produced("AssetPackageManifest/1", name="manifest", path=manifest_path.name),
            _produced("UsdRoundTripValidationReport/1", name="report", path=report_path.name),
        )
    )
    store = _RecordingStore()

    with pytest.raises(MethodExecutionError, match="does not identify the produced USD"):
        worker._publish_response(  # type: ignore[arg-type]
            spec,
            response,
            output,
            store,
        )
    assert store.published == []

    manifest["usd"] = {
        "digest": usd_digest,
        "size_bytes": usd.stat().st_size,
        "media_type": "model/vnd.usd+binary",
    }
    _write_json(manifest_path, manifest)
    result = worker._publish_response(  # type: ignore[arg-type]
        spec,
        response,
        output,
        store,
    )

    assert len(result.outputs) == 3
    assert len(store.published) == 3
