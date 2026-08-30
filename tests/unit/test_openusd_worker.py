from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT_ROOT = Path(__file__).parents[2]
PROJECT_SOURCE = PROJECT_ROOT / "src"
WORKERS_SOURCE = PROJECT_ROOT / "workers" / "src"
sys.path.insert(0, str(PROJECT_SOURCE))
sys.path.insert(0, str(WORKERS_SOURCE))

from physrealx_workers.openusd_worker import _handle  # noqa: E402

from physrealx.domain import (  # noqa: E402
    ArtifactRef,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodRuntime,
    OperationInput,
    OperationOutputRequirement,
    OperationSpec,
    ResourceType,
    VersionedRef,
    canonical_json_bytes,
    sha256_digest,
)
from physrealx.external_workers import (  # noqa: E402
    WorkerInputFile,
    WorkerRequest,
    WorkerResourceArtifact,
)

_IDENTITY_POSE = {
    "translation_m": [0.0, 0.0, 0.0],
    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
}


def _parameter(value: float, unit: str, evidence_digest: str) -> dict[str, Any]:
    width = max(abs(value) * 0.05, 1e-9)
    return {
        "value": value,
        "unit": unit,
        "source": "identified",
        "confidence_interval": [max(0.0, value - width), value + width],
        "evidence_digest": evidence_digest,
    }


def _mesh_payload(offset: float) -> bytes:
    return (
        "o tetra\n"
        f"v {offset:.4f} 0 0\n"
        f"v {offset + 0.1:.4f} 0 0\n"
        f"v {offset:.4f} 0.1 0\n"
        f"v {offset:.4f} 0 0.1\n"
        "f 1 3 2\n"
        "f 1 2 4\n"
        "f 1 4 3\n"
        "f 2 3 4\n"
    ).encode()


def _reference(payload: bytes, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        digest=sha256_digest(payload),
        size_bytes=len(payload),
        media_type=media_type,
    )


def _request(
    tmp_path: Path,
    *,
    scale_status: str = "resolved",
    physics_status: str = "IDENTIFIED",
    physics_evidence_state: str = "identified",
) -> WorkerRequest:
    inputs = tmp_path / "inputs"
    dependencies = inputs / "000-asset.artifacts"
    dependencies.mkdir(parents=True)

    scale_evidence_payload = b"verified-marker-scale:0.050000m"
    articulation_evidence_payload = b"held-out-interaction-pair:drawer"
    physics_evidence_payload = b"calibrated-force-position-trajectory"
    scale_evidence = _reference(scale_evidence_payload, "application/octet-stream")
    articulation_evidence = _reference(articulation_evidence_payload, "application/octet-stream")
    physics_evidence = _reference(physics_evidence_payload, "application/octet-stream")
    metric_scale = {
        "status": scale_status,
        "meters_per_unit": 1.0,
        "evidence_digest": scale_evidence.digest,
    }
    articulation = {
        "schema": "physrealx.articulation-model/1",
        "status": "IDENTIFIED",
        "evidence_state": "identified",
        "metric_scale": {
            "status": "resolved",
            "meters_per_unit": 1.0,
            "evidence_digest": scale_evidence.digest,
        },
        "links": ["base", "drawer"],
        "joints": [
            {
                "name": "drawer_slide",
                "parent": "base",
                "child": "drawer",
                "type": "prismatic",
                "axis": "X",
                "lower_limit": 0.0,
                "upper_limit": 0.28,
                "limit_unit": "m",
                "parent_frame": _IDENTITY_POSE,
                "child_frame": _IDENTITY_POSE,
            }
        ],
        "evidence_digest": articulation_evidence.digest,
    }
    bodies = []
    for link, mass in (("base", 8.0), ("drawer", 2.0)):
        bodies.append(
            {
                "link": link,
                "mass_kg": _parameter(mass, "kg", physics_evidence.digest),
                "center_of_mass_m": [0.0, 0.0, 0.0],
                "inertia_kg_m2": [
                    [0.08, 0.0, 0.0],
                    [0.0, 0.09, 0.0],
                    [0.0, 0.0, 0.1],
                ],
                "static_friction": _parameter(0.35, "1", physics_evidence.digest),
                "dynamic_friction": _parameter(0.28, "1", physics_evidence.digest),
                "restitution": _parameter(0.02, "1", physics_evidence.digest),
                "evidence_digest": physics_evidence.digest,
            }
        )
    physics = {
        "schema": "physrealx.identified-physics/1",
        "status": physics_status,
        "evidence_state": physics_evidence_state,
        "metric_scale": {
            "status": "resolved",
            "meters_per_unit": 1.0,
            "evidence_digest": scale_evidence.digest,
        },
        "bodies": bodies,
        "joints": [
            {
                "name": "drawer_slide",
                "damping": _parameter(0.7, "N*s/m", physics_evidence.digest),
                "friction": _parameter(0.24, "N", physics_evidence.digest),
                "stiffness": _parameter(0.0, "N/m", physics_evidence.digest),
                "evidence_digest": physics_evidence.digest,
            }
        ],
    }
    articulation_payload = canonical_json_bytes(articulation)
    physics_payload = canonical_json_bytes(physics)
    articulation_ref = _reference(articulation_payload, "application/json")
    physics_ref = _reference(physics_payload, "application/json")
    meshes = [
        ("base_visual", "base", _mesh_payload(0.0000)),
        ("drawer_visual", "drawer", _mesh_payload(0.0001)),
        ("base_collision", "base", _mesh_payload(0.0002)),
        ("drawer_collision", "drawer", _mesh_payload(0.0003)),
    ]
    mesh_refs = [
        (name, link, _reference(payload, "model/obj"), payload) for name, link, payload in meshes
    ]
    assembly = {
        "schema": "physrealx.asset-assembly-bundle/1",
        "resource_id": "synthetic.drawer.fixture",
        "version": "1.0.0",
        "display_name": "Synthetic Drawer Fixture",
        "metric_scale": metric_scale,
        "up_axis": "Z",
        "root_link": "base",
        "visual_meshes": [
            {
                "name": name,
                "link": link,
                "artifact": reference.model_dump(mode="json"),
                "units": "meters",
                "local_pose": _IDENTITY_POSE,
            }
            for name, link, reference, _ in mesh_refs[:2]
        ],
        "collision_meshes": [
            {
                "name": name,
                "link": link,
                "artifact": reference.model_dump(mode="json"),
                "units": "meters",
                "local_pose": _IDENTITY_POSE,
            }
            for name, link, reference, _ in mesh_refs[2:]
        ],
        "articulation_model": articulation_ref.model_dump(mode="json"),
        "identified_physics": physics_ref.model_dump(mode="json"),
        "evidence": [
            scale_evidence.model_dump(mode="json"),
            articulation_evidence.model_dump(mode="json"),
            physics_evidence.model_dump(mode="json"),
        ],
    }
    assembly_payload = canonical_json_bytes(assembly)
    assembly_path = inputs / "000-asset"
    assembly_path.write_bytes(assembly_payload)
    assembly_digest = sha256_digest(assembly_payload)
    dependency_values = [
        ("/articulation_model", articulation_ref, articulation_payload),
        ("/identified_physics", physics_ref, physics_payload),
        ("/evidence/0", scale_evidence, scale_evidence_payload),
        ("/evidence/1", articulation_evidence, articulation_evidence_payload),
        ("/evidence/2", physics_evidence, physics_evidence_payload),
        *[
            (
                "/{}/{}/artifact".format(
                    "visual_meshes" if index < 2 else "collision_meshes",
                    index if index < 2 else index - 2,
                ),
                reference,
                payload,
            )
            for index, (_, _, reference, payload) in enumerate(mesh_refs)
        ],
    ]
    resource_artifacts = []
    for index, (pointer, reference, payload) in enumerate(dependency_values):
        suffix = {
            "application/json": ".json",
            "application/octet-stream": ".bin",
            "model/obj": ".obj",
        }[reference.media_type]
        destination = dependencies / f"{index:06d}{suffix}"
        destination.write_bytes(payload)
        resource_artifacts.append(
            WorkerResourceArtifact(
                owner="asset",
                json_pointer=pointer,
                digest=reference.digest,
                size_bytes=reference.size_bytes,
                media_type=reference.media_type,
                path=destination.relative_to(tmp_path).as_posix(),
            )
        )
    descriptor = MethodDescriptor(
        capability=MethodCapability.ASSET_PACKAGE,
        name="openusd",
        version="1.0.0",
        api_version="1",
        source_uri="https://github.com/PixarAnimationStudios/OpenUSD",
        source_revision="v23.05",
        license_spdx="Apache-2.0",
        runtime=MethodRuntime(
            mode=MethodExecutionMode.PROCESS,
            accelerator="cpu",
            python=">=3.8,<3.13",
            environment_digest=f"sha256:{'e' * 64}",
        ),
        input_contract="AssetAssemblyBundle/1",
        output_contract=("UsdAssetBundle/1+AssetPackageManifest/1+UsdRoundTripValidationReport/1"),
    )
    operation = OperationSpec(
        resource_id="operation.asset.package.fixture",
        version="1.0.0",
        method=descriptor.bind(),
        inputs=(
            OperationInput(
                name="asset",
                resource=VersionedRef(
                    resource_type=ResourceType.ASSET,
                    resource_id="synthetic.drawer.fixture",
                    version="1.0.0",
                    digest=assembly_digest,
                ),
            ),
        ),
        outputs=(
            OperationOutputRequirement(name="usd_asset_bundle", contract="UsdAssetBundle/1"),
            OperationOutputRequirement(
                name="asset_package_manifest", contract="AssetPackageManifest/1"
            ),
            OperationOutputRequirement(
                name="roundtrip_validation_report", contract="UsdRoundTripValidationReport/1"
            ),
        ),
    )
    return WorkerRequest(
        operation=operation,
        inputs=(
            WorkerInputFile(
                name="asset",
                kind="resource",
                digest=assembly_digest,
                size_bytes=len(assembly_payload),
                media_type="application/vnd.physrealx.asset-assembly+json",
                path=assembly_path.relative_to(tmp_path).as_posix(),
                resource_type=ResourceType.ASSET,
            ),
        ),
        resource_artifacts=tuple(resource_artifacts),
    )


def test_openusd_worker_rejects_unresolved_metric_scale_before_importing_usd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path, scale_status="unresolved")
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="metric scale is unresolved"):
        _handle(request, output)


def test_openusd_worker_rejects_prior_physics_before_importing_usd(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(
        tmp_path,
        physics_status="PRIOR",
        physics_evidence_state="prior",
    )
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="must be IDENTIFIED or VALIDATED"):
        _handle(request, output)


def test_openusd_worker_rejects_incomplete_resource_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    request = request.model_copy(update={"resource_artifacts": request.resource_artifacts[:-1]})
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="resource closure is incomplete"):
        _handle(request, output)


def test_openusd_worker_authors_and_reopens_self_contained_physics_asset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("trimesh")
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    request = _request(tmp_path)
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)

    response = _handle(request, output)

    assert [item.contract for item in response.outputs] == [
        "UsdAssetBundle/1",
        "AssetPackageManifest/1",
        "UsdRoundTripValidationReport/1",
    ]
    stage = Usd.Stage.Open(str(output / "asset.usdc"))
    assert stage is not None
    assert UsdGeom.GetStageMetersPerUnit(stage) == pytest.approx(1.0)
    assert str(UsdGeom.GetStageUpAxis(stage)) == "Z"
    drawer = stage.GetPrimAtPath("/Asset/Links/drawer")
    assert drawer.HasAPI(UsdPhysics.RigidBodyAPI)
    assert UsdPhysics.MassAPI(drawer).GetMassAttr().Get() == pytest.approx(2.0)
    joint = UsdPhysics.PrismaticJoint.Get(stage, "/Asset/Joints/drawer_slide")
    assert joint.GetAxisAttr().Get() == "X"
    assert joint.GetUpperLimitAttr().Get() == pytest.approx(0.28)
    assert not stage.GetRootLayer().GetExternalReferences()
    manifest = json.loads((output / "asset_package_manifest.json").read_text(encoding="utf-8"))
    report = json.loads(
        (output / "usd_roundtrip_validation_report.json").read_text(encoding="utf-8")
    )
    assert manifest["self_contained"] is True
    assert manifest["external_dependencies"] == []
    assert report["status"] == "PASSED"
    assert report["failed_checks"] == []
    assert response.metrics["external_dependencies"] == 0.0
