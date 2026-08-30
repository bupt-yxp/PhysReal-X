from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest

_RESULT_ROOT = (
    Path(__file__).resolve().parents[2] / "experiments" / "results" / "synthetic-worker-self-tests"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def test_sysid_result_is_explicitly_synthetic_and_matches_frozen_contract() -> None:
    physics_path = _RESULT_ROOT / "sysid" / "identified_joint_physics.json"
    report_path = _RESULT_ROOT / "sysid" / "identification_report.json"
    physics = _load_json(physics_path)
    report = _load_json(report_path)

    assert "cabinet2" not in physics_path.read_text(encoding="utf-8")
    assert "cabinet2" not in report_path.read_text(encoding="utf-8")
    assert physics["schema"] == "physrealx.identified-joint-physics/1"
    assert physics["status"] == "IDENTIFIED"
    assert report["schema"] == "physrealx.physics-identification-report/1"
    assert report["status"] == "IDENTIFIED"
    assert report["diagnostics"]["holdout_rmse"] < report["diagnostics"]["prior_holdout_rmse"]
    assert physics["parameters"]["effective_inertia"]["value"] == pytest.approx(2.3, rel=0.01)


def test_openusd_result_is_self_contained_synthetic_fixture() -> None:
    result_root = _RESULT_ROOT / "openusd"
    usd_path = result_root / "asset.usdc"
    manifest = _load_json(result_root / "asset_package_manifest.json")
    report = _load_json(result_root / "usd_roundtrip_validation_report.json")

    assert manifest["resource_id"] == "synthetic.drawer.fixture"
    assert manifest["self_contained"] is True
    assert manifest["external_dependencies"] == []
    assert manifest["usd"]["digest"] == _sha256(usd_path)
    assert manifest["usd"]["size_bytes"] == usd_path.stat().st_size
    assert report["status"] == "PASSED"
    assert report["failed_checks"] == []
    assert report["usd_digest"] == manifest["usd"]["digest"]
    assert len(report["checks"]) == 40
    assert all(report["checks"].values())


def test_coacd_result_is_an_actual_bound_synthetic_decomposition() -> None:
    result_root = _RESULT_ROOT / "coacd"
    source_path = result_root / "source_visual_mesh.obj"
    collision_path = result_root / "collision.glb"
    report = _load_json(result_root / "collision_build_report.json")
    operation = _load_json(result_root / "operation_manifest.json")

    source_digest = _sha256(source_path)
    collision_digest = _sha256(collision_path)
    assert (
        source_digest == "sha256:f81fba84b1620d3dbea2180ae8836415e8cbbe98b4ae92ff615651da817b6d09"
    )
    assert (
        collision_digest
        == "sha256:4018fd7facdd89f591e0726f31e1ec233578d42fff45af53527d1854e37e3cec"
    )
    assert report["schema"] == "physrealx.collision-build-report/1"
    assert report["status"] == "PASSED"
    assert report["unit"] == "m"
    assert report["link"] == "synthetic_l_prism"
    assert report["method"] == {
        "name": "coacd",
        "source_revision": "v1.0.14",
        "version": "1.0.14",
    }
    assert report["source_geometry"] == {
        "digest": source_digest,
        "media_type": "model/obj",
        "size_bytes": source_path.stat().st_size,
    }
    assert report["collision_geometry"] == {
        "digest": collision_digest,
        "media_type": "model/gltf-binary",
        "size_bytes": collision_path.stat().st_size,
    }
    assert all(report["gates"].values())
    assert report["metrics"]["source_watertight"] == 1.0
    assert report["metrics"]["convex_parts"] == 2.0
    assert report["metrics"]["volume_ratio"] == pytest.approx(1.0003865753930263)

    spec = operation["spec"]
    assert spec["inputs"][0]["artifact"] == report["source_geometry"]
    assert spec["method"]["descriptor"]["name"] == "coacd"
    assert spec["method"]["descriptor"]["runtime"]["environment_digest"].startswith("sha256:")
    outputs = {item["name"]: item for item in operation["outputs"]}
    assert outputs["collision_geometry"]["contract"] == "CollisionGeometry/1"
    assert outputs["collision_geometry"]["artifact"] == report["collision_geometry"]
    assert outputs["collision_build_report"]["contract"] == "CollisionBuildReport/1"
    assert outputs["collision_build_report"]["artifact"]["digest"] == _sha256(
        result_root / "collision_build_report.json"
    )

    payload = collision_path.read_bytes()
    magic, version, declared_size = struct.unpack("<4sII", payload[:12])
    assert magic == b"glTF"
    assert version == 2
    assert declared_size == len(payload)
    assert "cabinet2" not in source_path.read_text(encoding="ascii").lower()
    assert (
        "cabinet2"
        not in (result_root / "operation_manifest.json").read_text(encoding="utf-8").lower()
    )
