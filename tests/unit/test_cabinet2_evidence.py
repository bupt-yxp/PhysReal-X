from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

_RESULT_ROOT = (
    Path(__file__).resolve().parents[2] / "experiments" / "results" / "cabinet2-lifecycle"
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def test_cabinet2_evidence_is_internally_bound_and_truthfully_blocked() -> None:
    evidence = _load_json(_RESULT_ROOT / "cabinet2_lifecycle_evidence.json")
    assert evidence["schema"] == "physrealx.cabinet2-lifecycle-evidence/1"
    assert evidence["source"]["files"] == 1149
    assert evidence["source"]["bytes"] == 2_016_218_738
    assert evidence["camera_scale_status"] == "unresolved"
    assert evidence["readiness"]["status"] == "BLOCKED"

    stages = {item["stage"]: item for item in evidence["readiness"]["stages"]}
    assert len(stages) == 18
    assert {name for name, item in stages.items() if item["status"] == "READY"} == {
        "ingestion",
        "camera",
    }
    assert all(
        item["status"] == "BLOCKED"
        for name, item in stages.items()
        if name not in {"ingestion", "camera"}
    )
    assert set(evidence["scientific_boundary"]) == {
        "asset_assembly",
        "collision",
        "ditto",
        "geometry",
        "metric_alignment",
        "physics_sysid",
        "policy_world_model",
        "ros2",
    }
    metric_installation = next(
        item for item in evidence["method_installations"] if item["stage"] == "metric_alignment"
    )
    assert metric_installation["method"] == "metric_alignment"
    assert metric_installation["installed"] is True
    assert metric_installation["executable"] is True
    assert metric_installation["ready"] is True
    assert stages["metric_alignment"]["status"] == "BLOCKED"
    assert [item["code"] for item in stages["metric_alignment"]["blockers"]] == ["evidence_missing"]

    report = _load_json(_RESULT_ROOT / "camera_recovery_report.json")
    rig = _load_json(_RESULT_ROOT / "camera_rig_calibration.json")
    assert report["scale_status"] == "unresolved"
    assert rig["scale_status"] == "unresolved"
    for name, actual in evidence["camera_metrics"].items():
        assert evidence["camera_operation"]["metrics"][name] == pytest.approx(actual)
    assert report["input"]["selected_images"] == evidence["camera_metrics"]["selected_images"]
    for name in (
        "component_count",
        "mean_reprojection_error_px",
        "mean_track_length",
        "registered_images",
        "registration_rate",
        "sparse_points",
    ):
        assert report["model"][name] == pytest.approx(evidence["camera_metrics"][name])
    assert report["execution"]["cpu_fallback_used"] is False
    assert report["execution"]["effective_device"] == "cpu"
    assert "log_tail" not in report["execution"]

    output_paths = {
        "camera_recovery_report": _RESULT_ROOT / "camera_recovery_report.json",
        "camera_rig_calibration": _RESULT_ROOT / "camera_rig_calibration.json",
        "sparse_reconstruction_bundle": _RESULT_ROOT / "sparse_reconstruction_bundle.zip",
    }
    for output in evidence["camera_operation"]["outputs"]:
        path = output_paths[output["name"]]
        assert output["artifact"]["size_bytes"] == path.stat().st_size
        assert output["artifact"]["digest"] == _sha256(path)


def test_cabinet2_runtime_attestations_and_sparse_archive_match_report() -> None:
    evidence = _load_json(_RESULT_ROOT / "cabinet2_lifecycle_evidence.json")
    attestation_root = _RESULT_ROOT / "runtime-attestations"
    for item in evidence["runtime_attestations"]:
        path = attestation_root / item["path"]
        assert path.is_file()
        assert item["size_bytes"] == path.stat().st_size
        assert item["sha256"] == _sha256(path)

    archive_path = _RESULT_ROOT / "sparse_reconstruction_bundle.zip"
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert names
        assert len(names) == len(set(names))
        assert all(not name.endswith("project.ini") for name in names)
        for name in names:
            member = PurePosixPath(name)
            assert not member.is_absolute()
            assert ".." not in member.parts


def test_documentation_images_are_exact_copies_of_measured_results() -> None:
    image_root = Path(__file__).resolve().parents[2] / "docs" / "assets" / "images" / "experiments"
    for result_name, documentation_name in (
        ("cabinet2_lifecycle_readiness.png", "cabinet2-lifecycle-readiness.png"),
        ("cabinet2_measured_metrics.png", "cabinet2-measured-metrics.png"),
    ):
        assert _sha256(_RESULT_ROOT / result_name) == _sha256(image_root / documentation_name)
