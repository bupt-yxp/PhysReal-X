from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
PROJECT_SOURCE = PROJECT_ROOT / "src"
WORKERS_SOURCE = PROJECT_ROOT / "workers" / "src"
sys.path.insert(0, str(PROJECT_SOURCE))
sys.path.insert(0, str(WORKERS_SOURCE))

from physrealx_workers.metric_alignment_worker import _handle  # noqa: E402

from physrealx.domain import (  # noqa: E402
    ArtifactRef,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodRuntime,
    OperationInput,
    OperationOutputRequirement,
    OperationSpec,
    canonical_json_bytes,
    sha256_digest,
)
from physrealx.errors import MethodExecutionError  # noqa: E402
from physrealx.external_workers import (  # noqa: E402
    WorkerInputFile,
    WorkerRequest,
    _validate_output_contract,
    _validate_output_set_closure,
)


def _similarity(points: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    angle = 0.31
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return 0.37 * (points @ rotation.T) + np.asarray([0.22, -0.41, 1.3])


def _correspondences(
    source: np.ndarray[Any, Any],
    *,
    prefix: str,
) -> list[dict[str, Any]]:
    target = _similarity(source)
    return [
        {
            "id": f"{prefix}-{index}",
            "source_xyz": point.tolist(),
            "target_xyz_m": metric.tolist(),
            "uncertainty_m": 0.0005,
        }
        for index, (point, metric) in enumerate(zip(source, target))
    ]


def _documents() -> tuple[dict[str, Any], dict[str, Any]]:
    rig: dict[str, Any] = {
        "schema": "physrealx.camera-rig-calibration/1",
        "method": {"name": "colmap", "version": "3.13.0"},
        "calibration_type": "independent_camera_trajectory",
        "coordinate_convention": {
            "frame": "one independent colmap_world frame per component",
            "pose": "world_to_camera",
            "quaternion_order": "wxyz",
            "translation_unit": "arbitrary",
        },
        "scale_status": "unresolved",
        "scale_reason": "similarity gauge",
        "components": [
            {
                "component_id": "0",
                "coordinate_frame": "colmap_component_0",
                "cameras": [
                    {
                        "camera_id": 1,
                        "model": "PINHOLE",
                        "width": 640,
                        "height": 480,
                        "parameters": [500.0, 500.0, 320.0, 240.0],
                    }
                ],
                "registered_views": [
                    {
                        "image_id": 1,
                        "camera_id": 1,
                        "image_name": "frame.png",
                        "source": {
                            "stream_name": "camera.rgb",
                            "state": "start",
                            "frame_index": 0,
                            "artifact_digest": f"sha256:{'1' * 64}",
                        },
                        "world_to_camera": {
                            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                            "translation": [0.0, 0.0, 0.0],
                        },
                    }
                ],
            }
        ],
    }
    train_source = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.2, 0.1, 0.8],
            [0.8, 0.2, 0.6],
            [0.1, 0.8, 0.5],
            [0.7, 0.7, 0.9],
        ]
    )
    holdout_source = np.asarray(
        [
            [0.15, 0.2, 0.2],
            [0.75, 0.2, 0.35],
            [0.2, 0.75, 0.55],
            [0.7, 0.65, 0.8],
        ]
    )
    evidence: dict[str, Any] = {
        "schema": "physrealx.metric-alignment-evidence/1",
        "resource_id": "cabinet2.metric-alignment",
        "version": "1.0.0",
        "unit": "m",
        "camera_rig_digest": "pending",
        "component_id": "0",
        "source_frame": "colmap_component_0",
        "target_frame": "cabinet2_capture_world",
        "metric_reference": {
            "kind": "calibrated_marker_length",
            "identifier": "aruco-board-001",
            "value_m": 0.2,
            "marker_family": "aruco_6x6_250",
            "marker_size_m": 0.2,
            "layout": "board-layout-v1",
            "layout_source_uri": "file:///capture/cabinet2/board-layout.json",
            "source": "calibration-certificate-001",
        },
        "provenance": {
            "capture_id": "cabinet2-metric-session-001",
            "measurement_method": "aruco_bundle_adjustment",
            "source_uri": "file:///capture/cabinet2/metric-evidence.json",
        },
        "train_correspondences": _correspondences(train_source, prefix="train"),
        "holdout_correspondences": _correspondences(holdout_source, prefix="holdout"),
    }
    # One grossly corrupted training target proves that RANSAC, not all-point LS, is used.
    evidence["train_correspondences"][-1]["target_xyz_m"] = [3.0, -2.0, 4.0]
    return rig, evidence


def _request(
    tmp_path: Path,
    *,
    mutate: Any | None = None,
    config: dict[str, Any] | None = None,
) -> WorkerRequest:
    input_directory = tmp_path / "inputs"
    input_directory.mkdir()
    rig, evidence = _documents()
    rig_payload = canonical_json_bytes(rig)
    rig_digest = sha256_digest(rig_payload)
    evidence["camera_rig_digest"] = rig_digest
    if mutate is not None:
        mutate(rig, evidence)
        rig_payload = canonical_json_bytes(rig)
        rig_digest = sha256_digest(rig_payload)
        if evidence.get("camera_rig_digest") != "intentionally-wrong":
            evidence["camera_rig_digest"] = rig_digest
    evidence_payload = canonical_json_bytes(evidence)
    evidence_digest = sha256_digest(evidence_payload)
    rig_path = input_directory / "000-camera_rig"
    evidence_path = input_directory / "001-evidence"
    rig_path.write_bytes(rig_payload)
    evidence_path.write_bytes(evidence_payload)
    descriptor = MethodDescriptor(
        capability=MethodCapability.CAMERA_ALIGN_METRIC,
        name="metric_alignment",
        version="1.0.0",
        api_version="1",
        source_uri="https://github.com/bupt-yxp/PhysReal-X",
        source_revision="adapter-1.0.0",
        license_spdx="MIT",
        runtime=MethodRuntime(
            mode=MethodExecutionMode.PROCESS,
            accelerator="cpu",
            python=">=3.10,<3.13",
            environment_digest=f"sha256:{'e' * 64}",
        ),
        input_contract="CameraRigCalibration/1+MetricAlignmentEvidence/1",
        output_contract="CameraRigCalibration/1+MetricAlignmentReport/1",
    )
    operation = OperationSpec(
        resource_id="operation.camera.metric.fixture",
        version="1.0.0",
        method=descriptor.bind(
            config={
                "inlier_threshold_m": 0.002,
                "max_holdout_rmse_m": 0.001,
                "max_holdout_relative_rmse": 0.001,
                **(config or {}),
            }
        ),
        inputs=(
            OperationInput(
                name="camera_rig",
                artifact=ArtifactRef(
                    digest=rig_digest,
                    size_bytes=len(rig_payload),
                    media_type="application/json",
                ),
            ),
            OperationInput(
                name="evidence",
                artifact=ArtifactRef(
                    digest=evidence_digest,
                    size_bytes=len(evidence_payload),
                    media_type="application/json",
                ),
            ),
        ),
        outputs=(
            OperationOutputRequirement(
                name="camera_rig_calibration", contract="CameraRigCalibration/1"
            ),
            OperationOutputRequirement(
                name="metric_alignment_report", contract="MetricAlignmentReport/1"
            ),
        ),
    )
    return WorkerRequest(
        operation=operation,
        inputs=(
            WorkerInputFile(
                name="camera_rig",
                kind="artifact",
                digest=rig_digest,
                size_bytes=len(rig_payload),
                media_type="application/json",
                path=rig_path.relative_to(tmp_path).as_posix(),
            ),
            WorkerInputFile(
                name="evidence",
                kind="artifact",
                digest=evidence_digest,
                size_bytes=len(evidence_payload),
                media_type="application/json",
                path=evidence_path.relative_to(tmp_path).as_posix(),
            ),
        ),
        resource_artifacts=(),
    )


def test_metric_alignment_recovers_scale_rejects_outlier_and_resolves_camera_rig(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    monkeypatch.chdir(tmp_path)

    response = _handle(request, outputs)

    assert [item.name for item in response.outputs] == [
        "camera_rig_calibration",
        "metric_alignment_report",
    ]
    assert response.metrics["scale_m_per_source_unit"] == pytest.approx(0.37, abs=1e-12)
    assert response.metrics["training_inlier_ratio"] == pytest.approx(7 / 8)
    assert response.metrics["holdout_rmse_m"] < 1e-12
    rig = json.loads((outputs / "camera-rig-calibration.json").read_text(encoding="utf-8"))
    report = json.loads((outputs / "metric-alignment-report.json").read_text(encoding="utf-8"))
    assert rig["scale_status"] == "resolved"
    assert rig["coordinate_convention"]["translation_unit"] == "m"
    assert rig["components"][0]["coordinate_frame"] == "cabinet2_capture_world"
    assert report["status"] == "PASSED"
    assert report["training"]["inliers"].count(False) == 1
    validated = []
    for requirement, produced in zip(request.operation.outputs, response.outputs):
        path = outputs / produced.path
        _validate_output_contract(
            requirement,
            produced,
            path,
            execution_size_limit=64 * 1024 * 1024,
        )
        validated.append((produced, path))
    _validate_output_set_closure(validated)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda _rig, evidence: evidence.update(unit="mm"), "unit must explicitly"),
        (
            lambda _rig, evidence: evidence.pop("metric_reference"),
            "metric_reference",
        ),
        (
            lambda _rig, evidence: evidence["metric_reference"].pop("marker_family"),
            "marker_family",
        ),
        (
            lambda _rig, evidence: evidence["holdout_correspondences"].clear(),
            "at least 3",
        ),
        (
            lambda _rig, evidence: evidence["holdout_correspondences"][0].update(
                id=evidence["train_correspondences"][0]["id"]
            ),
            "overlap",
        ),
        (
            lambda _rig, evidence: evidence["holdout_correspondences"][0].update(
                source_xyz=evidence["train_correspondences"][0]["source_xyz"]
            ),
            "coordinates overlap",
        ),
        (
            lambda _rig, evidence: evidence.update(camera_rig_digest="intentionally-wrong"),
            "not bound",
        ),
    ],
)
def test_metric_alignment_rejects_untraceable_or_leaking_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutate: Any,
    message: str,
) -> None:
    request = _request(tmp_path, mutate=mutate)
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match=message):
        _handle(request, output)

    assert list(output.iterdir()) == []


def test_metric_alignment_fails_closed_on_holdout_disagreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def corrupt_holdout(_rig: dict[str, Any], evidence: dict[str, Any]) -> None:
        for item in evidence["holdout_correspondences"]:
            item["target_xyz_m"][0] += 0.08

    request = _request(tmp_path, mutate=corrupt_holdout)
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="holdout RMSE gate"):
        _handle(request, output)

    assert list(output.iterdir()) == []


def test_metric_alignment_output_closure_rejects_cross_artifact_rebinding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)
    response = _handle(request, output)
    rig_path = output / "camera-rig-calibration.json"
    rig = json.loads(rig_path.read_text(encoding="utf-8"))
    rig["source_camera_rig"]["digest"] = f"sha256:{'0' * 64}"
    rig_path.write_text(json.dumps(rig), encoding="utf-8")
    produced = {item.name: item for item in response.outputs}

    with pytest.raises(MethodExecutionError, match="same inputs and transform"):
        _validate_output_set_closure(
            [
                (produced["camera_rig_calibration"], rig_path),
                (
                    produced["metric_alignment_report"],
                    output / "metric-alignment-report.json",
                ),
            ]
        )


def test_metric_alignment_rejects_degenerate_correspondence_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def make_collinear(_rig: dict[str, Any], evidence: dict[str, Any]) -> None:
        for index, item in enumerate(evidence["train_correspondences"]):
            item["source_xyz"] = [float(index), 0.0, 0.0]

    request = _request(tmp_path, mutate=make_collinear)
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="degenerate or collinear"):
        _handle(request, output)

    assert list(output.iterdir()) == []
