from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parents[2]
PROJECT_SOURCE = PROJECT_ROOT / "src"
WORKERS_SOURCE = PROJECT_ROOT / "workers" / "src"
sys.path.insert(0, str(PROJECT_SOURCE))
sys.path.insert(0, str(WORKERS_SOURCE))

from physrealx_workers.omnidc_worker import (  # noqa: E402
    _handle,
    _load_runtime,
    _probe,
    _source_tree_digest,
)

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
    _validate_output_contract,
)


def _npy(array: np.ndarray[Any, Any]) -> bytes:
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def _reference(payload: bytes, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        digest=sha256_digest(payload),
        size_bytes=len(payload),
        media_type=media_type,
    )


def _truth(height: int = 16, width: int = 20) -> np.ndarray[Any, Any]:
    y, x = np.mgrid[:height, :width]
    return np.asarray(1.0 + 0.01 * y + 0.005 * x, dtype=np.float32)


def _request(
    tmp_path: Path,
    *,
    overlap: bool = False,
    unresolved_scale: bool = False,
    checkpoint_mismatch: bool = False,
    config: dict[str, Any] | None = None,
) -> WorkerRequest:
    inputs = tmp_path / "inputs"
    dependencies = inputs / "000-rgbd.artifacts"
    dependencies.mkdir(parents=True)
    height, width = 16, 20
    y, x = np.mgrid[:height, :width]
    rgb = np.stack((x * 7, y * 11, (x + y) * 5), axis=-1).astype(np.uint8)
    truth = _truth(height, width)
    sparse_mask = (x + 2 * y) % 7 == 0
    holdout_mask = (3 * x + y) % 5 == 0
    if not overlap:
        holdout_mask &= ~sparse_mask
    sparse = np.where(sparse_mask, truth, 0.0).astype(np.float32)
    holdout = np.where(holdout_mask, truth, 0.0).astype(np.float32)
    checkpoint_payload = b"official-omnidc-v1.0-checkpoint-fixture"
    manifest_checkpoint_payload = (
        b"different-official-checkpoint" if checkpoint_mismatch else checkpoint_payload
    )
    items = [
        (_npy(rgb), "application/x-npy"),
        (_npy(sparse), "application/x-npy"),
        (_npy(holdout), "application/x-npy"),
        (manifest_checkpoint_payload, "application/x-pytorch"),
        (b"immutable-source-capture-manifest", "application/octet-stream"),
        (b"immutable-metric-camera-calibration", "application/octet-stream"),
    ]
    references = [_reference(payload, media_type) for payload, media_type in items]
    manifest: dict[str, Any] = {
        "schema": "physrealx.calibrated-rgbd-bundle/1",
        "resource_id": "cabinet2.rgbd",
        "version": "1.0.0",
        "calibration": [
            {
                "camera_id": "camera-0",
                "image_width": width,
                "image_height": height,
                "intrinsic_matrix": [
                    [120.0, 0.0, 9.5],
                    [0.0, 120.0, 7.5],
                    [0.0, 0.0, 1.0],
                ],
                "distortion": {"model": "none", "coefficients": []},
                "sensor_to_world": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
        ],
        "depth_encoding": {
            "unit": "m",
            "meters_per_unit": 0.001 if unresolved_scale else 1.0,
            "invalid_value": 0.0,
            "valid_min_m": 0.1,
            "valid_max_m": 5.0,
        },
        "frames": [
            {
                "frame_id": "frame-0000",
                "camera_id": "camera-0",
                "rgb": references[0].model_dump(mode="json"),
                "sparse_depth": references[1].model_dump(mode="json"),
                "holdout_depth": references[2].model_dump(mode="json"),
            }
        ],
        "checkpoint_manifest": {
            "method": "omni_dc",
            "release": "v1.0",
            "source_uri": "https://github.com/princeton-vl/OMNI-DC",
            "source_revision": "a" * 40,
            "artifact": references[3].model_dump(mode="json"),
        },
        "provenance": {
            "source_capture": references[4].model_dump(mode="json"),
            "calibration_evidence": references[5].model_dump(mode="json"),
        },
    }
    manifest_payload = canonical_json_bytes(manifest)
    manifest_path = inputs / "000-rgbd"
    manifest_path.write_bytes(manifest_payload)
    manifest_digest = sha256_digest(manifest_payload)
    resource_artifacts: list[WorkerResourceArtifact] = []
    pointers = [
        "/frames/0/rgb",
        "/frames/0/sparse_depth",
        "/frames/0/holdout_depth",
        "/checkpoint_manifest/artifact",
        "/provenance/source_capture",
        "/provenance/calibration_evidence",
    ]
    for index, ((payload, media_type), reference, pointer) in enumerate(
        zip(items, references, pointers)
    ):
        destination = dependencies / f"{index:06d}.bin"
        destination.write_bytes(payload)
        resource_artifacts.append(
            WorkerResourceArtifact(
                owner="rgbd",
                json_pointer=pointer,
                digest=reference.digest,
                size_bytes=reference.size_bytes,
                media_type=media_type,
                path=destination.relative_to(tmp_path).as_posix(),
            )
        )
    checkpoint_path = inputs / "checkpoint"
    checkpoint_path.write_bytes(checkpoint_payload)
    checkpoint_ref = _reference(checkpoint_payload, "application/x-pytorch")
    descriptor = MethodDescriptor(
        capability=MethodCapability.DEPTH_ENHANCE,
        name="omni_dc",
        version="1.0.0",
        api_version="1",
        source_uri="https://github.com/princeton-vl/OMNI-DC",
        source_revision="a" * 40,
        license_spdx="BSD-3-Clause",
        runtime=MethodRuntime(
            mode=MethodExecutionMode.PROCESS,
            accelerator="cuda",
            python=">=3.8,<3.9",
            environment_digest=f"sha256:{'e' * 64}",
            minimum_vram_gb=11.0,
        ),
        input_contract="CalibratedRGBDBundle/1",
        output_contract="EnhancedDepthBundle/1+DepthEnhancementReport/1",
    )
    operation = OperationSpec(
        resource_id="operation.depth.enhance.fixture",
        version="1.0.0",
        method=descriptor.bind(
            checkpoint=checkpoint_ref,
            config={
                "minimum_holdout_points": 20,
                "maximum_holdout_rmse_m": 0.02,
                "maximum_holdout_abs_rel": 0.02,
                "minimum_output_coverage": 1.0,
                "minimum_coverage_gain": 0.5,
                **(config or {}),
            },
        ),
        inputs=(
            OperationInput(
                name="rgbd",
                resource=VersionedRef(
                    resource_type=ResourceType.DATASET,
                    resource_id="cabinet2.rgbd",
                    version="1.0.0",
                    digest=manifest_digest,
                ),
            ),
        ),
        outputs=(
            OperationOutputRequirement(name="enhanced_depth", contract="EnhancedDepthBundle/1"),
            OperationOutputRequirement(
                name="enhancement_report", contract="DepthEnhancementReport/1"
            ),
        ),
    )
    return WorkerRequest(
        operation=operation,
        inputs=(
            WorkerInputFile(
                name="rgbd",
                kind="resource",
                digest=manifest_digest,
                size_bytes=len(manifest_payload),
                media_type="application/vnd.physrealx.dataset+json",
                path=manifest_path.relative_to(tmp_path).as_posix(),
                resource_type=ResourceType.DATASET,
            ),
        ),
        resource_artifacts=tuple(resource_artifacts),
        checkpoint=WorkerInputFile(
            name="checkpoint",
            kind="checkpoint",
            digest=checkpoint_ref.digest,
            size_bytes=checkpoint_ref.size_bytes,
            media_type=checkpoint_ref.media_type,
            path=checkpoint_path.relative_to(tmp_path).as_posix(),
        ),
    )


def _predict_truth(
    frames: Any,
    checkpoint: Path,
    release: str,
    maximum_depth_m: float,
) -> list[np.ndarray[Any, Any]]:
    assert checkpoint.is_file()
    assert release == "v1.0"
    assert maximum_depth_m == 5.0
    return [_truth(frame.camera.height, frame.camera.width) for frame in frames]


def test_worker_emits_self_contained_metric_bundle_and_gate_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)

    response = _handle(request, output, predictor=_predict_truth)

    assert [item.contract for item in response.outputs] == [
        "EnhancedDepthBundle/1",
        "DepthEnhancementReport/1",
    ]
    assert response.metrics["output_coverage"] == 1.0
    assert response.metrics["holdout_rmse_m"] == pytest.approx(0.0, abs=1e-7)
    for requirement, produced in zip(request.operation.outputs, response.outputs):
        _validate_output_contract(
            requirement,
            produced,
            output / produced.path,
            execution_size_limit=1024 * 1024 * 1024,
        )
    report = json.loads((output / "depth_enhancement_report.json").read_text())
    assert report["status"] == "PASSED"
    assert set(report["gates"].values()) == {"PASSED"}
    with zipfile.ZipFile(output / "enhanced_depth_bundle.zip") as archive:
        assert archive.namelist() == ["manifest.json", "depth/frame-0000.npy"]
        manifest = json.loads(archive.read("manifest.json"))
        depth_payload = archive.read("depth/frame-0000.npy")
    assert manifest["schema"] == "physrealx.enhanced-depth-bundle/1"
    assert manifest["status"] == "PASSED"
    assert request.checkpoint is not None
    assert manifest["checkpoint_digest"] == request.checkpoint.digest
    assert manifest["frames"][0]["depth"]["digest"] == sha256_digest(depth_payload)
    dense = np.load(io.BytesIO(depth_payload), allow_pickle=False)
    assert dense.dtype == np.float32
    assert np.isfinite(dense).all()
    assert (dense > 0.0).all()


@pytest.mark.parametrize(
    ("request_options", "message"),
    [
        ({"overlap": True}, "must be disjoint"),
        ({"unresolved_scale": True}, "resolved to metres"),
        ({"checkpoint_mismatch": True}, "does not match"),
    ],
)
def test_worker_rejects_invalid_evidence_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request_options: dict[str, Any],
    message: str,
) -> None:
    request = _request(tmp_path, **request_options)
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match=message):
        _handle(request, output, predictor=_predict_truth)

    assert not tuple(output.iterdir())


def test_worker_rejects_nonfinite_prediction_without_publishing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)

    def invalid_predictor(*_: Any) -> list[np.ndarray[Any, Any]]:
        prediction = _truth()
        prediction[0, 0] = np.nan
        return [prediction]

    with pytest.raises(ValueError, match="NaN or infinity"):
        _handle(request, output, predictor=invalid_predictor)

    assert not tuple(output.iterdir())


def test_worker_rejects_model_that_fails_held_out_consistency(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)

    def inaccurate_predictor(*_: Any) -> list[np.ndarray[Any, Any]]:
        return [np.full((16, 20), 4.0, dtype=np.float32)]

    with pytest.raises(ValueError, match="holdout RMSE"):
        _handle(request, output, predictor=inaccurate_predictor)

    assert not tuple(output.iterdir())


def test_probe_reports_not_ready_when_installation_is_missing(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    status = _probe(tmp_path / "missing-installation.json")

    assert status == 2
    record = json.loads(capsys.readouterr().out)
    assert record["schema"] == "physrealx.worker-probe/1"
    assert record["name"] == "omni_dc"
    assert record["ready"] is False
    assert record["protocol_version"] == "2"
    assert "unavailable" in record["reason"]


def test_installation_rejects_mutated_official_dependency_source(tmp_path: Path) -> None:
    source = tmp_path / "OMNI-DC"
    config_path = source / "src/config.py"
    model_path = source / "src/model/ognidc.py"
    license_path = source / "LICENSE"
    config_path.parent.mkdir(parents=True)
    model_path.parent.mkdir(parents=True)
    config_path.write_text("args = object()\n", encoding="utf-8")
    model_path.write_text("class OGNIDC: pass\n", encoding="utf-8")
    license_path.write_text("BSD-3-Clause fixture\n", encoding="utf-8")
    checkpoint = tmp_path / "modelv1.0.pt"
    checkpoint.write_bytes(b"checkpoint")
    rgb = tmp_path / "probe.png"
    rgb.write_bytes(b"probe-rgb")
    depth = tmp_path / "probe.npy"
    depth.write_bytes(b"probe-depth")
    runtime_paths = [config_path, model_path, license_path]
    manifest = {
        "schema": "physrealx.omnidc-installation/1",
        "source_uri": "https://github.com/princeton-vl/OMNI-DC",
        "source_revision": "d" * 40,
        "source_root": str(source),
        "source_tree_digest": _source_tree_digest(source),
        "runtime_files": [
            {
                "path": path.relative_to(source).as_posix(),
                "digest": sha256_digest(path.read_bytes()),
            }
            for path in runtime_paths
        ],
        "checkpoint": {
            "path": str(checkpoint),
            "digest": sha256_digest(checkpoint.read_bytes()),
            "release": "v1.0",
        },
        "self_test": {
            "rgb_path": str(rgb),
            "rgb_digest": sha256_digest(rgb.read_bytes()),
            "depth_path": str(depth),
            "depth_digest": sha256_digest(depth.read_bytes()),
        },
    }
    manifest_path = tmp_path / "installation.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert _load_runtime(manifest_path).source_revision == "d" * 40
    model_path.write_text("class OGNIDC: hacked = True\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="source tree digest changed"):
        _load_runtime(manifest_path)
