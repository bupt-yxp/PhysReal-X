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

import physrealx_workers.ditto_worker as ditto_worker  # noqa: E402

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


def _reference(payload: bytes, media_type: str) -> ArtifactRef:
    return ArtifactRef(
        digest=sha256_digest(payload),
        size_bytes=len(payload),
        media_type=media_type,
    )


def _npy_payload(value: np.ndarray[Any, Any]) -> bytes:
    import io

    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return stream.getvalue()


def _cube(offset: float) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
    vertices = np.asarray(
        [[offset + x, y, z] for x in (0.0, 0.08) for y in (0.0, 0.07) for z in (0.0, 0.06)],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
            [0, 1, 3],
            [0, 3, 2],
            [4, 6, 7],
            [4, 7, 5],
            [0, 4, 5],
            [0, 5, 1],
            [2, 3, 7],
            [2, 7, 6],
            [0, 2, 6],
            [0, 6, 4],
            [1, 5, 7],
            [1, 7, 3],
        ],
        dtype=np.int64,
    )
    return vertices, faces


def _prediction(**updates: Any) -> ditto_worker._Prediction:
    static_vertices, static_faces = _cube(-0.12)
    moving_vertices, moving_faces = _cube(0.04)
    value: dict[str, Any] = {
        "joint_type": "prismatic",
        "prismatic_probability": 0.93,
        "axis": np.asarray([1.0, 0.0, 0.0]),
        "pivot_m": np.asarray([0.04, 0.035, 0.03]),
        "motion": 0.04,
        "axis_candidates": np.tile(np.asarray([[1.0, 0.002, 0.0]]), (128, 1)),
        "static_vertices_m": static_vertices,
        "static_faces": static_faces,
        "moving_vertices_m": moving_vertices,
        "moving_faces": moving_faces,
        "normalization_center_m": np.zeros(3),
        "normalization_scale_m": 0.35,
    }
    value.update(updates)
    return ditto_worker._Prediction(**value)


def _request(
    tmp_path: Path,
    *,
    source_kind: str = "measured_interaction",
    alignment_status: str = "verified",
    revision: str = ditto_worker._UPSTREAM_REVISION,
    moving_displacement_m: float = 0.04,
    static_displacement_m: float = 0.0,
    revolute_angle_rad: float | None = None,
    include_sim_kinematics: bool = False,
    config: dict[str, Any] | None = None,
) -> WorkerRequest:
    rng = np.random.default_rng(42)
    before = rng.uniform([-0.2, -0.15, 0.0], [0.2, 0.15, 0.3], size=(1_200, 3)).astype(np.float32)
    after = before.copy()
    if revolute_angle_rad is None:
        after[:600, 0] += moving_displacement_m
    else:
        cosine = np.cos(revolute_angle_rad)
        sine = np.sin(revolute_angle_rad)
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
        after[:600, :2] = before[:600, :2] @ rotation.T
    before_object_mask = np.ones(len(before), dtype=np.bool_)
    after_object_mask = np.ones(len(after), dtype=np.uint8)

    holdout_before = rng.uniform([-0.15, -0.1, 0.02], [0.15, 0.1, 0.25], size=(160, 3)).astype(
        np.float32
    )
    holdout_mask = np.zeros(160, dtype=np.uint8)
    holdout_mask[:80] = 1
    holdout_after = holdout_before.copy()
    if revolute_angle_rad is None:
        holdout_after[:80, 0] += moving_displacement_m
    else:
        holdout_after[:80, :2] = holdout_before[:80, :2] @ rotation.T
    holdout_after[80:, 1] += static_displacement_m

    inputs = tmp_path / "inputs"
    dependencies = inputs / "000-pair.artifacts"
    dependencies.mkdir(parents=True)
    members: list[tuple[str, ArtifactRef, bytes]] = []

    def add(pointer: str, payload: bytes, media_type: str) -> dict[str, Any]:
        reference = _reference(payload, media_type)
        members.append((pointer, reference, payload))
        return reference.model_dump(mode="json")

    before_points = add("/before/points", _npy_payload(before), "application/x-npy")
    before_mask = add("/before/object_mask", _npy_payload(before_object_mask), "application/x-npy")
    after_points = add("/after/points", _npy_payload(after), "application/x-npy")
    after_mask = add("/after/object_mask", _npy_payload(after_object_mask), "application/x-npy")
    holdout_before_ref = add(
        "/interaction/holdout_correspondences/before_points",
        _npy_payload(holdout_before),
        "application/x-npy",
    )
    holdout_after_ref = add(
        "/interaction/holdout_correspondences/after_points",
        _npy_payload(holdout_after),
        "application/x-npy",
    )
    holdout_mask_ref = add(
        "/interaction/holdout_correspondences/moving_mask",
        _npy_payload(holdout_mask),
        "application/x-npy",
    )
    alignment_evidence = add("/alignment/evidence", b'{"fixture":"alignment"}', "application/json")
    calibration_evidence = add(
        "/cameras/0/calibration_evidence", b'{"fixture":"camera"}', "application/json"
    )
    interaction_evidence = add(
        "/interaction/evidence", b'{"fixture":"interaction"}', "application/json"
    )
    checkpoint = add(
        "/checkpoint/artifact", b"official-ditto-checkpoint-fixture", "application/x-pytorch"
    )
    manifest = {
        "schema": "physrealx.paired-interaction-point-cloud-bundle/1",
        "resource_id": "cabinet2.measured.interaction.001",
        "version": "1.0.0",
        "source_kind": source_kind,
        "coordinate_frame": {
            "frame_id": "cabinet2/object",
            "units": "m",
            "handedness": "right",
            "up_axis": "Z",
        },
        "alignment": {
            "status": alignment_status,
            "convention": "points_expressed_in_same_object_frame",
            "method": "fiducial_bundle_adjustment",
            "evidence": alignment_evidence,
        },
        "before": {
            "points": before_points,
            "object_mask": before_mask,
            "capture_id": "cabinet2-before-001",
        },
        "after": {
            "points": after_points,
            "object_mask": after_mask,
            "capture_id": "cabinet2-after-001",
        },
        "cameras": [
            {
                "camera_id": "camera-0",
                "intrinsics": [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
                "object_from_camera": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "calibration_evidence": calibration_evidence,
            }
        ],
        "interaction": {
            "kind": "robot_manipulation",
            "timestamp_order": "before_then_after",
            "evidence": interaction_evidence,
            "holdout_correspondences": {
                "before_points": holdout_before_ref,
                "after_points": holdout_after_ref,
                "moving_mask": holdout_mask_ref,
                "source": "tracked_rgbd_features",
            },
        },
        "checkpoint": {
            "schema": "physrealx.ditto-checkpoint-manifest/1",
            "artifact": checkpoint,
            "upstream_repository": "https://github.com/UT-Austin-RPL/Ditto",
            "upstream_revision": revision,
            "experiment": "Ditto_s2m",
            "license_spdx": "MIT",
            "architecture": {
                "encoder": "PointNet++",
                "geometry_backbone": "ConvONets",
                "joint_decoder": "dense_screw_parameter_voting",
            },
        },
    }
    if include_sim_kinematics:
        manifest["kinematics"] = {
            "source": "cabinet2/sim/kinematics.json",
            "joint_axis": [1.0, 0.0, 0.0],
        }
    for index, (_, _, payload) in enumerate(members):
        (dependencies / f"{index:06d}.bin").write_bytes(payload)
    manifest_payload = canonical_json_bytes(manifest)
    manifest_path = inputs / "000-pair"
    manifest_path.write_bytes(manifest_payload)
    manifest_digest = sha256_digest(manifest_payload)
    descriptor = MethodDescriptor(
        capability=MethodCapability.ARTICULATION_RECOVER,
        name="ditto",
        version="1.0.0",
        api_version="1",
        source_uri="https://github.com/UT-Austin-RPL/Ditto",
        source_revision=ditto_worker._UPSTREAM_REVISION,
        license_spdx="MIT",
        runtime=MethodRuntime(
            mode=MethodExecutionMode.PROCESS,
            accelerator="cuda",
            python=">=3.8,<3.9",
            environment_digest=f"sha256:{'d' * 64}",
        ),
        input_contract="PairedInteractionPointCloudBundle/1",
        output_contract="ArticulationModel/1+ArticulationRecoveryReport/1",
    )
    operation = OperationSpec(
        resource_id="operation.articulation.ditto.fixture",
        version="1.0.0",
        method=descriptor.bind(config=config or {}),
        inputs=(
            OperationInput(
                name="pair",
                resource=VersionedRef(
                    resource_type=ResourceType.OBSERVATION_SET,
                    resource_id="cabinet2.measured.interaction.001",
                    version="1.0.0",
                    digest=manifest_digest,
                ),
            ),
        ),
        outputs=(
            OperationOutputRequirement(name="articulation_model", contract="ArticulationModel/1"),
            OperationOutputRequirement(
                name="articulation_recovery_report",
                contract="ArticulationRecoveryReport/1",
            ),
        ),
    )
    resource_artifacts = tuple(
        WorkerResourceArtifact(
            owner="pair",
            json_pointer=pointer,
            digest=reference.digest,
            size_bytes=reference.size_bytes,
            media_type=reference.media_type,
            path=(dependencies / f"{index:06d}.bin").relative_to(tmp_path).as_posix(),
        )
        for index, (pointer, reference, _) in enumerate(members)
    )
    return WorkerRequest(
        operation=operation,
        inputs=(
            WorkerInputFile(
                name="pair",
                kind="resource",
                digest=manifest_digest,
                size_bytes=len(manifest_payload),
                media_type="application/vnd.physrealx.paired-interaction+json",
                path=manifest_path.relative_to(tmp_path).as_posix(),
                resource_type=ResourceType.OBSERVATION_SET,
            ),
        ),
        resource_artifacts=resource_artifacts,
    )


def _execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: WorkerRequest,
    prediction: ditto_worker._Prediction | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ditto_worker,
        "_run_official_ditto",
        lambda evidence, options: prediction or _prediction(),
    )
    response = ditto_worker._handle(request, output)
    model = json.loads((output / "articulation_model.json").read_text(encoding="utf-8"))
    report = json.loads((output / "articulation_recovery_report.json").read_text(encoding="utf-8"))
    return model, report, dict(response.metrics)


def test_ditto_publishes_only_holdout_validated_articulation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, report, metrics = _execute(tmp_path, monkeypatch, _request(tmp_path))

    assert model["status"] == "IDENTIFIED"
    assert model["upstream"]["revision"] == ditto_worker._UPSTREAM_REVISION
    assert model["joints"][0]["type"] == "prismatic"
    assert model["joints"][0]["axis"] == pytest.approx([1.0, 0.0, 0.0])
    assert model["joints"][0]["limits"]["status"] == "unresolved"
    assert "kinematics" not in json.dumps(model).lower()
    assert report["gates"]["held_out_screw_transform"] == "passed"
    assert metrics["holdout_rmse_m"] < 1.0e-7
    assert metrics["holdout_improvement_fraction"] > 0.99


def test_ditto_validates_revolute_screw_transform(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    angle = 0.28
    model, report, metrics = _execute(
        tmp_path,
        monkeypatch,
        _request(tmp_path, revolute_angle_rad=angle),
        _prediction(
            joint_type="revolute",
            prismatic_probability=0.04,
            axis=np.asarray([0.0, 0.0, 1.0]),
            pivot_m=np.zeros(3),
            motion=angle,
            axis_candidates=np.tile(np.asarray([[0.001, 0.0, 1.0]]), (128, 1)),
        ),
    )

    assert model["joints"][0]["type"] == "revolute"
    assert model["joints"][0]["observed_displacement"]["unit"] == "rad"
    assert report["gates"]["held_out_screw_transform"] == "passed"
    assert metrics["holdout_rmse_m"] < 1.0e-7


@pytest.mark.parametrize(
    ("request_kwargs", "message"),
    [
        ({"source_kind": "simulation"}, "never simulation"),
        ({"include_sim_kinematics": True}, "unsupported fields: kinematics"),
        ({"alignment_status": "assumed"}, "independently verified"),
        ({"revision": "0" * 40}, "pinned"),
        ({"moving_displacement_m": 0.0001}, "excitation gate"),
        ({"static_displacement_m": 0.03}, "contradict"),
    ],
)
def test_ditto_rejects_invalid_or_simulation_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request_kwargs: dict[str, Any],
    message: str,
) -> None:
    request = _request(tmp_path, **request_kwargs)
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        ditto_worker,
        "_run_official_ditto",
        lambda evidence, options: pytest.fail("inference must not run for rejected evidence"),
    )

    with pytest.raises(ValueError, match=message):
        ditto_worker._handle(request, output)


def test_ditto_rejects_ambiguous_joint_type(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="joint-type prediction is ambiguous"):
        _execute(
            tmp_path,
            monkeypatch,
            _request(tmp_path),
            _prediction(prismatic_probability=0.53),
        )


def test_ditto_rejects_ambiguous_axis_votes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(9)
    axes = rng.normal(size=(128, 3))
    with pytest.raises(ValueError, match="axis votes are ambiguous"):
        _execute(
            tmp_path,
            monkeypatch,
            _request(tmp_path),
            _prediction(axis_candidates=axes),
        )


def test_ditto_rejects_screw_transform_that_fails_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="fails held-out points"):
        _execute(
            tmp_path,
            monkeypatch,
            _request(tmp_path),
            _prediction(motion=0.12),
        )


def test_ditto_rejects_degenerate_part_mesh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vertices = np.zeros((40, 3), dtype=np.float64)
    faces = np.tile(np.asarray([[0, 1, 2]], dtype=np.int64), (40, 1))
    with pytest.raises(ValueError, match="geometrically degenerate"):
        _execute(
            tmp_path,
            monkeypatch,
            _request(tmp_path),
            _prediction(moving_vertices_m=vertices, moving_faces=faces),
        )


def test_probe_fails_closed_without_attested_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("PHYSREALX_DITTO_PYTHON", raising=False)
    monkeypatch.delenv("PHYSREALX_DITTO_REPOSITORY", raising=False)
    monkeypatch.delenv("PHYSREALX_DITTO_CHECKPOINT", raising=False)
    monkeypatch.delenv("PHYSREALX_DITTO_CHECKPOINT_DIGEST", raising=False)

    assert ditto_worker._probe() == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "physrealx.worker-probe/1"
    assert payload["name"] == "ditto"
    assert payload["ready"] is False
    assert "PHYSREALX_DITTO_PYTHON" in payload["reason"]


def test_protocol_worker_invokes_isolated_python38_bridge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    monkeypatch.chdir(tmp_path)
    evidence = ditto_worker._validate_manifest(request, ditto_worker._options({}))
    repository = tmp_path / "official-ditto"
    repository.mkdir()
    python = Path(sys.executable).resolve(strict=True)
    monkeypatch.setenv("PHYSREALX_DITTO_PYTHON", str(python))
    monkeypatch.setenv("PHYSREALX_DITTO_REPOSITORY", str(repository))
    monkeypatch.setenv("PHYSREALX_DITTO_CHECKPOINT", str(evidence.checkpoint_path))
    monkeypatch.setenv("PHYSREALX_DITTO_CHECKPOINT_DIGEST", evidence.checkpoint_digest)
    monkeypatch.setenv("PHYSREALX_DITTO_EXPERIMENT", "Ditto_s2m")

    def bridge_fixture(command: list[str], **kwargs: Any) -> Any:
        assert Path(command[0]) == python
        assert Path(command[1]).name == "ditto_inference_bridge.py"
        assert command[2] == "--request"
        bridge_request = json.loads(Path(command[3]).read_text(encoding="utf-8"))
        prediction = _prediction()
        np.savez_compressed(
            bridge_request["output_npz"],
            joint_type=np.asarray([1], dtype=np.int8),
            prismatic_probability=np.asarray([prediction.prismatic_probability], dtype=np.float64),
            axis=prediction.axis,
            pivot_m=prediction.pivot_m,
            motion=np.asarray([prediction.motion], dtype=np.float64),
            axis_candidates=prediction.axis_candidates,
            static_vertices_m=prediction.static_vertices_m,
            static_faces=prediction.static_faces,
            moving_vertices_m=prediction.moving_vertices_m,
            moving_faces=prediction.moving_faces,
            normalization_center_m=prediction.normalization_center_m,
            normalization_scale_m=np.asarray([prediction.normalization_scale_m], dtype=np.float64),
        )
        return ditto_worker.subprocess.CompletedProcess(command, 0, '{"ok": true}\n', "")

    monkeypatch.setattr(ditto_worker.subprocess, "run", bridge_fixture)

    result = ditto_worker._run_official_ditto(evidence, ditto_worker._options({}))

    assert result.joint_type == "prismatic"
    assert result.motion == pytest.approx(0.04)
    assert result.axis == pytest.approx([1.0, 0.0, 0.0])


def test_runtime_rejects_request_checkpoint_not_bound_at_installation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(tmp_path)
    monkeypatch.chdir(tmp_path)
    evidence = ditto_worker._validate_manifest(request, ditto_worker._options({}))
    repository = tmp_path / "official-ditto"
    repository.mkdir()
    monkeypatch.setenv("PHYSREALX_DITTO_PYTHON", str(Path(sys.executable).resolve()))
    monkeypatch.setenv("PHYSREALX_DITTO_REPOSITORY", str(repository))
    monkeypatch.setenv("PHYSREALX_DITTO_CHECKPOINT", str(evidence.checkpoint_path))
    monkeypatch.setenv("PHYSREALX_DITTO_CHECKPOINT_DIGEST", "sha256:" + "0" * 64)
    monkeypatch.setenv("PHYSREALX_DITTO_EXPERIMENT", "Ditto_s2m")

    with pytest.raises(RuntimeError, match="does not match the checkpoint bound"):
        ditto_worker._run_official_ditto(evidence, ditto_worker._options({}))
