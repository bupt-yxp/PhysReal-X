from __future__ import annotations

import importlib.util
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from physrealx.contracts import MethodProbe
from physrealx.domain import (
    ArtifactRef,
    MethodCapability,
    OperationInput,
    OperationOutputRequirement,
    OperationSpec,
)
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations.artgs import (
    ARTGS_DESCRIPTOR,
    ArtGSMethodAdapter,
    ArtGSRunConfig,
    _ArtGSRunResult,
    _validate_source_snapshot,
    artgs_source_root,
)
from physrealx.integrations.artgs_capture import ArtGSPreparedInput, prepare_artgs_capture
from physrealx.integrations.rgbd import MetricRGBDInput, load_rgbd
from physrealx.platform import PhysRealX
from physrealx.storage import LocalArtifactStore


def test_native_rotation_axis_handles_the_required_rank_two_system():
    path = artgs_source_root() / "utils/joint_geometry.py"
    spec = importlib.util.spec_from_file_location("artgs_joint_geometry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    axis = np.array([1.0, 2.0, -1.0])
    axis /= np.linalg.norm(axis)
    point = np.array([0.3, -0.1, 0.2])
    point -= axis * (point @ axis)
    skew = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    for angle in (1e-4, 0.4, np.pi):
        rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
        translation = (np.eye(3) - rotation) @ point
        actual = module.rotation_axis_position(rotation, translation, axis)
        assert actual == pytest.approx(point, abs=1e-8)
        assert rotation @ actual + translation == pytest.approx(actual, abs=1e-8)
    with pytest.raises(ValueError, match="zero rotation"):
        module.rotation_axis_position(np.eye(3), np.zeros(3), axis)
    with pytest.raises(ValueError, match="reflection"):
        module.rotation_axis_position(np.diag([1, 1, -1]), np.zeros(3), [1, 0, 0])


def test_native_configuration_uses_portable_json_and_explicit_overrides(tmp_path, monkeypatch):
    import argparse
    import sys

    path = artgs_source_root() / "arguments/__init__.py"
    spec = importlib.util.spec_from_file_location("artgs_arguments", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configuration = tmp_path / "cfg_args.json"
    configuration.write_text(json.dumps({"num_slots": 2, "slot_size": 128}))
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--num_slots", type=int)
    monkeypatch.setattr(
        sys, "argv", ["render.py", "--model_path", str(tmp_path), "--num_slots", "3"]
    )
    resolved = module.get_combined_args(parser)
    assert resolved.num_slots == 3 and resolved.slot_size == 128
    configuration.unlink()
    (tmp_path / "cfg_args").write_text("raise RuntimeError('old executable configuration')")
    with pytest.raises(FileNotFoundError):
        module.get_combined_args(parser)


def _array(store, array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return store.put_bytes(stream.getvalue(), media_type="application/x-npy").model_dump(
        mode="json"
    )


def _source(store, *, conditioned=False, count=4):
    streams, frames, payloads = [], [], []
    height, width = 24, 32
    y, x = np.mgrid[:height, :width]
    rays = np.dstack(
        ((x - width / 2) / 30, (y - height / 2) / 30, np.ones((height, width)))
    ).astype(np.float32)
    intrinsics = {
        "camera_id": 1,
        "model": "PINHOLE",
        "width": width,
        "height": height,
        "params": [30, 30, width / 2, height / 2],
    }
    for state in ("start", "end"):
        images = []
        for index in range(count):
            identity = f"{state}.{index}"
            rgb = np.full((height, width, 3), 80 + index, dtype=np.uint8)
            rgb_ref = _array(store, rgb)
            pose = [[1, 0, 0, index * 0.01], [0, 1, 0, 0], [0, 0, 1, 0]]
            images.append(
                {
                    "name": identity,
                    "camera_id": 1,
                    "source_artifact_id": rgb_ref["artifact_id"],
                    "cam_from_world": pose,
                }
            )
            stream = io.BytesIO()
            np.save(
                stream, np.full((height, width), 1.0 + index / 10, np.float32), allow_pickle=False
            )
            path = f"depth/{identity}.npy"
            payload = stream.getvalue()
            payloads.append((path, payload))
            frames.append(
                {
                    "frame_id": identity,
                    "rgb": rgb_ref,
                    "depth": {
                        "path": path,
                        "size_bytes": len(payload),
                        "unit": "m",
                        "dtype": "float32",
                        "media_type": "application/x-npy",
                    },
                    "view": {
                        "stream_name": state,
                        "image_name": identity,
                        "source_rgb_artifact_id": rgb_ref["artifact_id"],
                        "cam_from_world": pose,
                        "rays": _array(store, rays),
                        "object_mask": _array(store, np.ones((height, width), np.uint8)),
                    },
                }
            )
        streams.append(
            {"stream_name": state, "components": [{"cameras": [intrinsics], "images": images}]}
        )
    camera = store.put_json(
        {
            "schema": "physrealx.camera-rig-calibration/1",
            "scale_status": "resolved",
            "coordinate_frame": "object",
            "coordinate_convention": {"translation_unit": "m"},
            "streams": streams,
            "metric_evidence_scope": "supplied_usd_height_prior"
            if conditioned
            else "independent_metric_alignment",
        }
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema": "physrealx.enhanced-depth-bundle/1",
                    "source_camera_artifact_id": camera.artifact_id,
                    "depth_encoding": {"unit": "m"},
                    "frames": frames,
                    "status": "SELF_SUPERVISED" if conditioned else "PASSED",
                    "evaluation_scope": "masked_observed_pixels"
                    if conditioned
                    else "independent_reference",
                }
            ),
        )
        for name, payload in payloads:
            archive.writestr(name, payload)
    depth = store.put_bytes(buffer.getvalue(), media_type="application/zip")
    reference = store.put_json(
        {"schema": "physrealx.usd-reference/1", "links": [{"name": "root"}, {"name": "drawer"}]}
    )
    source = MetricRGBDInput(camera_rig=camera, enhanced_depth=depth, reference_model=reference)
    return store.put_json(source.model_dump(mode="json", by_alias=True)), source


class _FakeRunner:
    def probe(self) -> MethodProbe:
        return MethodProbe(True, "deterministic ArtGS fixture")

    def run(
        self,
        dataset_root: Path,
        output_root: Path,
        manifest: ArtGSPreparedInput,
        config: ArtGSRunConfig,
    ) -> _ArtGSRunResult:
        assert dataset_root.name == manifest.scene_id
        model = output_root / "model"
        reconstruction = model / "reconstruction"
        point_cloud = reconstruction / "point_cloud" / "iteration_best" / "point_cloud.ply"
        deformation = reconstruction / "deform" / "iteration_best" / "deform.pth"
        joint_info = reconstruction / "train" / "ours_best" / "joint_info.json"
        previews = reconstruction / "test" / "ours_best" / "interpolation"
        previews.mkdir(parents=True)
        (reconstruction / "cfg_args.json").write_text(
            json.dumps({"num_slots": manifest.num_slots, "source_path": str(dataset_root)})
        )
        for index in range(5):
            (previews / f"{index:02d}.png").write_bytes(b"explicit rendering fixture")
        (previews / "states.json").write_text('{"states":[0,0.25,0.5,0.75,1]}')
        for path, payload in (
            (point_cloud, b"ply gaussian fixture"),
            (deformation, b"deformation fixture"),
            (joint_info, b'{"joint_types":["s","p"]}'),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        meshes = []
        for index in range(manifest.num_slots):
            mesh = reconstruction / "train" / "ours_best" / "meshes" / f"start_{index}.ply"
            mesh.parent.mkdir(parents=True, exist_ok=True)
            mesh.write_bytes(f"ply link {index}".encode())
            meshes.append(mesh)
        held_out = manifest.validation_frames > 0
        metrics = {
            "train_psnr": 28.0,
            "train_ssim": 0.94,
            "train_lpips": 0.10,
        }
        if held_out:
            metrics.update(
                {
                    "validation_psnr": 25.0,
                    "validation_ssim": 0.90,
                    "validation_lpips": 0.14,
                }
            )
        return _ArtGSRunResult(
            model_root=model,
            mesh_paths=tuple(meshes),
            metrics=metrics,
            evaluation_scope="held_out_views" if held_out else "training_views",
        )


def _operation(source: ArtifactRef) -> OperationSpec:
    return OperationSpec(
        resource_id="operation.geometry.artgs.fixture",
        version="1.0.0",
        method=ARTGS_DESCRIPTOR.bind(config={"num_slots": 2}),
        inputs=(OperationInput(name="source", artifact=source),),
        outputs=(
            OperationOutputRequirement(
                name="visual_geometry_bundle",
                contract="VisualGeometryBundle/1",
            ),
            OperationOutputRequirement(
                name="geometry_reconstruction_report",
                contract="GeometryReconstructionReport/1",
            ),
        ),
    )


def test_preparation_rectifies_metric_views_and_excludes_holdout_geometry(tmp_path):
    pytest.importorskip("pycolmap")
    cv2 = pytest.importorskip("cv2")
    plyfile = pytest.importorskip("plyfile")
    store = LocalArtifactStore(tmp_path / "artifacts")
    _, source = _source(store)
    prepared = tmp_path / "prepared"
    result = prepare_artgs_capture(
        source, store, prepared, num_slots=2, state_streams=("start", "end")
    )
    assert result.training_frames == 4 and result.validation_frames == 4
    assert set(result.training_frame_ids).isdisjoint(result.validation_frame_ids)
    depths = cv2.imread(str(prepared / "start/depth/000001.png"), cv2.IMREAD_UNCHANGED)
    np.testing.assert_array_equal(depths, np.full((24, 32), 1100, np.uint16))
    points = plyfile.PlyData.read(prepared / "points3d.ply")["vertex"]
    assert set(np.round(points["z"], 1)) == {np.float32(1.1), np.float32(1.2)}
    transform = json.loads((prepared / "transforms_train_start.json").read_text())["frames"][0][
        "transform_matrix"
    ]
    np.testing.assert_allclose(np.asarray(transform)[:3, :3], np.diag([1, -1, -1]))


def test_rgbd_validation_rejects_pose_identity_tampering(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    _, source = _source(store)
    camera = json.loads(store.get_bytes(source.camera_rig))
    camera["streams"][0]["components"][0]["images"][0]["cam_from_world"][0][3] = 9
    changed = source.model_copy(update={"camera_rig": store.put_json(camera)})
    with pytest.raises(DomainValidationError, match="bound to another camera"):
        load_rgbd(changed, store)


def test_adapter_publishes_metric_visual_geometry_and_held_out_report(tmp_path):
    pytest.importorskip("pycolmap")
    pytest.importorskip("cv2")
    store = LocalArtifactStore(tmp_path / "artifacts")
    reference, _ = _source(store)
    result = ArtGSMethodAdapter(_FakeRunner()).execute(_operation(reference), store)
    report = json.loads(store.get_bytes(result.outputs[1].artifact))
    assert report["status"] == "PASSED"
    assert report["evaluation_scope"] == "held_out_views"
    assert report["independent_physical_validation"] is False
    assert result.metrics["mesh_count"] == 2
    with zipfile.ZipFile(io.BytesIO(store.get_bytes(result.outputs[0].artifact))) as archive:
        configuration = json.loads(archive.read("model/config.json"))
        assert configuration["num_slots"] == 2
        assert "source_path" not in configuration and "model_path" not in configuration


def test_conditioned_depth_is_usable_without_promoting_evidence(tmp_path):
    pytest.importorskip("pycolmap")
    pytest.importorskip("cv2")
    store = LocalArtifactStore(tmp_path / "artifacts")
    reference, _ = _source(store, conditioned=True)
    result = ArtGSMethodAdapter(_FakeRunner()).execute(_operation(reference), store)
    report = json.loads(store.get_bytes(result.outputs[1].artifact))
    assert report["status"] == "MODEL_CONDITIONED"
    assert report["depth_evaluation_scope"] == "masked_observed_pixels"


def test_insufficient_holdout_views_are_rejected(tmp_path):
    pytest.importorskip("pycolmap")
    pytest.importorskip("cv2")
    store = LocalArtifactStore(tmp_path / "artifacts")
    _, source = _source(store, count=3)
    with pytest.raises(DomainValidationError, match="held-out views"):
        prepare_artgs_capture(source, store, tmp_path / "prepared", num_slots=2)


def test_default_workspace_registers_embedded_artgs(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)

    descriptor = platform.methods.get(MethodCapability.GEOMETRY_RECONSTRUCT, "artgs")
    assert descriptor == ARTGS_DESCRIPTOR
    assert descriptor.license_spdx == "LicenseRef-ArtGS-Research-Only"
    assert descriptor.source_revision == "7c1f41be2cb8b96a.physrealx3"


def test_embedded_source_snapshot_uses_the_registered_revision(tmp_path: Path) -> None:
    revision = "7c1f41be2cb8b96abca13c6a9668dcf7e06d8c2a"
    assert _validate_source_snapshot(artgs_source_root()) == revision

    source = tmp_path / "artgs"
    source.mkdir()
    (source / "train.py").write_text("print('fixture')\n", encoding="utf-8")
    (source / "SOURCE.toml").write_text(
        "\n".join(
            (
                'schema = "physrealx.vendored-source/1"',
                'revision = "7c1f41be2cb8b96abca13c6a9668dcf7e06d8c2a"',
                "",
            )
        ),
        encoding="utf-8",
    )
    assert _validate_source_snapshot(source) == revision

    (source / "SOURCE.toml").write_text(
        'schema = "physrealx.vendored-source/1"\nrevision = "wrong"\n',
        encoding="utf-8",
    )
    with pytest.raises(MethodExecutionError, match="revision"):
        _validate_source_snapshot(source)
