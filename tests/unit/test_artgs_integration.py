from __future__ import annotations

import io
import json
import struct
import zipfile
import zlib
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
    ArtGSInputManifest,
    ArtGSMethodAdapter,
    ArtGSRunConfig,
    _ArtGSRunResult,
    _validate_source_snapshot,
    artgs_source_root,
    package_artgs_input,
)
from physrealx.platform import PhysRealX
from physrealx.storage import LocalArtifactStore


def _transform_document(state: str, *, held_out: bool = False) -> dict[str, object]:
    offset = 2 if held_out else 0
    return {
        "camera_angle_x": 0.9,
        "camera_angle_y": 0.7,
        "frames": [
            {
                "file_path": f"{state}/rgba/{index + offset:03d}.png",
                "time": float(index),
                "transform_matrix": [
                    [1.0, 0.0, 0.0, 0.1 * index],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            }
            for index in range(2)
        ],
    }


def _png(*, depth: bool, value: int) -> bytes:
    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    bit_depth, color_type = (16, 0) if depth else (8, 6)
    pixel = struct.pack(">H", 1_000 + value) if depth else bytes((value, 10, 20, 255))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, bit_depth, color_type, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(b"\x00" + pixel))
        + chunk(b"IEND", b"")
    )


def _prepared_source(root: Path, *, held_out: bool = True) -> Path:
    root.mkdir()
    (root / "transforms_train.json").write_text("{}\n", encoding="utf-8")
    for state in ("start", "end"):
        (root / f"transforms_train_{state}.json").write_text(
            json.dumps(_transform_document(state), sort_keys=True),
            encoding="utf-8",
        )
        if held_out:
            (root / f"transforms_test_{state}.json").write_text(
                json.dumps(_transform_document(state, held_out=True), sort_keys=True),
                encoding="utf-8",
            )
        for index in range(4 if held_out else 2):
            rgba = root / state / "rgba" / f"{index:03d}.png"
            rgba.parent.mkdir(parents=True, exist_ok=True)
            rgba.write_bytes(_png(depth=False, value=index))
    return root


def _depth_bindings(*, held_out: bool = True) -> dict[str, str]:
    return {
        f"{state}/rgba/{index:03d}.png": f"{state}-{index:03d}"
        for state in ("start", "end")
        for index in range(4 if held_out else 2)
    }


def _metric_evidence(
    artifacts: LocalArtifactStore,
    frame_ids: tuple[str, ...],
) -> tuple[ArtifactRef, ArtifactRef]:
    camera = artifacts.put_json(
        {
            "schema": "physrealx.camera-rig-calibration/1",
            "scale_status": "resolved",
            "coordinate_convention": {
                "frame": "cabinet2_object",
                "pose": "world_to_camera",
                "quaternion_order": "wxyz",
                "translation_unit": "m",
            },
            "method": {"name": "metric_alignment", "version": "1.0.0"},
            "components": [{"component_id": "0", "registered_views": [{}]}],
        }
    )
    depth_path = artifacts.root / "enhanced-depth-fixture.zip"
    records = []
    payloads = []
    for index, frame_id in enumerate(frame_ids):
        stream = io.BytesIO()
        np.save(stream, np.full((1, 1), 1.0 + index / 100.0, dtype=np.float32))
        payload = stream.getvalue()
        path = f"depth/{frame_id}.npy"
        payloads.append((path, payload))
        records.append(
            {
                "frame_id": frame_id,
                "depth": {
                    "path": path,
                    "size_bytes": len(payload),
                    "media_type": "application/x-npy",
                    "dtype": "float32",
                    "unit": "m",
                    "invalid_value": None,
                },
            }
        )
    with zipfile.ZipFile(depth_path, "w") as archive:
        archive.writestr(
            "manifest.json",
            json.dumps(
                {
                    "schema": "physrealx.enhanced-depth-bundle/1",
                    "status": "PASSED",
                    "depth_encoding": {"unit": "m"},
                    "frames": records,
                },
                sort_keys=True,
            ),
        )
        for path, payload in payloads:
            archive.writestr(path, payload)
    depth = artifacts.put_file(depth_path, media_type="application/zip")
    depth_path.unlink()
    return camera, depth


class _FakeRunner:
    def probe(self) -> MethodProbe:
        return MethodProbe(True, "deterministic ArtGS fixture")

    def run(
        self,
        dataset_root: Path,
        output_root: Path,
        manifest: ArtGSInputManifest,
        config: ArtGSRunConfig,
    ) -> _ArtGSRunResult:
        assert dataset_root.name == manifest.scene_id
        model = output_root / "model"
        reconstruction = model / "reconstruction"
        point_cloud = reconstruction / "point_cloud" / "iteration_best" / "point_cloud.ply"
        deformation = reconstruction / "deform" / "iteration_best" / "deform.pth"
        joint_info = reconstruction / "train" / "ours_best" / "joint_info.json"
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


def _operation(source: ArtifactRef, *, allow_training_views: bool = False) -> OperationSpec:
    return OperationSpec(
        resource_id="operation.geometry.artgs.fixture",
        version="1.0.0",
        method=ARTGS_DESCRIPTOR.bind(
            config={"allow_training_view_evaluation": allow_training_views}
        ),
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


def test_input_bundle_is_structurally_bound_to_metric_evidence(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    bindings = _depth_bindings()
    camera, depth = _metric_evidence(artifacts, tuple(bindings.values()))
    source = _prepared_source(tmp_path / "prepared")

    first = package_artgs_input(
        source,
        scene_id="cabinet2",
        camera_rig=camera,
        enhanced_depth=depth,
        depth_frame_ids=bindings,
        num_slots=2,
        artifacts=artifacts,
    )
    second = package_artgs_input(
        source,
        scene_id="cabinet2",
        camera_rig=camera,
        enhanced_depth=depth,
        depth_frame_ids=bindings,
        num_slots=2,
        artifacts=artifacts,
    )

    first_archive = tmp_path / "first.zip"
    second_archive = tmp_path / "second.zip"
    artifacts.materialize(first, first_archive)
    artifacts.materialize(second, second_archive)
    assert first_archive.read_bytes() == second_archive.read_bytes()
    with zipfile.ZipFile(first_archive) as bundle:
        manifest = ArtGSInputManifest.model_validate_json(bundle.read("bundle-manifest.json"))
    assert manifest.source_camera == camera
    assert manifest.source_depth == depth
    assert manifest.training_frames == 4
    assert manifest.validation_frames == 4
    assert len(manifest.depth_bindings) == 8
    assert {item.enhanced_frame_id for item in manifest.depth_bindings} == set(bindings.values())


def test_adapter_publishes_metric_visual_geometry_and_held_out_report(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    bindings = _depth_bindings()
    camera, depth = _metric_evidence(artifacts, tuple(bindings.values()))
    source = package_artgs_input(
        _prepared_source(tmp_path / "prepared"),
        scene_id="cabinet2",
        camera_rig=camera,
        enhanced_depth=depth,
        depth_frame_ids=bindings,
        num_slots=2,
        artifacts=artifacts,
    )

    result = ArtGSMethodAdapter(_FakeRunner()).execute(_operation(source), artifacts)

    assert tuple(item.contract for item in result.outputs) == (
        "VisualGeometryBundle/1",
        "GeometryReconstructionReport/1",
    )
    report = json.loads(artifacts.get_bytes(result.outputs[1].artifact))
    assert report["status"] == "PASSED"
    assert report["evaluation_scope"] == "held_out_views"
    assert set(report["gates"].values()) == {True}
    assert result.metrics["mesh_count"] == 2.0


def test_training_view_only_run_requires_explicit_non_promotable_mode(tmp_path: Path) -> None:
    artifacts = LocalArtifactStore(tmp_path / "artifacts")
    bindings = _depth_bindings(held_out=False)
    camera, depth = _metric_evidence(artifacts, tuple(bindings.values()))
    source = package_artgs_input(
        _prepared_source(tmp_path / "prepared", held_out=False),
        scene_id="cabinet2",
        camera_rig=camera,
        enhanced_depth=depth,
        depth_frame_ids=bindings,
        num_slots=2,
        artifacts=artifacts,
    )
    adapter = ArtGSMethodAdapter(_FakeRunner())

    with pytest.raises(DomainValidationError, match="requires held-out views"):
        adapter.execute(_operation(source), artifacts)

    result = adapter.execute(_operation(source, allow_training_views=True), artifacts)
    report = json.loads(artifacts.get_bytes(result.outputs[1].artifact))
    assert report["status"] == "TRAINING_VIEWS_ONLY"
    assert report["gates"]["held_out_evaluation"] is False


def test_default_workspace_registers_embedded_artgs(tmp_path: Path) -> None:
    platform = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)

    descriptor = platform.methods.get(MethodCapability.GEOMETRY_RECONSTRUCT, "artgs")
    assert descriptor == ARTGS_DESCRIPTOR
    assert descriptor.license_spdx == "LicenseRef-ArtGS-Research-Only"
    assert descriptor.source_revision == "7c1f41be2cb8b96a.physrealx1"


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
