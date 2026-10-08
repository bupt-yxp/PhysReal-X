"""End-to-end checks for calibrated video articulation evidence."""

import io
import json

import numpy as np
import pytest

try:
    import trimesh
except ModuleNotFoundError as exc:
    if exc.name == "trimesh":
        pytest.skip("Install the physical extra for mesh tests", allow_module_level=True)
    raise
from PIL import Image

from physrealx import PhysRealX
from physrealx.integrations.video_articulation import (
    VideoArticulationEvidence,
    VideoArticulationFrame,
    VideoArticulationRefinement,
    VideoArticulationRefinementConfig,
    observations,
)
from physrealx.reconstruction import joint_transform


def _mesh_artifact(px: PhysRealX, mesh: trimesh.Trimesh):
    return px.artifacts.put_bytes(mesh.export(file_type="glb"), media_type="model/gltf-binary")


def _array_artifact(px: PhysRealX, value: np.ndarray):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    return px.artifacts.put_bytes(stream.getvalue(), media_type="application/x-npy")


def _png_artifact(px: PhysRealX, value: np.ndarray):
    stream = io.BytesIO()
    Image.fromarray(value).save(stream, format="PNG")
    return px.artifacts.put_bytes(stream.getvalue(), media_type="image/png")


@pytest.mark.parametrize("offset, expected", [(0.09, "PASSED"), (0.075, "FAILED")])
def test_calibrated_video_refinement_preserves_range_and_requires_new_collision(
    tmp_path, offset, expected
):
    px = PhysRealX.open(tmp_path)
    body = trimesh.creation.box([0.146, 0.20, 0.02])
    body.apply_translation([-offset, 0, 0])
    cover = trimesh.creation.box([0.146, 0.20, 0.02])
    cover.apply_translation([offset, 0, 0])
    body_ref, cover_ref = _mesh_artifact(px, body), _mesh_artifact(px, cover)
    reference = px.artifacts.put_json(
        {
            "schema": "physrealx.usd-reference/1",
            "unit": "m",
            "coordinate_frame": "reference_z_up",
            "root_link": "body",
            "links": [
                {"name": "body", "geometry": body_ref.model_dump(mode="json")},
                {"name": "cover", "geometry": cover_ref.model_dump(mode="json")},
            ],
            "joints": [
                {
                    "name": "hinge",
                    "parent": "body",
                    "child": "cover",
                    "joint_type": "hinge",
                    "axis": [0, 1, 0],
                    "origin_m": [0, 0, 0],
                    "lower_limit": 0,
                    "upper_limit": 0.7,
                }
            ],
        }
    )
    registration = px.artifacts.put_json(
        {
            "schema": "physrealx.metric-alignment-report/1",
            "source_reference": reference.model_dump(mode="json"),
        }
    )
    motion = {
        "joint": "hinge",
        "parent": "body",
        "child": "cover",
        "kind": "hinge",
        "axis": [0.0, 1.0, 0.0],
        "origin_m": [0.0, 0.0, 0.0],
        "range": [0.0, 0.7],
        "positions": [0.0, 0.7],
        "policy": "module_one_predicted_start_end",
    }
    geometry = px.artifacts.put_json(
        {
            "schema": "physrealx.registered-reconstruction-geometry/4",
            "source_reference": reference.model_dump(mode="json"),
            "source_registration": registration.model_dump(mode="json"),
            "geometries": {
                "body": body_ref.model_dump(mode="json"),
                "cover": cover_ref.model_dump(mode="json"),
            },
            "reconstructed_motion": motion,
        }
    )

    height = width = 256
    fx = fy = 500.0
    cx = cy = 128.0
    camera = np.eye(4)
    camera[:3, :3] = np.diag([1.0, -1.0, -1.0])
    camera[:3, 3] = [0.0, 0.0, 0.7]
    body_points = trimesh.sample.sample_surface(body, 30_000, seed=2)[0]
    cover_points = trimesh.sample.sample_surface(cover, 30_000, seed=3)[0]
    frames = []
    for index, position in enumerate((0.0, 0.22, 0.46, 0.7)):
        camera_points, labels = [], []
        for points, label, transform in (
            (body_points, 1, np.eye(4)),
            (
                cover_points,
                2,
                joint_transform("hinge", [0, 1, 0], [0, 0, 0], position),
            ),
        ):
            values = trimesh.transform_points(trimesh.transform_points(points, transform), camera)
            visible = values[:, 2] > 0
            camera_points.append(values[visible])
            labels.append(np.full(np.count_nonzero(visible), label, dtype=np.uint8))
        points = np.concatenate(camera_points)
        point_labels = np.concatenate(labels)
        x = np.rint(fx * points[:, 0] / points[:, 2] + cx).astype(int)
        y = np.rint(fy * points[:, 1] / points[:, 2] + cy).astype(int)
        inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x, y, z, point_labels = (
            x[inside],
            y[inside],
            points[inside, 2],
            point_labels[inside],
        )
        order = np.argsort(z)[::-1]
        depth = np.zeros((height, width), dtype=np.float32)
        mask = np.zeros((height, width), dtype=np.uint8)
        depth[y[order], x[order]] = z[order]
        mask[y[order], x[order]] = point_labels[order]
        assert min(np.count_nonzero(mask == 1), np.count_nonzero(mask == 2)) > 100
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        rgb[mask == 1], rgb[mask == 2] = [120, 120, 120], [30, 30, 30]
        frames.append(
            {
                "frame_id": str(index),
                "partition": "fit" if index % 2 == 0 else "check",
                "rgb": _png_artifact(px, rgb),
                "depth": _array_artifact(px, depth),
                "link_mask": _png_artifact(px, mask),
                "intrinsics": [fx, fy, cx, cy],
                "camera_from_reference": camera.tolist(),
                "positions": {"hinge": position},
            }
        )

    evidence = VideoArticulationEvidence(
        reference=reference,
        registration=registration,
        source_geometry=geometry,
        capture_id="synthetic-calibrated-video",
        depth_scale_m=1,
        link_ids={"body": 1, "cover": 2},
        frames=frames,
    )
    whole_object = []
    for frame in evidence.frames:
        labels = np.asarray(Image.open(io.BytesIO(px.artifacts.get_bytes(frame.link_mask))))
        values = {
            **frame.model_dump(),
            "frame_id": f"object-{frame.frame_id}",
            "link_mask": None,
            "object_mask": _png_artifact(px, (labels > 0).astype(np.uint8) * 255),
            "depth_scale_m": 1.0,
        }
        whole_object.append(VideoArticulationFrame.model_validate(values))
        with pytest.raises(ValueError, match="exactly one"):
            VideoArticulationFrame.model_validate({**values, "link_mask": frame.link_mask})
    anchored = evidence.model_copy(update={"frames": tuple(whole_object)})
    decoded = observations(
        px.artifacts,
        anchored,
        json.loads(px.artifacts.get_bytes(reference)),
        motion,
        VideoArticulationRefinementConfig(points_per_frame=100),
    )
    assert {p: len(v) for p, v in decoded["__object__"].items()} == {"fit": 2, "check": 2}
    assert all(frame.image is not None for frame in decoded["__object__"]["fit"])
    evidence_ref = px.artifacts.put_json(evidence.model_dump(mode="json", by_alias=True))
    config = {
        "points_per_frame": 1500,
        "solver_iterations": 10,
        "remesh_rounds": 0,
        "surface_query_backend": "cpu",
        "cpu_query_workers": 1,
        "clearance_samples": 9,
        # The near-threshold rejection case must remain inconclusive when its
        # certificate budget is exhausted, rather than spending millions of
        # queries to demonstrate the same publication block in a CPU test.
        "verification_cell_budget": 1000,
    }
    output = px.assets.refine_video_articulation(evidence_ref, config=config)
    result = json.loads(px.artifacts.get_bytes(output))

    from unittest.mock import patch

    with patch(
        "physrealx.integrations.video_articulation.create_surface_query",
        side_effect=AssertionError("cache missed"),
    ):
        assert (
            px.assets.refine_video_articulation(evidence_ref, config=config, cached_result=output)
            == output
        )
        # Restart provenance does not invalidate a completed result. Keep it in
        # the report, but don't make a notebook redo the whole solve to omit it.
        resumed_document = json.loads(px.artifacts.get_bytes(output))
        resumed_document["config"]["resume_checkpoint"] = result["checkpoints"][0]
        resumed_output = px.artifacts.put_json(resumed_document)
        assert (
            px.assets.refine_video_articulation(
                evidence_ref, config=config, cached_result=resumed_output
            )
            == resumed_output
        )
        with pytest.raises(AssertionError, match="cache missed"):
            px.assets.refine_video_articulation(
                evidence_ref,
                config={**config, "resume_checkpoint": result["checkpoints"][-1]},
                cached_result=resumed_output,
            )
        with pytest.raises(AssertionError, match="cache missed"):
            px.assets.refine_video_articulation(
                evidence_ref, config={**config, "seed": 101}, cached_result=output
            )

    assert result["status"] == expected
    assert result["heldout_images"]["frames"]
    assert all(
        row["candidate"]["frame_id"] in {"1", "3"} for row in result["heldout_images"]["frames"]
    )
    altered = {**result, "optimized_motion": {**result["optimized_motion"], "range": [0, 0.5]}}
    with pytest.raises(ValueError, match="preserve source articulation"):
        VideoArticulationRefinement.model_validate(altered)
    if expected == "FAILED":
        assert result["geometries"] is None
        assert result["collision_proxy_status"] == "BLOCKED"
        with pytest.raises(
            ValueError, match=r"certified clearance|independent (image|surface) validation"
        ):
            VideoArticulationRefinement.model_validate(
                {
                    **result,
                    "status": "PASSED",
                    "geometries": result["diagnostic_geometries"],
                    "collision_proxy_status": "REQUIRES_REBUILD",
                }
            )
        return
    assert result["source_motion"] == motion
    assert result["optimized_motion"]["range"] == motion["range"]
    assert result["optimized_motion"]["positions"] == motion["positions"]
    assert result["joint_range_changed"] is False
    assert result["collision_proxy_status"] == "REQUIRES_REBUILD"
    assert result["swept_surface_clearance"]["status"] in {"CLEAR", "PASSED"}
    assert set(result["geometries"]) == {"body", "cover"}
    # Exercise the actual rebuilt-proxy and dynamics path, rather than treating
    # a successful visual-surface certificate as a complete physical asset.
    from physrealx.domain import AssetSpec

    def parameter(value):
        return dict(value=value, source="default", confidence=1.0)

    source_asset = AssetSpec(
        resource_id="video.synthetic",
        version="1.0.0",
        display_name="Synthetic video fixture",
        root_link="body",
        links=("body", "cover"),
        geometries=[
            dict(
                name=f"{name}_{role}",
                link=name,
                role=role,
                representation="mesh",
                uri=f"artifact://{ref.artifact_id}",
            )
            for name, ref in (("body", body_ref), ("cover", cover_ref))
            for role in ("visual", "collision")
        ],
        joints=[
            dict(
                name="hinge",
                parent="body",
                child="cover",
                joint_type="revolute",
                axis=motion["axis"],
                lower_limit=0,
                upper_limit=0.7,
                damping=0.1,
                friction=0.05,
                effort_limit=3,
            )
        ],
        physics=[
            dict(
                link=name,
                mass_kg=parameter(1),
                center_of_mass_m=list(mesh.center_mass),
                inertia_kg_m2=np.diag([0.01, 0.01, 0.01]).tolist(),
                static_friction=parameter(0.4),
                dynamic_friction=parameter(0.2),
                restitution=parameter(0.01),
            )
            for name, mesh in (("body", body), ("cover", cover))
        ],
        evidence=[reference, geometry],
        metadata={"reconstructed_motion": motion},
    )
    assembly = px.artifacts.put_json(
        dict(
            schema="physrealx.asset-assembly-bundle/1",
            asset_spec=source_asset.model_dump(mode="json"),
        )
    )
    proxies = px.assets.rebuild_video_collision(output)
    validation = px.assets.validate_video_articulation(
        output, assembly, proxies, config={"rollout_seconds": 0.05}
    )
    verified = json.loads(px.artifacts.get_bytes(validation))
    assert verified["status"] == "PASSED", verified
    assert verified["continuous_proxy_clearance"]["continuous_surface_clearance_certified"]
    assert verified["physics"]["sweep_strategy"]["base_samples"] == 361
    assert len(verified["physics"]["rollouts"]) == 3
    assert all(row["clearance_passed"] for row in verified["physics"]["rollouts"])
    assert verified["published_asset"] is False
