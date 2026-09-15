"""Exercise the reference/RGB-D path on explicit synthetic sensor fixtures."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from physrealx import ArtifactRef, PhysRealX
from physrealx.errors import DomainValidationError, MethodExecutionError, ResourceConflictError
from physrealx.integrations._artifacts import read_array
from physrealx.integrations.capture_rgbd import (
    _metric_object_depth_validity,
    capture_frames,
    capture_support_region,
)
from physrealx.integrations.depth import CalibratedRGBDBundle, DepthFrame
from physrealx.integrations.rgbd import MetricRGBDInput
from physrealx.integrations.usd_reference import evaluate_reference_kinematics, import_usd_reference

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("scipy")
pytest.importorskip("pycolmap")
trimesh = pytest.importorskip("trimesh")
Usd = pytest.importorskip("pxr.Usd")
Gf = pytest.importorskip("pxr.Gf")
Sdf = pytest.importorskip("pxr.Sdf")
UsdGeom = pytest.importorskip("pxr.UsdGeom")
UsdPhysics = pytest.importorskip("pxr.UsdPhysics")


def output(operation, name):
    return next(item.artifact for item in operation.outputs if item.name == name)


def read(px, artifact):
    return json.loads(px.artifacts.get_bytes(artifact))


def write_reference(path: Path, *, up="Z", scale=0.01, hinge=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageMetersPerUnit(stage, scale)
    UsdGeom.SetStageUpAxis(stage, up)
    root = UsdGeom.Xform.Define(stage, "/model")
    stage.SetDefaultPrim(root.GetPrim())
    for name, center, extent in (
        ("body", (0, 0, 17), (35, 34, 34)),
        ("drawer", (0, 0, 20), (25, 25, 10)),
    ):
        body = UsdGeom.Xform.Define(stage, f"/model/{name}")
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        box = trimesh.creation.box(extents=extent)
        box.vertices += np.array(center)
        mesh = UsdGeom.Mesh.Define(stage, f"/model/{name}/mesh")
        mesh.CreatePointsAttr(box.vertices.tolist())
        mesh.CreateFaceVertexCountsAttr([3] * len(box.faces))
        mesh.CreateFaceVertexIndicesAttr(box.faces.ravel().tolist())
    kind = UsdPhysics.RevoluteJoint if hinge else UsdPhysics.PrismaticJoint
    joint = kind.Define(stage, "/model/joint")
    joint.CreateBody0Rel().SetTargets([Sdf.Path("/model/body")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path("/model/drawer")])
    joint.CreateAxisAttr("X")
    joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 20))
    joint.CreateLocalRot0Attr(Gf.Quatf(1))
    joint.CreateLowerLimitAttr(0)
    joint.CreateUpperLimitAttr(90 if hinge else 28)
    stage.GetRootLayer().Save()


@pytest.fixture
def capture(tmp_path):
    root = tmp_path / "capture"
    write_reference(root / "sim/usd/model.usda")
    (root / "sim/info.json").write_text(json.dumps({"dimension(meter)": "0.9 x 0.8 x 0.7"}))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    for state in ("start", "end"):
        directory = root / f"real/can/{state}"
        (directory / "img").mkdir(parents=True)
        (directory / "mask").mkdir(parents=True)
        (directory / "depth").mkdir()
        for index in range(8):
            rgb = np.full((240, 320, 3), 230, np.uint8)
            for identifier, (x, y) in zip(
                range(2, 6), ((20, 15), (255, 15), (20, 190), (255, 190))
            ):
                marker = cv2.aruco.generateImageMarker(dictionary, identifier, 35)
                rgb[y : y + 35, x : x + 35] = marker[..., None]
            depth = np.full((240, 320), 1000, np.uint16)
            # One sensor unit is shared by both articulated states even though their
            # observed heights differ.  The taller endpoint matches the USD envelope.
            depth[65:175, 105:215] = 660 if state == "start" else 800
            depth[110:112, 150:152] = 0
            # Positive sensor corruption inside the annotated object. It is not
            # the uint16 invalid sentinel and must be rejected geometrically.
            depth[85:87, 130:132] = 65000
            rgb[65:175, 105:215] = [75, 130, 180]
            assert cv2.imwrite(str(directory / f"img/{index}.png"), rgb)
            mask = np.zeros(depth.shape, np.uint8)
            mask[65:175, 105:215] = 255
            assert cv2.imwrite(str(directory / f"mask/{index}.png"), mask)
            np.save(directory / f"depth/{index}.npy", depth)
    px = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    config = {
        "depth_unit": "native_u16_unscaled",
        "coordinate_frame": "unregistered",
        "include_candidates": True,
    }
    obs = px.assets.ingest(
        root,
        resource_id="synthetic.observations",
        version="1.0.0",
        display_name="Synthetic registered RGB-D fixture",
        config=config,
    )
    manifest = px.assets.get_observations(obs)
    streams = []
    for state in ("start", "end"):
        name = f"can.{state}.rgb"
        rgb = next(s for s in manifest.streams if s.name == name)
        images = [
            {
                "name": f"{i:06d}.png",
                "camera_id": 1,
                "source_frame_index": i,
                "source_artifact_id": ref.artifact_id,
                "cam_from_world": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
            }
            for i, ref in enumerate(rgb.artifacts)
        ]
        tracks = []
        for identifier, (x, y) in enumerate(((35, 30), (270, 30), (35, 205), (270, 205))):
            tracks.append(
                {
                    "marker_id": identifier + 2,
                    "corner_index": 0,
                    "xyz": [(x - 160) / 260 * 5, (y - 120) / 260 * 5, 5],
                    "observations": [
                        {"image_name": image["name"], "xy": [x, y]} for image in images
                    ],
                }
            )
        streams.append(
            {
                "stream_name": name,
                "components": [
                    {
                        "component_id": 0,
                        "cameras": [
                            {
                                "camera_id": 1,
                                "model": "SIMPLE_PINHOLE",
                                "width": 320,
                                "height": 240,
                                "params": [260, 160, 120],
                            }
                        ],
                        "images": images,
                        "fiducial_tracks": tracks,
                    }
                ],
            }
        )
    camera = px.artifacts.put_json(
        {
            "schema": "physrealx.camera-rig-calibration/1",
            "scale_status": "unresolved",
            "source_observations": obs.model_dump(mode="json"),
            "streams": streams,
            "fiducial_alignments": [
                {
                    "source_stream_name": "can.start.rgb",
                    "target_stream_name": "can.end.rgb",
                    "scale": 1.0,
                    "target_from_source": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
                }
            ],
        }
    )
    return px, root, obs, camera, config


def scale_inputs(px, obs, camera):
    reference = px.assets.import_reference(obs, usd_path="sim/usd/model.usda")
    payload = px.artifacts.get_bytes(obs.artifact_id)
    obs_artifact = ArtifactRef(
        artifact_id=obs.artifact_id,
        media_type="application/vnd.physrealx.observation-set+json",
        size_bytes=len(payload),
    )
    evidence = px.artifacts.put_json(
        {
            "schema": "physrealx.model-scale-input/1",
            "observations": obs_artifact.model_dump(mode="json"),
            "reference_model": reference.model_dump(mode="json"),
        }
    )
    return reference, evidence


@pytest.mark.parametrize("frame_budget", [150, None])
def test_supplied_usd_and_real_array_contracts_execute_all_cpu_stages(capture, frame_budget):
    px, root, obs, camera, config = capture
    reference, evidence = scale_inputs(px, obs, camera)
    model = read(px, reference)
    assert model["dimensions_m"] == pytest.approx([0.35, 0.34, 0.34])
    assert model["metadata_conflicts"][0]["resolution"] == "explicitly_selected_usd"
    assert model["joints"][0]["upper_limit"] == pytest.approx(0.28)
    metric = px.assets.align_cameras_metric(camera, evidence, method="usd_depth_scale")
    report = read(px, output(metric, "metric_alignment_report"))
    assert report["status"] == "MODEL_CONDITIONED"
    assert not report["independent_metrology"]
    for state in report["streams"].values():
        assert state["depth_scale_m"] == pytest.approx(0.001, rel=0.02)
        assert state["scale_m"] == pytest.approx(0.2, rel=0.02)
        assert set(state["fit_views"]).isdisjoint(state["holdout_views"])
    assert len({state["native_height"] for state in report["streams"].values()}) == 2
    assert len({state["depth_scale_m"] for state in report["streams"].values()}) == 1
    metric_camera = output(metric, "camera_rig_calibration")
    # A view without enough visible markers for height estimation remains a
    # registered reconstruction view. Do not clip it to a partial image hull.
    metric_document = read(px, metric_camera)
    for calibration in metric_document["depth_calibrations"].values():
        calibration["support_views"] = calibration["support_views"][:1]
    metric_camera = px.artifacts.put_json(metric_document)
    rgbd = px.assets.prepare_rgbd(obs, metric_camera, stride=1, max_frames_per_stream=frame_budget)
    prepared = read(px, rgbd)
    assert len(prepared["frames"]) == 16
    assert all(frame["view"]["mask_source"] == "dataset_annotation" for frame in prepared["frames"])
    first = prepared["frames"][0]
    prepared_depth = read_array(px.artifacts, ArtifactRef.model_validate(first["depth"]))
    prepared_mask = read_array(
        px.artifacts, ArtifactRef.model_validate(first["view"]["object_mask"])
    )
    assert not prepared_depth[85:87, 130:132].any()
    assert prepared_mask[85:87, 130:132].all()
    assert first["view"]["depth_validity"]["rejected_foreground_pixels"] >= 4
    depth = px.assets.enhance_depth(rgbd, method="rgb_guided_fill")
    depth_report = read(px, output(depth, "enhancement_report"))
    assert depth_report["status"] == "SELF_SUPERVISED"
    assert not depth_report["gates"]["independent_reference"]
    assert depth.metrics["filled_pixels"] > 0
    source = px.artifacts.put_json(
        MetricRGBDInput(
            camera_rig=metric_camera,
            enhanced_depth=output(depth, "enhanced_depth"),
            reference_model=reference,
        )
    )
    geometry = px.assets.reconstruct_geometry(source, method="rgbd_fusion")
    surfaces = read(px, output(geometry, "geometry_reconstruction_report"))
    assert len(surfaces["observed_surfaces"]) == 2
    assert geometry.metrics["max_heldout_surface_median_m"] < 0.001
    assert not surfaces["gates"]["complete_unobserved_geometry"]
    for surface in surfaces["observed_surfaces"]:
        assert set(surface["fit_frame_ids"]).isdisjoint(surface["holdout_frame_ids"])
        mesh = trimesh.load(
            io.BytesIO(px.artifacts.get_bytes(ArtifactRef.model_validate(surface["artifact"]))),
            file_type="ply",
        )
        assert len(mesh.faces) > 100
    joint = model["joints"][0]
    state = evaluate_reference_kinematics(reference, px.artifacts, {joint["name"]: 0.28})
    assert np.asarray(read(px, state)["link_transforms"]["drawer"])[:3, 3] == pytest.approx(
        [0.28, 0, 0]
    )
    with pytest.raises(DomainValidationError, match="limits"):
        evaluate_reference_kinematics(reference, px.artifacts, {joint["name"]: 0.5})
    with pytest.raises(DomainValidationError, match="exactly"):
        evaluate_reference_kinematics(reference, px.artifacts, {})
    assert (
        px.assets.ingest(
            root,
            resource_id=obs.resource_id,
            version=obs.version,
            display_name="reuse",
            config=config,
        )
        == obs
    )
    with pytest.raises(ResourceConflictError, match="configuration"):
        px.assets.ingest(
            root,
            resource_id=obs.resource_id,
            version=obs.version,
            display_name="different",
            config={**config, "include_candidates": False},
        )


def test_capture_support_requires_observed_marker_geometry(capture):
    px, _, obs, camera, _ = capture
    _, evidence = scale_inputs(px, obs, camera)
    metric = px.assets.align_cameras_metric(camera, evidence, method="usd_depth_scale")
    document = read(px, output(metric, "camera_rig_calibration"))
    capture_support_region(document)
    document["streams"][0]["components"][0]["fiducial_tracks"] = []
    document["streams"][1]["components"][0]["fiducial_tracks"] = []
    with pytest.raises(DomainValidationError, match="four finite"):
        capture_support_region(document)


def test_metric_depth_validity_uses_world_geometry_not_camera_distance():
    depth = np.full((4, 5), 10.0)
    rays = np.zeros((4, 5, 3))
    rays[..., 2] = 1
    pose = np.column_stack((np.eye(3), [0, 0, 9.9]))
    support = {
        "origin": np.zeros(3),
        "normal": np.array([0, 0, 1]),
        "height_m": 0.2,
        "plane_rmse_m": 0.001,
    }
    valid = _metric_object_depth_validity(depth, rays, np.ones(depth.shape, bool), pose, support)
    assert valid.all()
    depth[0, 0] = 20
    valid = _metric_object_depth_validity(depth, rays, np.ones(depth.shape, bool), pose, support)
    assert not valid[0, 0]
    assert valid[1:, 1:].all()


def test_reference_loading_preserves_hinge_units_and_rejects_missing_units(capture, tmp_path):
    px, _, obs, _, _ = capture
    path = tmp_path / "hinge.usda"
    write_reference(path, hinge=True)
    original = px.assets.get_observations(obs)
    attachment = next(item for item in original.attachments if item.path.endswith("model.usda"))
    replacements = tuple(
        item.model_copy(
            update={"artifact": px.artifacts.put_file(path, media_type="model/vnd.usd")}
        )
        if item == attachment
        else item
        for item in original.attachments
    )
    manifest = original.model_copy(update={"attachments": replacements})
    reference = import_usd_reference(manifest, px.artifacts, usd_path="sim/usd/model.usda")
    model = read(px, reference)
    joint = model["joints"][0]
    assert joint["upper_limit"] == pytest.approx(np.pi / 2)
    state = read(
        px, evaluate_reference_kinematics(reference, px.artifacts, {joint["name"]: np.pi / 2})
    )
    transform = np.asarray(state["link_transforms"]["drawer"])
    assert transform[:3, :3] @ [0, 1, 0] == pytest.approx([0, 0, 1], abs=1e-6)
    stage = Usd.Stage.Open(str(path))
    stage.ClearMetadata("metersPerUnit")
    stage.GetRootLayer().Save()
    replacements = tuple(
        item.model_copy(
            update={"artifact": px.artifacts.put_file(path, media_type="model/vnd.usd")}
        )
        if item == attachment
        else item
        for item in original.attachments
    )
    with pytest.raises(DomainValidationError, match="explicitly author"):
        import_usd_reference(
            original.model_copy(update={"attachments": replacements}),
            px.artifacts,
            usd_path="sim/usd/model.usda",
        )
    with pytest.raises(DomainValidationError, match="package-relative"):
        import_usd_reference(original, px.artifacts, usd_path="../model.usda")
    with pytest.raises(DomainValidationError, match="absent"):
        import_usd_reference(original, px.artifacts, usd_path="sim/missing.usda")


def test_model_scale_rejects_camera_source_mismatch_and_insufficient_evidence(capture):
    px, _, obs, camera, _ = capture
    _, evidence = scale_inputs(px, obs, camera)
    document = read(px, camera)
    document["source_observations"]["artifact_id"] = evidence.artifact_id
    with pytest.raises(DomainValidationError, match="differ"):
        px.assets.align_cameras_metric(
            px.artifacts.put_json(document), evidence, method="usd_depth_scale"
        )
    document = read(px, camera)
    document["streams"][0]["components"][0]["fiducial_tracks"] = []
    with pytest.raises(MethodExecutionError, match="insufficient"):
        px.assets.align_cameras_metric(
            px.artifacts.put_json(document), evidence, method="usd_depth_scale"
        )
    image = read(px, camera)["streams"][0]["components"][0]["images"][0]
    image["source_artifact_id"] = evidence.artifact_id
    with pytest.raises(DomainValidationError, match="different capture"):
        list(
            capture_frames(px.assets.get_observations(obs), px.artifacts, "can.start.rgb", [image])
        )


def test_self_supervised_depth_cannot_claim_an_independent_reference(capture):
    _px, _, _, camera, _ = capture
    frame = DepthFrame(
        frame_id="one", rgb=camera, depth=camera, depth_scale_m=0.001, capture_id="source"
    )
    with pytest.raises(ValueError, match="evaluation scope"):
        CalibratedRGBDBundle(camera_rig=camera, frames=(frame,))
    with pytest.raises(ValueError, match="declared together"):
        frame.model_validate({**frame.model_dump(), "reference_capture_id": "other"})


@pytest.mark.parametrize("count", [True, 2, 512.5, "150", 513])
def test_rgbd_frame_budget_rejects_invalid_values(count):
    from physrealx.integrations.capture_rgbd import prepare_rgbd_frames

    with pytest.raises(DomainValidationError, match="integer stride"):
        prepare_rgbd_frames(None, None, None, max_frames_per_stream=count)


def test_all_rgbd_views_reject_oversized_stream_before_decoding(capture):
    px, _, obs, camera, _ = capture
    document = read(px, camera)
    document["streams"][-1]["components"][0]["images"] *= 65
    reference = px.artifacts.put_json(document)
    with pytest.raises(DomainValidationError, match="safety limit"):
        px.assets.prepare_rgbd(obs, reference, max_frames_per_stream=None)


def test_explicit_motion_selection_preserves_world_hinge_and_static_geometry(capture, tmp_path):
    px, _, obs, _, _ = capture
    path = tmp_path / "projected.usda"
    write_reference(path, hinge=True)
    stage = Usd.Stage.Open(str(path))
    joint = UsdPhysics.RevoluteJoint.Get(stage, "/model/joint")
    joint.GetBody0Rel().ClearTargets(True)
    joint.GetLocalPos0Attr().Set(Gf.Vec3f(1, 2, 3))
    anchor = UsdPhysics.FixedJoint.Define(stage, "/model/anchor")
    anchor.CreateBody1Rel().SetTargets([Sdf.Path("/model/body")])
    extra = UsdGeom.Xform.Define(stage, "/model/Unused")
    UsdPhysics.RigidBodyAPI.Apply(extra.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, "/model/Unused/mesh")
    box = trimesh.creation.box(extents=(1, 1, 1))
    mesh.CreatePointsAttr(box.vertices.tolist())
    mesh.CreateFaceVertexCountsAttr([3] * len(box.faces))
    mesh.CreateFaceVertexIndicesAttr(box.faces.ravel().tolist())
    stage.GetRootLayer().Save()
    original = px.assets.get_observations(obs)
    manifest = original.model_copy(
        update={
            "attachments": tuple(
                item.model_copy(
                    update={"artifact": px.artifacts.put_file(path, media_type="model/vnd.usd")}
                )
                if item.path == "sim/usd/model.usda"
                else item
                for item in original.attachments
            )
        }
    )
    reference = import_usd_reference(
        manifest,
        px.artifacts,
        usd_path="sim/usd/model.usda",
        root_link="body",
        moving_links=("drawer",),
    )
    model = read(px, reference)
    assert len(model["links"]) == 2 and len(model["joints"]) == 1
    assert model["joints"][0]["origin_m"] == pytest.approx([0.01, 0.02, 0.03])
    assert model["joints"][0]["parent"] == "body"
    assert model["motion_selection"]["fixed_bodies"] == ["Unused"]
    assert next(link for link in model["links"] if link["name"] == "body")["triangle_count"] == 24
    with pytest.raises(DomainValidationError, match="unknown"):
        import_usd_reference(
            manifest,
            px.artifacts,
            usd_path="sim/usd/model.usda",
            root_link="body",
            moving_links=("typo",),
        )


def test_reference_import_bytes_are_stable_across_wall_clock_changes(capture, monkeypatch):
    import io
    import zipfile

    px, _, observations, _, _ = capture
    first = px.assets.import_reference(observations, usd_path="sim/usd/model.usda")
    monkeypatch.setattr(zipfile.time, "localtime", lambda *args: (2031, 2, 3, 4, 5, 6, 0, 0, 0))
    second = px.assets.import_reference(observations, usd_path="sim/usd/model.usda")
    model, repeated = read(px, first), read(px, second)
    assert px.artifacts.get_bytes(model["bundle"]["artifact_id"]) == px.artifacts.get_bytes(
        repeated["bundle"]["artifact_id"]
    )
    for original_link, repeated_link in zip(model["links"], repeated["links"]):
        assert px.artifacts.get_bytes(original_link["geometry"]["artifact_id"]) == (
            px.artifacts.get_bytes(repeated_link["geometry"]["artifact_id"])
        )
    with zipfile.ZipFile(io.BytesIO(px.artifacts.get_bytes(model["bundle"]["artifact_id"]))) as z:
        assert all(entry.date_time == (1980, 1, 1, 0, 0, 0) for entry in z.infolist())
