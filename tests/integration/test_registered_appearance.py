"""Registration and packaging on known synthetic geometry; native rendering is opt-in."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from physrealx import PhysRealX
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations import registration as module
from physrealx.integrations.rgbd import MetricRGBDInput, RGBDView

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")


def output(operation, name):
    return next(row.artifact for row in operation.outputs if row.name == name)


@pytest.fixture
def registration_input(tmp_path: Path, monkeypatch):
    px = PhysRealX.open(tmp_path / "workspace", discover_plugins=False)
    body = trimesh.creation.box([0.35, 0.34, 0.34])
    body.apply_translation([-0.03, 0, 0.17])
    drawer = trimesh.creation.box([0.25, 0.25, 0.10])
    drawer.apply_translation([0.04, 0, 0.075])
    reference = px.artifacts.put_json(
        {
            "schema": "physrealx.usd-reference/1",
            "unit": "m",
            "coordinate_frame": "reference_z_up",
            "root_link": "body",
            "dimensions_m": [0.35, 0.34, 0.34],
            "links": [
                {
                    "name": name,
                    "geometry": px.artifacts.put_bytes(
                        mesh.export(file_type="glb"), media_type="model/gltf-binary"
                    ).model_dump(mode="json"),
                }
                for name, mesh in [("body", body), ("drawer", drawer)]
            ],
            "joints": [
                {
                    "name": "slide",
                    "joint_type": "slide",
                    "parent": "body",
                    "child": "drawer",
                    "axis": [1, 0, 0],
                    "origin_m": [0, 0, 0],
                    "lower_limit": 0.0,
                    "upper_limit": 0.28,
                }
            ],
        }
    )
    capture = {
        "schema": "physrealx.camera-rig-calibration/1",
        "coordinate_frame": "capture_metric",
        "coordinate_convention": {"frame": "capture_metric", "translation_unit": "m"},
        "reference_height_m": 0.34,
        "scale_status": "resolved",
        "unit": "m",
        "streams": [],
    }
    views, train, test = [], [], []
    rotation = np.diag([1.0, -1.0, -1.0])
    shift = np.array([0.0, 0.0, 1.0])
    for state, q in [("closed", 0.05), ("open", 0.21)]:
        images = []
        for index in range(6):
            identifier = f"{state}.{index}"
            points = np.concatenate(
                [
                    trimesh.sample.sample_surface(body, 1600, seed=index)[0],
                    trimesh.sample.sample_surface(drawer, 800, seed=index + 30)[0] + [q, 0, 0],
                ]
            )
            points = (points - shift) @ rotation
            depth = points[:, 2].reshape(40, 60).astype("float32")
            rays = (points / points[:, 2, None]).reshape(40, 60, 3)
            pose = np.column_stack([np.eye(3), np.zeros(3)])
            views.append(
                RGBDView(
                    identifier,
                    state,
                    {},
                    pose,
                    np.zeros((40, 60, 3), "uint8"),
                    depth,
                    rays,
                    np.ones((40, 60), bool),
                )
            )
            images.append({"cam_from_world": pose.tolist(), "name": identifier})
            (test if index % 3 == 0 else train).append(identifier)
        capture["streams"].append(
            {
                "stream_name": state,
                "components": [
                    {
                        "component_id": 0,
                        "images": images,
                        "fiducial_tracks": [
                            {"xyz": [x, y, 1.0]}
                            for x, y in [(-0.6, -0.6), (0.6, -0.6), (0.6, 0.6), (-0.6, 0.6)]
                        ],
                    }
                ],
            }
        )
    camera = px.artifacts.put_json(capture)
    depth_ref = px.artifacts.put_bytes(
        b"synthetic depth loader fixture", media_type="application/zip"
    )
    # Registration consumes the public prediction, never a method-specific archive.
    bundle = px.artifacts.put_bytes(b"opaque reconstruction bundle", media_type="application/zip")
    mesh_ref = px.artifacts.put_bytes(b"mesh fixture", media_type="model/ply").model_dump(
        mode="json"
    )
    report = px.artifacts.put_json(
        {
            "schema": "physrealx.geometry-reconstruction-report/1",
            "source_camera_artifact_id": camera.artifact_id,
            "source_depth_artifact_id": depth_ref.artifact_id,
            "evaluation_scope": "held_out_views",
            "training_frame_ids": train,
            "validation_frame_ids": test,
            "bundle": bundle.model_dump(mode="json"),
            "prediction": {
                "parts": {
                    "root": {},
                    "drawer": {"parent": "root", "kind": "slide", "axis": [2, 0, 0], "span": 0.1},
                },
                "states": {"start": {"root": mesh_ref, "drawer": mesh_ref}, "end": {}},
                "motion_pair": ["start", "end"],
            },
        }
    )
    request = module.ReferenceRegistrationInput(
        rgbd=MetricRGBDInput(
            camera_rig=camera, enhanced_depth=depth_ref, reference_model=reference
        ),
        reconstruction_report=report,
    )
    monkeypatch.setattr(
        module, "load_rgbd", lambda source, store: (json.loads(json.dumps(capture)), {}, views)
    )
    return px, camera, request, views, rotation, shift


def test_articulated_registration_preserves_projection_and_rejects_bad_holdout(
    registration_input, tmp_path
):
    px, camera, request, views, known_rotation, known_translation = registration_input
    # Wrong predicted type/axis must remain measurable, not steer the reference pose.
    prediction = json.loads(px.artifacts.get_bytes(request.reconstruction_report))
    prediction["prediction"]["parts"]["drawer"].update(kind="hinge", axis=[0, 0, 1])
    request = request.model_copy(
        update={"reconstruction_report": px.artifacts.put_json(prediction)}
    )
    evidence = px.artifacts.put_json(request)
    operation = px.assets.align_cameras_metric(
        camera,
        evidence,
        method="articulated_surface_icp",
        config={"joint": "slide", "points_per_view": 2400},
    )
    report_ref = output(operation, "metric_alignment_report")
    report = module.load_registration(px.artifacts, report_ref)
    rotation, translation = np.asarray(report["rotation"]), np.asarray(report["translation_m"])
    np.testing.assert_allclose(rotation, known_rotation, atol=0.02)
    np.testing.assert_allclose(translation, known_translation, atol=0.008)
    assert report["evaluations"]["closed"]["joint_position"] == pytest.approx(0.05, abs=0.012)
    assert report["evaluations"]["open"]["joint_position"] == pytest.approx(0.21, abs=0.012)
    registered = json.loads(px.artifacts.get_bytes(output(operation, "camera_rig_calibration")))
    pose = np.asarray(registered["streams"][0]["components"][0]["images"][0]["cam_from_world"])
    original = np.array([0.1, 0.2, 0.8])
    aligned = original @ rotation.T + translation
    np.testing.assert_allclose(pose[:, :3] @ aligned + pose[:, 3], original, atol=1e-8)
    assert report["gates"]["independent_metric_truth"] is False
    from pxr import Usd

    from physrealx import ArtifactRef, AssetSpec
    from physrealx.integrations.asset_prior import PhysicsPrior, ReferenceAssemblyInput
    from physrealx.integrations.usd import validate_usd_stage

    model = json.loads(px.artifacts.get_bytes(request.rgbd.reference_model))
    collisions = tuple(
        output(
            px.assets.build_collision(
                ArtifactRef.model_validate(link["geometry"]),
                method="coacd",
                config={"source_unit": "m", "link": link["name"]},
            ),
            "collision_build_report",
        )
        for link in model["links"]
    )
    assembly_input = ReferenceAssemblyInput(
        resource_id="registered.fixture",
        version="1.0.0",
        display_name="Synthetic registered fixture",
        reference_model=request.rgbd.reference_model,
        collision_reports=collisions,
        registration=report_ref,
        physics_prior=PhysicsPrior(
            density_kg_m3=650,
            static_friction=0.3,
            dynamic_friction=0.2,
            restitution=0,
            joint_damping=1,
            joint_friction=0.1,
            effort_limit=40,
            joint_models_adjacent_contact=True,
        ),
    )
    assembly = output(
        px.assets.assemble(
            px.artifacts.put_json(assembly_input), method="reference_asset_assembly"
        ),
        "asset_assembly",
    )
    asset = AssetSpec.model_validate(json.loads(px.artifacts.get_bytes(assembly))["asset_spec"])
    assert asset.metadata["registered_appearance"] == report_ref.model_dump(mode="json")
    package = px.assets.package(assembly, method="openusd")
    assert package.metrics["registered_appearance"] == 1
    path = tmp_path / "registered.usdz"
    px.artifacts.materialize(output(package, "usd_asset_bundle"), path)
    with zipfile.ZipFile(path) as archive:
        assert archive.read("appearance/artgs.zip") == px.artifacts.get_bytes(
            ArtifactRef.model_validate(report["source_visual"])
        )
    stage = Usd.Stage.Open(str(path))
    validate_usd_stage(stage, asset, px.artifacts)
    stage.SetEditTarget(stage.GetSessionLayer())
    stage.GetDefaultPrim().GetAttribute("physrealx:registeredAppearance").Set("{}")
    with pytest.raises(MethodExecutionError, match="appearance registration"):
        validate_usd_stage(stage, asset, px.artifacts)
    changed_model = px.artifacts.put_json(model)
    wrong_assembly = assembly_input.model_copy(update={"reference_model": changed_model})
    with pytest.raises(DomainValidationError, match="another USD reference"):
        px.assets.assemble(px.artifacts.put_json(wrong_assembly), method="reference_asset_assembly")
    tampered = {**report, "rotation": (-rotation).tolist()}
    with pytest.raises(DomainValidationError, match="proper rigid transform"):
        module.load_registration(px.artifacts, px.artifacts.put_json(tampered))
    # Perturb only held-out geometry; the solver must not absorb it during fitting.
    for view in views:
        if view.frame_id in report["holdout_frame_ids"]:
            view.rays[..., 0] += 0.10 / view.depth
    with pytest.raises(MethodExecutionError, match="held-out surface"):
        px.assets.align_cameras_metric(
            camera,
            px.artifacts.put_json(request),
            method="articulated_surface_icp",
            config={"joint": "slide", "points_per_view": 2400},
        )


def test_registration_rejects_camera_and_split_mismatch(registration_input):
    px, camera, request, _, _, _ = registration_input
    different = px.artifacts.put_json({"fixture": "different camera"})
    with pytest.raises(DomainValidationError, match="another camera"):
        px.assets.align_cameras_metric(
            different,
            px.artifacts.put_json(request),
            method="articulated_surface_icp",
            config={"joint": "slide", "points_per_view": 2400},
        )
    report = json.loads(px.artifacts.get_bytes(request.reconstruction_report))
    report["validation_frame_ids"].append(report["training_frame_ids"][0])
    leaked = request.model_copy(update={"reconstruction_report": px.artifacts.put_json(report)})
    with pytest.raises(DomainValidationError, match="views overlap"):
        px.assets.align_cameras_metric(
            camera,
            px.artifacts.put_json(leaked),
            method="articulated_surface_icp",
            config={"joint": "slide", "points_per_view": 2400},
        )


def test_registration_reuses_training_time_reference_alignment(registration_input, monkeypatch):
    px, camera, request, _, rotation, translation = registration_input
    reconstruction = json.loads(px.artifacts.get_bytes(request.reconstruction_report))
    training = tuple(sorted(reconstruction["training_frame_ids"]))
    holdout = tuple(sorted(reconstruction["validation_frame_ids"]))
    reconstruction["reference_depth_alignment"] = module.ReferenceDepthAlignment(
        source_camera=request.rgbd.camera_rig,
        source_depth=request.rgbd.enhanced_depth,
        source_reference=request.rgbd.reference_model,
        source_frame="capture_metric",
        target_frame="reference_z_up",
        observed_joint="slide",
        joint_type="slide",
        joint_axis=(1, 0, 0),
        joint_origin_m=(0, 0, 0),
        joint_limits=(0, 0.28),
        state_order=("closed", "open"),
        rotation=tuple(tuple(float(value) for value in row) for row in rotation),
        translation_m=tuple(float(value) for value in translation),
        evaluations={
            "closed": module.AlignmentStateEvidence(
                median_m=0.001, p90_m=0.002, points=100, joint_position=0.05
            ),
            "open": module.AlignmentStateEvidence(
                median_m=0.001, p90_m=0.002, points=100, joint_position=0.21
            ),
        },
        foreground_fractions={"closed/train": 1, "open/train": 1},
        train_frame_ids=training,
        holdout_frame_ids=holdout,
        metrics={"max_holdout_median_m": 0.001, "max_holdout_p90_m": 0.002},
    ).model_dump(mode="json", by_alias=True)
    cached_request = request.model_copy(
        update={"reconstruction_report": px.artifacts.put_json(reconstruction)}
    )

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("cached alignment must not be fitted again")

    monkeypatch.setattr(module, "fit_reference_depth_alignment", unexpected_fit)
    operation = px.assets.align_cameras_metric(
        camera,
        px.artifacts.put_json(cached_request),
        method="articulated_surface_icp",
        config={"joint": "slide"},
    )
    report = module.load_registration(px.artifacts, output(operation, "metric_alignment_report"))
    assert report["fit_stage"] == "pre_reconstruction"
    assert report["gates"]["reference_guided_training"] is True


def test_registration_split_ignores_unselected_source_views(registration_input):
    _, _, _, views, _, _ = registration_input
    extra = RGBDView(
        "closed.unselected",
        "closed",
        views[0].camera,
        views[0].pose,
        views[0].rgb,
        views[0].depth,
        views[0].rays,
        views[0].mask,
    )
    selected, train_ids, test_ids = module._select_registration_views(
        [*views, extra],
        [view.frame_id for index, view in enumerate(views) if index % 6 != 0],
        [views[0].frame_id, views[6].frame_id],
    )
    assert extra.frame_id not in {view.frame_id for view in selected}
    assert {view.frame_id for view in selected} == train_ids | test_ids


@pytest.mark.parametrize(
    ("training", "holdout", "message"),
    [
        (["closed.1", "missing"], ["closed.0"], "unknown capture"),
        (["closed.1", "closed.1"], ["closed.0"], "unique nonempty"),
        (["closed.1"], [], "unique nonempty"),
    ],
)
def test_registration_split_rejects_invalid_references(
    registration_input, training, holdout, message
):
    _, _, _, views, _, _ = registration_input
    with pytest.raises(DomainValidationError, match=message):
        module._select_registration_views(views, training, holdout)


def test_connected_foreground_uses_reference_extent_and_rejects_invalid_dimensions():
    rng = np.random.default_rng(3)
    body = rng.uniform([0, 0, 0.04], [0.1, 0.1, 0.14], (2000, 3))
    stray = rng.uniform([0.5, 0, 0.02], [0.52, 0.02, 0.04], (200, 3))
    points = module.connected_foreground(
        np.concatenate([body, stray]), 0, 0.012, 0.012, [0.1, 0.1, 0.14]
    )
    assert len(points) == len(body)
    with pytest.raises(DomainValidationError, match="reference dimensions"):
        module.connected_foreground(body, 0, 0.012, 0.012, [0, 0.1, 0.14])


def test_connected_foreground_keeps_low_profile_object_above_measured_plane_clearance():
    rng = np.random.default_rng(4)
    body = rng.uniform([0, 0, 0.0065], [0.1, 0.1, 0.02], (2000, 3))
    points = module.connected_foreground(body, 0, 0.012, 0.006, [0.1, 0.1, 0.1])
    assert len(points) == len(body)


def test_hinge_surface_fit_preserves_descending_capture_positions():
    import numpy as np

    from physrealx.integrations.registration import fit_articulated_surfaces
    from physrealx.reconstruction import joint_transform

    rng = np.random.default_rng(7)
    static = rng.uniform([-0.18, 0, 0.03], [0.18, 0.2, 0.22], (600, 3))
    moving = rng.uniform([-0.16, -0.012, 0.03], [0.16, 0.012, 0.2], (600, 3))
    origin = [0, 0, 0.03]
    observed = []
    for position in (1.1, 0.3):
        motion = joint_transform("hinge", [1, 0, 0], origin, position)
        observed.append(np.concatenate([static, moving @ motion[:3, :3].T + motion[:3, 3]]))
    rotation, translation, positions, _ = fit_articulated_surfaces(
        observed,
        static,
        moving,
        np.zeros(3),
        (0, np.pi / 2),
        seed=7,
        max_points=1000,
        joint_type="hinge",
        axis=[1, 0, 0],
        origin=origin,
    )
    assert positions == pytest.approx([1.1, 0.3], abs=0.01)
    assert rotation == pytest.approx(np.eye(3), abs=0.01)
    assert translation == pytest.approx(np.zeros(3), abs=0.002)
