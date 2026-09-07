"""Registration and packaging on known synthetic geometry; native rendering is opt-in."""

from __future__ import annotations

import io
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
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr(
            "model/joint_info.json",
            json.dumps(
                [
                    {"id": 0, "joint": "heavy"},
                    {
                        "id": 1,
                        "joint": "slider",
                        "jointData": {"axis": {"direction": [1.0, 0.0, 0.0]}},
                    },
                ]
            ),
        )
    bundle = px.artifacts.put_bytes(payload.getvalue(), media_type="application/zip")
    report = px.artifacts.put_json(
        {
            "schema": "physrealx.geometry-reconstruction-report/1",
            "source_camera_artifact_id": camera.artifact_id,
            "source_depth_artifact_id": depth_ref.artifact_id,
            "evaluation_scope": "held_out_views",
            "training_frame_ids": train,
            "validation_frame_ids": test,
            "bundle": bundle.model_dump(mode="json"),
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
    with pytest.raises(DomainValidationError, match="split leaks"):
        px.assets.align_cameras_metric(
            camera,
            px.artifacts.put_json(leaked),
            method="articulated_surface_icp",
            config={"joint": "slide", "points_per_view": 2400},
        )


def test_connected_foreground_is_bounded_and_excludes_detached_objects():
    rng = np.random.default_rng(3)
    body = rng.uniform([0, 0, 0.04], [0.1, 0.1, 0.14], (2000, 3))
    stray = rng.uniform([0.5, 0, 0.02], [0.52, 0.02, 0.04], (200, 3))
    points = module.connected_foreground(np.concatenate([body, stray]), 0, 0.012)
    assert len(points) == len(body)
    with pytest.raises(DomainValidationError, match="bounded spatial grid"):
        module.connected_foreground(np.concatenate([body, body + 1e10]), 0, 1e-4)
