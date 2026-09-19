"""Synthetic evidence tests; no generated sample is presented as a real measurement."""

import io
import json

import numpy as np
import pytest
import trimesh
from PIL import Image
from pydantic import ValidationError

from physrealx import ArtifactRef, AssetSpec, PhysRealX
from physrealx.errors import DomainValidationError
from physrealx.integrations._asset_preview import link_transforms, pose_matrix
from physrealx.integrations.physical_estimation import PhysicsPrediction, locked_motion
from physrealx.integrations.usd_reference import reference_link_transforms
from physrealx.integrations.video_geometry import GeometryRefinementConfig, fit_surface


def estimate(value):
    return {
        "value": value,
        "lower": value * 0.5,
        "upper": max(value * 2, 0.01),
        "basis": "synthetic regression fixture",
    }


def fixture(px, overlap, joint_origin=(0, 0, 0), localized_overlap=False):
    links = []
    for name, location in (("body", [0, 0, -0.05]), ("cover", [0, 0, -0.04 if overlap else 0.15])):
        mesh = trimesh.creation.box([0.3, 0.2, 0.02])
        mesh.apply_translation(location)
        geometry = mesh
        if name == "cover" and localized_overlap:
            intrusion = trimesh.creation.box([0.06, 0.04, 0.015])
            intrusion.apply_translation([0, 0, -0.05])
            geometry = trimesh.util.concatenate([mesh, intrusion])
        links.append(
            {
                "name": name,
                "geometry": px.artifacts.put_bytes(
                    geometry.export(file_type="glb"), media_type="model/gltf-binary"
                ).model_dump(mode="json"),
            }
        )
    model = {
        "schema": "physrealx.usd-reference/1",
        "unit": "m",
        "coordinate_frame": "reference_z_up",
        "root_link": "body",
        "links": links,
        "joints": [
            {
                "name": "hinge",
                "parent": "body",
                "child": "cover",
                "joint_type": "hinge",
                "limit_unit": "rad",
                "axis": [1, 0, 0],
                "origin_m": list(joint_origin),
                "lower_limit": -1.5,
                "upper_limit": 1.5,
            }
        ],
    }
    reference = px.artifacts.put_json(model)
    dummy = px.artifacts.put_json({"scope": "synthetic"}).model_dump(mode="json")
    registration = {
        "schema": "physrealx.metric-alignment-report/1",
        "status": "PASSED",
        "evidence_scope": "model_conditioned_full_pose_registration",
        "target_frame": "reference_z_up",
        "rotation": np.eye(3).tolist(),
        "translation_m": [0, 0, 0],
        "source_reference": reference.model_dump(mode="json"),
        "source_camera": dummy,
        "source_visual": dummy,
        "motion_groups": {"0": ["body"], "1": ["cover"]},
        "train_frame_ids": ["a"],
        "holdout_frame_ids": ["b"],
        "observed_joint": "hinge",
        "state_order": ["start", "end"],
        "evaluations": {"start": {"joint_position": 0.3}, "end": {"joint_position": -0.3}},
    }
    registered = px.artifacts.put_json(registration)
    reconstructed = px.artifacts.put_json(
        {
            "schema": "physrealx.registered-reconstruction-geometry/1",
            "coordinate_frame": "reference_link_rest",
            "source_reference": reference.model_dump(mode="json"),
            "source_registration": registered.model_dump(mode="json"),
            "geometries": {link["name"]: link["geometry"] for link in links},
        }
    )
    prediction = {
        "bodies": {
            name: {
                "material": "synthetic",
                "mass_kg": estimate(mass),
                "static_friction": estimate(0.4),
                "dynamic_friction": estimate(0.2),
                "restitution": estimate(0.01),
            }
            for name, mass in (("body", 1.2), ("cover", 0.4))
        },
        "joints": {
            "hinge": {
                "damping": estimate(0.1),
                "friction": estimate(0.05),
                "effort_limit": estimate(3),
            }
        },
    }
    prior = px.artifacts.put_json(
        {
            "schema": "physrealx.visual-physics-prior/1",
            "source_reference": reference.model_dump(mode="json"),
            "registration": registered.model_dump(mode="json"),
            "source_geometry": reconstructed.model_dump(mode="json"),
            "prediction": prediction,
            "motion": locked_motion(model, registration),
            "evidence_state": "prior",
        }
    )
    reports = [
        px.assets.build_collision(
            ArtifactRef.model_validate(link["geometry"]),
            method="coacd",
            config={"source_unit": "m", "link": link["name"]},
        )
        .output("collision_build_report")
        .model_dump(mode="json")
        for link in links
    ]
    request = {
        "schema": "physrealx.reference-assembly-input/1",
        "resource_id": "visual.fixture",
        "version": "1.0.0",
        "display_name": "Fixture",
        "reference_model": reference.model_dump(mode="json"),
        "reconstructed_geometry": reconstructed.model_dump(mode="json"),
        "registration": registered.model_dump(mode="json"),
        "collision_reports": reports,
        "predicted_physics": prior.model_dump(mode="json"),
    }
    return request, model, registration, prediction


def test_predicted_asset_preserves_endpoints_and_does_not_certify_physics(tmp_path):
    px = PhysRealX.open(tmp_path)
    request, _, _, _ = fixture(px, False)
    assembled = px.assets.assemble(
        px.artifacts.put_json(request), method="reference_asset_assembly"
    ).output("asset_assembly")
    asset = AssetSpec.model_validate(json.loads(px.artifacts.get_bytes(assembled))["asset_spec"])
    assert (asset.joints[0].lower_limit, asset.joints[0].upper_limit) == (-0.3, 0.3)
    assert asset.contact_exclusions == ()
    assert asset.physics[0].mass_kg.value == 1.2
    assert asset.evidence_state.value == "prior"
    report = px.assets.check_physics_feasibility(
        assembled, config={"sweep_samples": 11, "rollout_seconds": 0.02}
    )
    result = json.loads(px.artifacts.get_bytes(report))
    assert result["status"] == "PASSED"
    assert result["physical_validation"] is False
    assert result["locked_range"] == [-0.3, 0.3]


def test_registered_reconstruction_preserves_shared_rest_coordinates(tmp_path):
    px = PhysRealX.open(tmp_path)
    request, model, _, _ = fixture(px, False, joint_origin=(0.03, 0.04, 0.05))
    assembled = px.assets.assemble(
        px.artifacts.put_json(request), method="reference_asset_assembly"
    ).output("asset_assembly")
    asset = AssetSpec.model_validate(json.loads(px.artifacts.get_bytes(assembled))["asset_spec"])
    local_poses = {
        geometry.link: geometry.local_pose.translation_m for geometry in asset.geometries
    }
    assert local_poses["body"] == (0.0, 0.0, 0.0)
    assert local_poses["cover"] == (-0.03, -0.04, -0.05)
    physics = {body.link: body for body in asset.physics}
    np.testing.assert_allclose(physics["body"].center_of_mass_m, [0, 0, -0.05])
    np.testing.assert_allclose(physics["cover"].center_of_mass_m, [-0.03, -0.04, 0.10])
    cover_visual = next(
        geometry
        for geometry in asset.geometries
        if geometry.link == "cover" and geometry.role.value == "visual"
    )
    expected = reference_link_transforms(model, {"hinge": 0.3})["cover"]
    actual = link_transforms(asset, {"hinge": 0.3})["cover"] @ pose_matrix(
        cover_visual.local_pose
    )
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_collision_fails_without_reducing_observed_range(tmp_path):
    px = PhysRealX.open(tmp_path)
    request, _, _, _ = fixture(px, True)
    assembled = px.assets.assemble(
        px.artifacts.put_json(request), method="reference_asset_assembly"
    ).output("asset_assembly")
    result = json.loads(
        px.artifacts.get_bytes(
            px.assets.check_physics_feasibility(
                assembled, config={"sweep_samples": 31, "rollout_seconds": 0.02}
            )
        )
    )
    assert result["status"] == "FAILED"
    assert result["max_penetration_m"] > 0.001
    assert result["locked_range"] == [-0.3, 0.3]
    assert result["adjacent_contact_filter_disabled_for_audit"]


def test_joint_local_geometry_refinement_removes_visual_ghost_and_preserves_range(tmp_path):
    px = PhysRealX.open(tmp_path)
    request, _, _, _ = fixture(px, False, localized_overlap=True)
    assembled = px.assets.assemble(
        px.artifacts.put_json(request), method="reference_asset_assembly"
    ).output("asset_assembly")
    check_config = {"sweep_samples": 31, "rollout_seconds": 0.02}
    failed = px.assets.check_physics_feasibility(assembled, config=check_config)
    assert json.loads(px.artifacts.get_bytes(failed))["status"] == "FAILED"

    refinement = px.assets.refine_articulated_geometry(assembled, failed)
    report = json.loads(px.artifacts.get_bytes(refinement))
    assert report["status"] == "PASSED"
    assert report["after"]["max_penetration_m"] <= 1e-6
    assert report["visual_geometry_refined"]
    assert report["joint_model_unchanged"]
    assert report["parent_child_collision_enabled"]
    assert report["removed_volume_fraction"] < 0.2
    assert report["removed_surface_fraction"] < 0.25
    assert report["surface_error_p95_m"] < 0.008

    candidate = ArtifactRef.model_validate(report["candidate_assembly_input"])
    refined = px.assets.assemble(candidate, method="reference_asset_assembly").output(
        "asset_assembly"
    )
    final = json.loads(
        px.artifacts.get_bytes(px.assets.check_physics_feasibility(refined, config=check_config))
    )
    assert final["status"] == "PASSED"
    before_asset = AssetSpec.model_validate(
        json.loads(px.artifacts.get_bytes(assembled))["asset_spec"]
    )
    after_asset = AssetSpec.model_validate(
        json.loads(px.artifacts.get_bytes(refined))["asset_spec"]
    )
    assert (after_asset.joints[0].lower_limit, after_asset.joints[0].upper_limit) == (-0.3, 0.3)
    before_visuals = {
        item.link: item.uri for item in before_asset.geometries if item.role.value == "visual"
    }
    after_visuals = {
        item.link: item.uri for item in after_asset.geometries if item.role.value == "visual"
    }
    assert before_visuals["body"] == after_visuals["body"]
    assert before_visuals["cover"] != after_visuals["cover"]


def test_invalid_prediction_and_wrong_source_are_rejected(tmp_path):
    px = PhysRealX.open(tmp_path)
    request, _, _, prediction = fixture(px, False)
    prediction["bodies"]["body"]["mass_kg"]["value"] = -1
    with pytest.raises(ValidationError):
        PhysicsPrediction.model_validate(prediction)
    prior = json.loads(
        px.artifacts.get_bytes(ArtifactRef.model_validate(request["predicted_physics"]))
    )
    prior["motion"]["range"] = [-0.1, 0.1]
    request["predicted_physics"] = px.artifacts.put_json(prior).model_dump(mode="json")
    with pytest.raises(DomainValidationError, match="endpoints"):
        px.assets.assemble(px.artifacts.put_json(request), method="reference_asset_assembly")


def test_geometry_improves_heldout_surface_without_topology_change():
    original = trimesh.creation.icosphere(subdivisions=2, radius=0.10)
    truth = original.copy()
    truth.vertices *= 1.015
    fit = trimesh.sample.sample_surface(truth, 3000, seed=1)[0]
    check = trimesh.sample.sample_surface(truth, 1500, seed=2)[0]
    refined, report = fit_surface(original, fit, check, GeometryRefinementConfig())
    assert report["status"] == "ACCEPTED"
    assert report["check_p90_candidate_m"] < report["check_p90_before_m"]
    assert report["max_vertex_displacement_m"] <= 0.003
    np.testing.assert_array_equal(refined.faces, original.faces)


def test_wrong_geometry_update_is_rejected_on_heldout():
    original = trimesh.creation.icosphere(subdivisions=2, radius=0.10)
    wrong = original.copy()
    wrong.vertices *= 1.015
    fit = trimesh.sample.sample_surface(wrong, 1500, seed=1)[0]
    check = trimesh.sample.sample_surface(original, 1500, seed=2)[0]
    refined, report = fit_surface(original, fit, check, GeometryRefinementConfig())
    assert report["status"] == "REJECTED"
    np.testing.assert_array_equal(refined.vertices, original.vertices)


def test_video_backprojection_rejects_camera_errors_and_out_of_range_motion(tmp_path):
    from physrealx.integrations.video_geometry import VideoGeometryEvidence, _observations

    px = PhysRealX.open(tmp_path)
    request, model, registration, _ = fixture(px, False)
    frames = []
    for index in range(4):
        stream = io.BytesIO()
        np.save(stream, np.full((20, 20), 1 + index * 0.001), allow_pickle=False)
        depth = px.artifacts.put_bytes(stream.getvalue(), media_type="application/x-npy")
        mask = np.ones((20, 20), dtype=np.uint8)
        mask[:, 10:] = 2
        stream = io.BytesIO()
        Image.fromarray(mask).save(stream, format="PNG")
        label = px.artifacts.put_bytes(stream.getvalue(), media_type="image/png")
        camera = np.eye(4)
        camera[2, 3] = 1
        frames.append(
            {
                "frame_id": str(index),
                "partition": "fit" if index < 2 else "check",
                "depth": depth,
                "link_mask": label,
                "intrinsics": [100, 100, 10, 10],
                "camera_from_reference": camera.tolist(),
                "positions": {"hinge": 0},
            }
        )
    evidence = VideoGeometryEvidence(
        reference=request["reference_model"],
        registration=request["registration"],
        source_geometry=request["reconstructed_geometry"],
        capture_id="synthetic",
        depth_scale_m=1,
        link_ids={"body": 1, "cover": 2},
        frames=frames,
    )
    motion = locked_motion(model, registration)
    observed = _observations(px.artifacts, evidence, model, motion, GeometryRefinementConfig())
    assert len(observed["body"]["fit"]) == 2
    np.testing.assert_allclose(observed["body"]["fit"][0][:, 2], 0)
    frames[0]["positions"] = {"hinge": 1.2}
    invalid = evidence.model_copy(update={"frames": frames})
    with pytest.raises(DomainValidationError, match="locked"):
        _observations(px.artifacts, invalid, model, motion, GeometryRefinementConfig())
    frames[0]["positions"] = {"hinge": 0}
    frames[0]["camera_from_reference"][0][0] = 2
    invalid = evidence.model_copy(update={"frames": frames})
    with pytest.raises(DomainValidationError, match="calibrated"):
        _observations(px.artifacts, invalid, model, motion, GeometryRefinementConfig())
