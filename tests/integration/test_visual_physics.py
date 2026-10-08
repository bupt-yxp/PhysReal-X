"""Synthetic evidence tests; no generated sample is presented as a real measurement."""

import io
import json
from copy import deepcopy
from pathlib import Path
from subprocess import CompletedProcess

import numpy as np
import pytest

try:
    import trimesh
except ModuleNotFoundError as exc:
    if exc.name == "trimesh":
        pytest.skip("Install the physical extra for mesh tests", allow_module_level=True)
    raise
from PIL import Image
from pydantic import ValidationError

from physrealx import ArtifactRef, AssetSpec, PhysRealX
from physrealx.errors import DomainValidationError
from physrealx.integrations._asset_preview import link_transforms, pose_matrix
from physrealx.integrations.physical_estimation import PhysicsPrediction
from physrealx.integrations.reconstructed_asset import reconstructed_motion
from physrealx.integrations.usd_reference import reference_link_transforms
from physrealx.integrations.video_articulation import (
    VideoArticulationRefinementConfig,
)
from physrealx.reconstruction import joint_transform


def estimate(value):
    return {
        "value": value,
        "lower": value * 0.5,
        "upper": max(value * 2, 0.01),
        "basis": "synthetic regression fixture",
    }


def test_registration_preserves_reconstruction_faces_colors_and_pose(tmp_path):
    from physrealx.integrations.collision import load_meshes

    px = PhysRealX.open(tmp_path)
    _, reference, registration, _ = fixture(px, False)
    registration["source_frame"] = "capture_z_up"
    capture_to_reference = trimesh.transformations.rotation_matrix(0.4, [0, 0, 1])
    capture_to_reference[:3, 3] = [0.2, -0.1, 0.3]
    registration["rotation"] = capture_to_reference[:3, :3].tolist()
    registration["translation_m"] = capture_to_reference[:3, 3].tolist()
    captured = {}
    state = {}
    for group, link in zip(("0", "1"), reference["links"], strict=True):
        mesh = load_meshes(px.artifacts, ArtifactRef.model_validate(link["geometry"]))[0]
        mesh.visual.vertex_colors = np.column_stack(
            [np.arange(len(mesh.vertices), dtype=np.uint8) * 20] * 3
            + [np.full(len(mesh.vertices), 255, dtype=np.uint8)]
        )
        mesh.apply_transform(np.linalg.inv(capture_to_reference))
        state[group] = px.artifacts.put_bytes(
            mesh.export(file_type="ply"), media_type="model/ply"
        ).model_dump(mode="json")
        captured[link["name"]] = load_meshes(
            px.artifacts, ArtifactRef.model_validate(state[group]), process=False
        )[0]
        # An unavailable reference surface must not be read or used to fill the gap.
        link["geometry"] = registration["source_visual"]
    registration["source_reference"] = px.artifacts.put_json(reference).model_dump(mode="json")
    registered = px.artifacts.put_json(registration)
    reconstruction = px.artifacts.put_json(
        {
            "schema": "physrealx.geometry-reconstruction-report/1",
            "unit": "m",
            "coordinate_frame": "capture_z_up",
            "bundle": registration["source_visual"],
            "prediction": {
                "parts": {"0": {}, "1": {"parent": "0", "kind": "hinge", "span": -0.6}},
                "states": {"start": state, "end": {}},
                "motion_pair": ["start", "end"],
                "segmented": True,
            },
        }
    )
    result = px.assets.register_reconstruction_geometry(reconstruction, registered)
    report = json.loads(px.artifacts.get_bytes(result))
    assert report["geometry_policy"] == "coordinate_transform_only"
    motion = report["reconstructed_motion"]
    assert motion["range"] == [-0.6, 0.0]
    np.testing.assert_allclose(motion["axis"], capture_to_reference[:3, :3] @ [1, 0, 0])
    np.testing.assert_allclose(motion["origin_m"], capture_to_reference[:3, 3])
    for name, source in captured.items():
        output = load_meshes(
            px.artifacts, ArtifactRef.model_validate(report["geometries"][name]), process=False
        )[0]
        assert len(output.vertices) == len(source.vertices)
        np.testing.assert_array_equal(output.faces, source.faces)
        np.testing.assert_array_equal(output.visual.vertex_colors, source.visual.vertex_colors)
        expected = trimesh.transform_points(source.vertices, capture_to_reference)
        np.testing.assert_allclose(output.vertices, expected, atol=1e-7, rtol=0)
    cover = load_meshes(
        px.artifacts, ArtifactRef.model_validate(report["geometries"]["cover"]), process=False
    )[0]
    predicted_end = trimesh.transform_points(
        cover.vertices,
        joint_transform(motion["kind"], motion["axis"], motion["origin_m"], motion["positions"][1]),
    )
    source_end = trimesh.transform_points(
        captured["cover"].vertices,
        capture_to_reference @ joint_transform("hinge", [1, 0, 0], [0, 0, 0], -0.6),
    )
    np.testing.assert_allclose(predicted_end, source_end, atol=1e-7, rtol=0)


@pytest.mark.parametrize("change", ["old_schema", "surface_edit"])
def test_assembly_rejects_stale_or_modified_reconstruction(tmp_path, change):
    px = PhysRealX.open(tmp_path)
    request, _, _, _ = fixture(px, False)
    geometry = json.loads(
        px.artifacts.get_bytes(ArtifactRef.model_validate(request["reconstructed_geometry"]))
    )
    if change == "old_schema":
        geometry["schema"] = "physrealx.registered-reconstruction-geometry/1"
    else:
        geometry["geometry_policy"] = "surface_completion"
    request["reconstructed_geometry"] = px.artifacts.put_json(geometry).model_dump(mode="json")
    with pytest.raises(DomainValidationError):
        px.assets.assemble(px.artifacts.put_json(request), method="reference_asset_assembly")


def fixture(px, overlap, joint_origin=(0, 0, 0), localized_overlap=False):
    links = []
    for name, location in (("body", [0, 0, -0.05]), ("cover", [0, 0, -0.04 if overlap else 0.15])):
        mesh = trimesh.creation.box([0.3, 0.2, 0.02])
        mesh.apply_translation(location)
        links.append(
            {
                "name": name,
                "geometry": px.artifacts.put_bytes(
                    mesh.export(file_type="glb"), media_type="model/gltf-binary"
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
    source_reconstruction_document = {
        "schema": "physrealx.geometry-reconstruction-report/1",
        "unit": "m",
        "coordinate_frame": "capture_z_up",
        "bundle": dummy,
        "prediction": {
            "parts": {
                "0": {},
                "1": {
                    "parent": "0",
                    "kind": "hinge",
                    "axis": [1, 0, 0],
                    "origin": list(joint_origin),
                    "span": -0.6,
                },
            },
            "states": {
                "start": {
                    "0": links[0]["geometry"],
                    "1": links[1]["geometry"],
                },
                "end": {},
            },
            "motion_pair": ["start", "end"],
            "segmented": True,
        },
    }
    source_reconstruction = px.artifacts.put_json(source_reconstruction_document)
    motion = reconstructed_motion(source_reconstruction_document, registration, model)
    reconstructed = px.artifacts.put_json(
        {
            "schema": "physrealx.registered-reconstruction-geometry/4",
            "coordinate_frame": "registered_reconstruction_start",
            "source_reconstruction": source_reconstruction.model_dump(mode="json"),
            "source_reference": reference.model_dump(mode="json"),
            "source_registration": registered.model_dump(mode="json"),
            "geometries": {link["name"]: link["geometry"] for link in links},
            "source_positions": {"hinge": 0.0},
            "reconstructed_motion": motion,
            "pose_policy": "module_one_predicted_start_pose_locked",
            "source_pose_reprojection_error_m": {"body": 0.0, "cover": 0.0},
            "geometry_policy": "coordinate_transform_only",
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
            "schema": "physrealx.visual-physics-prior/3",
            "source_reference": reference.model_dump(mode="json"),
            "registration": registered.model_dump(mode="json"),
            "source_geometry": reconstructed.model_dump(mode="json"),
            "prediction": prediction,
            "reconstructed_motion": motion,
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
    if localized_overlap:
        # The reconstructed appearance is correct. Only the collision proxy has a
        # small, disconnected false-positive fragment near the hinge.
        cover = trimesh.creation.box([0.3, 0.2, 0.02])
        cover.apply_translation([0, 0, 0.15])
        intrusion = trimesh.creation.box([0.06, 0.04, 0.015])
        intrusion.apply_translation([0, 0, -0.05])
        proxy = trimesh.Scene()
        proxy.add_geometry(cover, node_name="cover", geom_name="convex_0")
        proxy.add_geometry(intrusion, node_name="intrusion", geom_name="convex_1")
        proxy_ref = px.artifacts.put_bytes(
            proxy.export(file_type="glb"), media_type="model/gltf-binary"
        ).model_dump(mode="json")
        cover_index = next(index for index, link in enumerate(links) if link["name"] == "cover")
        report = json.loads(
            px.artifacts.get_bytes(ArtifactRef.model_validate(reports[cover_index]))
        )
        report["collision_geometry"] = proxy_ref
        report["metrics"]["convex_parts"] = 2.0
        reports[cover_index] = px.artifacts.put_json(report).model_dump(mode="json")
    request = {
        "schema": "physrealx.reference-assembly-input/2",
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


def test_predicted_asset_preserves_reconstructed_motion_and_does_not_certify_physics(tmp_path):
    px = PhysRealX.open(tmp_path)
    request, _, _, _ = fixture(px, False)
    assembled = px.assets.assemble(
        px.artifacts.put_json(request), method="reference_asset_assembly"
    ).output("asset_assembly")
    asset = AssetSpec.model_validate(json.loads(px.artifacts.get_bytes(assembled))["asset_spec"])
    assert (asset.joints[0].lower_limit, asset.joints[0].upper_limit) == (-0.6, 0.0)
    assert asset.metadata["reconstructed_motion"]["positions"] == [0.0, -0.6]
    assert asset.metadata["joint_limit_source"] == "module_one_reconstruction_start_end"
    assert asset.contact_exclusions == ()
    assert asset.physics[0].mass_kg.value == 1.2
    assert asset.evidence_state.value == "prior"
    report = px.assets.check_physics_feasibility(
        assembled, config={"sweep_samples": 11, "rollout_seconds": 0.02}
    )
    result = json.loads(px.artifacts.get_bytes(report))
    assert result["status"] == "PASSED"
    assert result["physical_validation"] is False
    assert result["locked_range"] == [-0.6, 0.0]


def test_removed_reference_assembly_schema_is_rejected(tmp_path):
    px = PhysRealX.open(tmp_path)
    request, _, _, _ = fixture(px, False)
    request["schema"] = "physrealx.reference-assembly-input/1"

    with pytest.raises(DomainValidationError, match="requires schema"):
        px.assets.assemble(px.artifacts.put_json(request), method="reference_asset_assembly")


def _video_refinement_fixture(px: PhysRealX, request: dict) -> tuple[ArtifactRef, dict]:
    source = json.loads(
        px.artifacts.get_bytes(ArtifactRef.model_validate(request["reconstructed_geometry"]))
    )
    source_motion = source["reconstructed_motion"]
    axis = np.asarray([0.99995, 0.01, 0.0])
    axis /= np.linalg.norm(axis)
    optimized = {
        **source_motion,
        "axis": axis.tolist(),
        "origin_m": [0.0, 0.001, 0.0],
        "policy": "video_constrained_axis_origin_fixed_range",
    }
    document = {
        "schema": "physrealx.video-articulation-refinement/2",
        "status": "PASSED",
        "evidence_scope": "calibrated_video_constrained_articulation",
        "source_reference": request["reference_model"],
        "source_geometry": request["reconstructed_geometry"],
        "registration": request["registration"],
        "source_evidence": px.artifacts.put_json({"scope": "synthetic-video"}).model_dump(
            mode="json"
        ),
        "config": VideoArticulationRefinementConfig().model_dump(mode="json"),
        "source_motion": source_motion,
        "optimized_motion": optimized,
        "range_preserved": True,
        "joint": {},
        "links": {},
        "heldout": {
            name: {
                "before": {"p90_m": 0.001, "samples": 10},
                "candidate": {"p90_m": 0.001, "samples": 10},
            }
            for name in source["geometries"]
        },
        "heldout_images": {
            "passed": True,
            "frames": [
                {
                    side: {
                        "frame_id": "synthetic-check",
                        "mask_iou": 1.0,
                        "rgb_mae": 0.0,
                        "depth_p90_m": 0.001,
                    }
                    for side in ("before", "candidate")
                }
            ],
        },
        "swept_surface_clearance": {
            "continuous_surface_clearance_certified": True,
            "minimum_clearance_lower_bound_m": 0.0002,
        },
        "surface_query": {"backend": "synthetic"},
        "optimizer": {"geometry_valid": True},
        "diagnostic_geometries": source["geometries"],
        "joint_range_changed": False,
        "collision_proxy_status": "REQUIRES_REBUILD",
        "visual_geometry_changed": False,
        "joint_model_changed": True,
        "independent_observation_validation": False,
        "physical_validation": False,
        "geometries": source["geometries"],
    }
    return px.artifacts.put_json(document), document


def test_assembly_accepts_video_joint_update_but_preserves_source_range(tmp_path):
    px = PhysRealX.open(tmp_path)
    request, _, _, _ = fixture(px, False)
    refinement, document = _video_refinement_fixture(px, request)
    request["video_articulation_refinement"] = refinement.model_dump(mode="json")

    assembly = px.assets.assemble(
        px.artifacts.put_json(request), method="reference_asset_assembly"
    ).output("asset_assembly")
    asset = AssetSpec.model_validate(json.loads(px.artifacts.get_bytes(assembly))["asset_spec"])

    assert asset.version.startswith("1.0.1+evidence.")
    assert asset.joints[0].axis == tuple(document["optimized_motion"]["axis"])
    assert (asset.joints[0].lower_limit, asset.joints[0].upper_limit) == (-0.6, 0.0)
    assert asset.metadata["joint_limit_source"] == (
        "module_one_start_end_preserved_by_video_refinement"
    )
    assert asset.metadata["reconstructed_motion"] == document["optimized_motion"]


def test_assembly_rejects_video_refinement_that_changes_joint_range(tmp_path):
    px = PhysRealX.open(tmp_path)
    request, _, _, _ = fixture(px, False)
    _, document = _video_refinement_fixture(px, request)
    document["optimized_motion"]["range"] = [-0.5, 0.0]
    request["video_articulation_refinement"] = px.artifacts.put_json(document).model_dump(
        mode="json"
    )

    with pytest.raises(DomainValidationError, match="invalid video articulation refinement"):
        px.assets.assemble(px.artifacts.put_json(request), method="reference_asset_assembly")


@pytest.mark.parametrize(
    "fault", ["missing_3d", "bad_3d", "missing_images", "bad_depth", "wrong_frame"]
)
def test_publication_rechecks_measurements_instead_of_trusting_passed_flags(tmp_path, fault):
    px = PhysRealX.open(tmp_path)
    request, _, _, _ = fixture(px, False)
    _, document = _video_refinement_fixture(px, request)
    if fault == "missing_3d":
        document["heldout"] = {}
    elif fault == "bad_3d":
        document["heldout"]["body"]["candidate"]["p90_m"] = 0.005
    elif fault == "missing_images":
        document["heldout_images"]["frames"] = []
    elif fault == "bad_depth":
        document["heldout_images"]["frames"][0]["candidate"]["depth_p90_m"] = None
    else:
        document["heldout_images"]["frames"][0]["candidate"]["frame_id"] = "another-frame"
    request["video_articulation_refinement"] = px.artifacts.put_json(document).model_dump(
        mode="json"
    )
    with pytest.raises(DomainValidationError, match="invalid video articulation refinement"):
        px.assets.assemble(px.artifacts.put_json(request), method="reference_asset_assembly")


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
    expected = reference_link_transforms(model, {"hinge": -0.3})["cover"]
    actual = link_transforms(asset, {"hinge": -0.3})["cover"] @ pose_matrix(cover_visual.local_pose)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_collision_fails_without_reducing_reconstructed_joint_range(tmp_path):
    from physrealx.integrations.assembly import assembled_asset
    from physrealx.integrations.physical_feasibility import (
        FeasibilityConfig,
        evaluate_asset_feasibility,
    )

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
    assert result["locked_range"] == [-0.6, 0.0]
    assert result["adjacent_contact_filter_disabled_for_audit"]
    assert "contact_constraints" not in result
    detailed = evaluate_asset_feasibility(
        px.artifacts,
        assembled_asset(json.loads(px.artifacts.get_bytes(assembled)), px.artifacts),
        config=FeasibilityConfig(sweep_samples=31, rollout_seconds=0.02),
        retain_contact_constraints=True,
    )
    assert detailed["contact_constraints"]
    assert all(
        contact["penetration_m"] > 0.001
        and -0.6 <= contact["q"] <= 0.0
        and len(contact["contact_position_m"]) == len(contact["normal_a_to_b"]) == 3
        for contact in detailed["contact_constraints"]
    )


def test_candidate_clearance_sweep_can_defer_dynamic_rollouts(tmp_path, monkeypatch):
    import mujoco

    from physrealx.integrations.assembly import assembled_asset
    from physrealx.integrations.physical_feasibility import (
        FeasibilityConfig,
        evaluate_asset_feasibility,
    )

    px = PhysRealX.open(tmp_path)
    request, _, _, _ = fixture(px, True)
    assembled = px.assets.assemble(
        px.artifacts.put_json(request), method="reference_asset_assembly"
    ).output("asset_assembly")
    asset = assembled_asset(json.loads(px.artifacts.get_bytes(assembled)), px.artifacts)

    def unexpected_step(*args, **kwargs):
        pytest.fail("intermediate collision candidates must defer dynamic rollouts")

    monkeypatch.setattr(mujoco, "mj_step", unexpected_step)
    result = evaluate_asset_feasibility(
        px.artifacts,
        asset,
        config=FeasibilityConfig(
            sweep_samples=3,
            adaptive_sweep_levels=0,
            rollout_seconds=0.02,
        ),
        run_rollouts=False,
    )

    assert result["rollouts"] == []
    assert not result["rollouts_evaluated"]
    assert len(result["sweep"]) == 3
    assert result["sweep_strategy"]["base_samples"] == 3
    assert result["sweep_strategy"]["evaluated_samples"] == 3
    assert result["sweep_strategy"]["workers"] >= 1


@pytest.fixture(scope="module")
def repaired_geometry(tmp_path_factory):
    px = PhysRealX.open(tmp_path_factory.mktemp("articulated-repair"))
    request, _, _, _ = fixture(px, False, localized_overlap=True)
    assembled = px.assets.assemble(
        px.artifacts.put_json(request), method="reference_asset_assembly"
    ).output("asset_assembly")
    check_config = {"sweep_samples": 31, "rollout_seconds": 0.02}
    failed = px.assets.check_physics_feasibility(assembled, config=check_config)
    assert json.loads(px.artifacts.get_bytes(failed))["status"] == "FAILED"

    refinement = px.assets.refine_articulated_collision(assembled, failed)
    report = json.loads(px.artifacts.get_bytes(refinement))
    return px, assembled, failed, report, check_config


def test_joint_local_collision_refinement_preserves_range_and_evidence_scope(
    repaired_geometry,
):
    px, assembled, _, report, check_config = repaired_geometry
    assert report["status"] == "PASSED"
    assert report["acceptance_scope"] == "local_surface_coverage_and_simulation_feasibility"
    assert report["physical_validation"] is False
    assert report["independent_observation_validation"] is False
    assert report["after"]["max_penetration_m"] <= 0.001
    assert report["visual_geometry_unchanged"]
    assert report["collision_geometry_refined"]
    assert report["joint_model_unchanged"]
    assert report["parent_child_collision_enabled"]
    assert report["source_face_ownership_preserved"]
    assert report["source_asset_lock"]["preserved_exactly"] is True
    assert report["source_asset_lock"]["joint"]["range"] == [-0.6, 0.0]
    assert all(item["status"] == "PASSED" for item in report["fidelity"].values())
    assert max(item["bidirectional_sample_max_m"] for item in report["fidelity"].values()) <= 0.003
    assert report["checkpoint"] is not None
    assert report["surface_query"]["backend"] in {"trimesh_cpu", "warp_cuda"}
    assert report["visual_motion_audit"]["locked_range"] == [-0.6, 0.0]
    assert report["visual_motion_audit"]["certified_lower_bound"] is False
    assert report["visual_motion_audit"]["samples"] == 61

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
    assert before_asset.version == "1.0.0"
    assert after_asset.version.startswith("1.0.1+evidence.")
    assert (after_asset.joints[0].lower_limit, after_asset.joints[0].upper_limit) == (-0.6, 0.0)
    before_visuals = {
        item.link: item.uri for item in before_asset.geometries if item.role.value == "visual"
    }
    after_visuals = {
        item.link: item.uri for item in after_asset.geometries if item.role.value == "visual"
    }
    assert before_visuals == after_visuals
    assert before_asset.physics == after_asset.physics
    assert before_asset.root_pose == after_asset.root_pose
    before_collisions = {
        item.link: item.uri for item in before_asset.geometries if item.role.value == "collision"
    }
    after_collisions = {
        item.link: item.uri for item in after_asset.geometries if item.role.value == "collision"
    }
    assert all(before_collisions[link] != after_collisions[link] for link in ("body", "cover"))
    evidence = json.loads(
        px.artifacts.get_bytes(ArtifactRef.model_validate(report["collision_refinement"]))
    )
    assert evidence["schema"] == "physrealx.articulated-collision-refinement/5"
    assert evidence["status"] == "CANDIDATE"
    assert evidence["evidence_scope"] == "collision_consistency_hypothesis"
    assert evidence["visual_geometry_changed"] is False
    assert evidence["collision_geometry_changed"] is True
    collision = next(
        value
        for ref in report["collision_reports"]
        for value in [json.loads(px.artifacts.get_bytes(ArtifactRef.model_validate(ref)))]
        if value["link"] == "cover"
    )
    assert "relative_volume_error" not in collision["metrics"]
    assert "volume_error" not in collision["gates"]


def test_articulated_refinement_resumes_the_last_accepted_checkpoint(repaired_geometry):
    px, assembled, failed, original, _ = repaired_geometry

    repeated = json.loads(
        px.artifacts.get_bytes(px.assets.refine_articulated_collision(assembled, failed))
    )

    assert repeated["resumed_from_checkpoint"] is True
    assert repeated["checkpoint"] is not None
    assert repeated["after"]["max_penetration_m"] == pytest.approx(
        original["after"]["max_penetration_m"]
    )
    assert repeated["source_asset_lock"] == original["source_asset_lock"]


@pytest.mark.parametrize(
    "tampering",
    [
        "parent_geometry",
        "moving_link",
        "range",
        "source_assembly",
        "source_sweep",
        "source_collision",
        "missing_collision_provenance",
        "other_collision",
        "fidelity_claim",
        "legacy_refinement",
    ],
)
def test_articulated_repair_rejects_inconsistent_provenance(repaired_geometry, tampering):
    px, _, _, result, _ = repaired_geometry

    def read(ref):
        return json.loads(px.artifacts.get_bytes(ArtifactRef.model_validate(ref)))

    def put(value):
        return px.artifacts.put_json(value).model_dump(mode="json")

    request = read(result["candidate_assembly_input"])
    refined = read(request["articulated_collision_refinement"])
    reports = [read(ref) for ref in request["collision_reports"]]
    collision = next(report for report in reports if report["link"] == "cover")
    if tampering == "fidelity_claim":
        refined["fidelity"]["cover"]["contact_coverage"] = 0.5
    elif tampering == "legacy_refinement":
        refined["schema"] = "physrealx.articulated-collision-refinement/1"
    elif tampering == "parent_geometry":
        refined["visual_geometries"]["body"] = refined["visual_geometries"]["cover"]
    elif tampering == "moving_link":
        refined["links"] = ["body"]
    elif tampering == "range":
        refined["locked_range"] = [-0.1, 0.1]
    elif tampering == "source_assembly":
        assembly = read(refined["source_assembly"])
        assembly["asset_spec"]["display_name"] = "Forged assembly"
        refined["source_assembly"] = put(assembly)
    elif tampering == "source_sweep":
        sweep = read(refined["source_feasibility"])
        sweep["source_assembly"] = put({"unrelated": True})
        refined["source_feasibility"] = put(sweep)
    elif tampering == "source_collision":
        previous = read(collision["articulation_refinement"]["source_collision_report"])
        previous["unrelated_evidence"] = True
        collision["articulation_refinement"]["source_collision_report"] = put(previous)
    elif tampering == "missing_collision_provenance":
        del collision["articulation_refinement"]
    else:
        next(report for report in reports if report["link"] == "body")["collision_geometry"] = (
            collision["collision_geometry"]
        )
    request["articulated_collision_refinement"] = put(refined)
    for report in reports:
        if "articulation_refinement" in report:
            report["articulation_refinement"]["collision_refinement"] = deepcopy(
                request["articulated_collision_refinement"]
            )
    request["collision_reports"] = [put(report) for report in reports]
    with pytest.raises(DomainValidationError):
        px.assets.assemble(px.artifacts.put_json(request), method="reference_asset_assembly")


def test_invalid_prediction_and_wrong_source_are_rejected(tmp_path):
    px = PhysRealX.open(tmp_path)
    request, _, _, prediction = fixture(px, False)
    prediction["bodies"]["body"]["mass_kg"]["value"] = -1
    with pytest.raises(ValidationError):
        PhysicsPrediction.model_validate(prediction)
    prior = json.loads(
        px.artifacts.get_bytes(ArtifactRef.model_validate(request["predicted_physics"]))
    )
    prior["reconstructed_motion"]["range"] = [-0.1, 0.1]
    request["predicted_physics"] = px.artifacts.put_json(prior).model_dump(mode="json")
    with pytest.raises(DomainValidationError, match="reconstructed motion"):
        px.assets.assemble(px.artifacts.put_json(request), method="reference_asset_assembly")


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "same_image_bytes",
        "revision",
        "weights",
        "template",
        "image",
        "image_provenance",
        "worker",
        "json",
        "prediction",
    ],
)
def test_physics_prior_cache_is_bound_to_inference_inputs(tmp_path, monkeypatch, change):
    from physrealx.integrations import physical_estimation

    px = PhysRealX.open(tmp_path / "project")
    request, _, _, prediction = fixture(px, False)
    model = tmp_path / "qwen-fixture"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"synthetic weights")
    (model / "chat_template.jinja").write_text("synthetic template")
    stream = io.BytesIO()
    Image.new("RGB", (4, 4)).save(stream, format="PNG")
    image = px.artifacts.put_bytes(stream.getvalue(), media_type="image/png")
    calls = []

    def infer(args, **kwargs):
        calls.append(args)
        Path(args[-1]).write_text(
            json.dumps(
                {
                    "text": json.dumps(prediction),
                    "local_model_sha256": physical_estimation.model_file_fingerprints(str(model)),
                }
            )
        )
        return CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(physical_estimation.subprocess, "run", infer)
    inputs = {
        "reference": ArtifactRef.model_validate(request["reference_model"]),
        "registration": ArtifactRef.model_validate(request["registration"]),
        "geometry": ArtifactRef.model_validate(request["reconstructed_geometry"]),
        "images": (image, image),
        "python": "/synthetic/python",
        "model": str(model),
        "revision": "synthetic-revision",
    }
    cached = px.assets.predict_physics(**inputs)
    saved = json.loads(px.artifacts.get_bytes(cached))
    if change == "same_image_bytes":
        path = tmp_path / "same-image.png"
        path.write_bytes(px.artifacts.get_bytes(image))
        duplicate = px.artifacts.put_file(path, media_type="image/png")
        assert duplicate.artifact_id != image.artifact_id
        inputs["images"] = (duplicate, duplicate)
    elif change == "revision":
        inputs["revision"] = "different-revision"
    elif change == "weights":
        (model / "model.safetensors").write_bytes(b"different weights")
    elif change == "template":
        (model / "chat_template.jinja").write_text("different template")
    elif change in {"image", "image_provenance"}:
        stream = io.BytesIO()
        Image.new("RGB", (4, 4), "red").save(stream, format="PNG")
        changed_image = px.artifacts.put_bytes(stream.getvalue(), media_type="image/png")
        if change == "image":
            inputs["images"] = (image, changed_image)
        else:
            saved["images"] = [changed_image.model_dump(mode="json")] * 2
            cached = px.artifacts.put_json(saved)
    elif change == "worker":
        saved["inference_identity"]["worker_sha256"] = "obsolete-code"
        cached = px.artifacts.put_json(saved)
    elif change == "json":
        cached = px.artifacts.put_bytes(b"damaged JSON", media_type="application/json")
    elif change == "prediction":
        saved["prediction"]["bodies"]["body"]["mass_kg"]["value"] = -1
        cached = px.artifacts.put_json(saved)
    updated = px.assets.predict_physics(**inputs, cached_prior=cached)
    cache_hit = change in {"none", "same_image_bytes"}
    assert len(calls) == (1 if cache_hit else 2)
    assert updated == cached if cache_hit else px.artifacts.verify(updated)


def test_video_backprojection_rejects_camera_errors_and_out_of_range_motion(tmp_path):
    from physrealx.integrations.video_articulation import (
        VideoArticulationEvidence,
        observations,
    )

    px = PhysRealX.open(tmp_path)
    request, model, _, _ = fixture(px, False)
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
        stream = io.BytesIO()
        Image.fromarray(np.full((20, 20, 3), index, dtype=np.uint8)).save(stream, format="PNG")
        rgb = px.artifacts.put_bytes(stream.getvalue(), media_type="image/png")
        camera = np.eye(4)
        camera[2, 3] = 1
        frames.append(
            {
                "frame_id": str(index),
                "partition": "fit" if index < 2 else "check",
                "rgb": rgb,
                "depth": depth,
                "link_mask": label,
                "intrinsics": [100, 100, 10, 10],
                "camera_from_reference": camera.tolist(),
                "positions": {"hinge": 0},
            }
        )
    evidence = VideoArticulationEvidence(
        reference=request["reference_model"],
        registration=request["registration"],
        source_geometry=request["reconstructed_geometry"],
        capture_id="synthetic",
        depth_scale_m=1,
        link_ids={"body": 1, "cover": 2},
        frames=frames,
    )
    motion = json.loads(
        px.artifacts.get_bytes(ArtifactRef.model_validate(request["reconstructed_geometry"]))
    )["reconstructed_motion"]
    config = VideoArticulationRefinementConfig()
    observed = observations(px.artifacts, evidence, model, motion, config)
    assert len(observed["body"]["fit"]) == 2
    np.testing.assert_allclose(observed["body"]["fit"][0].points_reference[:, 2], 0)
    frames[0]["positions"] = {"hinge": 1.2}
    invalid = VideoArticulationEvidence.model_validate(
        {**evidence.model_dump(mode="python"), "frames": frames}
    )
    with pytest.raises(DomainValidationError, match="exceeds"):
        observations(px.artifacts, invalid, model, motion, config)
    frames[0]["positions"] = {"hinge": 0}
    frames[0]["camera_from_reference"][0][0] = 2
    invalid = VideoArticulationEvidence.model_validate(
        {**evidence.model_dump(mode="python"), "frames": frames}
    )
    with pytest.raises(DomainValidationError, match="calibrated"):
        observations(px.artifacts, invalid, model, motion, config)


def test_refinement_budget_failure_does_not_publish_an_accepted_asset(tmp_path):
    px = PhysRealX.open(tmp_path)
    request, _, _, _ = fixture(px, True)
    assembly = px.assets.assemble(
        px.artifacts.put_json(request), method="reference_asset_assembly"
    ).output("asset_assembly")
    feasibility = px.assets.check_physics_feasibility(
        assembly, config={"sweep_samples": 5, "rollout_seconds": 0.02}
    )
    ref = px.assets.refine_articulated_collision(
        assembly, feasibility, config={"max_rounds": 1, "max_parts_per_link": 2}
    )
    report = json.loads(px.artifacts.get_bytes(ref))
    assert report["status"] == "FAILED"
    assert report["candidate_assembly_input"] is None
    assert report["source_face_ownership_preserved"] is True
    assert report["locked_range"] == [-0.6, 0.0]
    preview = px.assets.preview_physical_asset(assembly, rejected_refinement=ref, frames=2)
    with Image.open(io.BytesIO(px.artifacts.get_bytes(preview))) as gif:
        assert json.loads(gif.info["comment"])["rejected_refinement"] == ref.model_dump(mode="json")
        assert gif.size == (1350, 360)


def test_clear_asset_is_audited_without_forcing_a_geometry_change(tmp_path):
    px = PhysRealX.open(tmp_path)
    request, _, _, _ = fixture(px, False)
    source = px.artifacts.put_json(request)
    assembly = px.assets.assemble(source, method="reference_asset_assembly").output(
        "asset_assembly"
    )
    check = px.assets.check_physics_feasibility(
        assembly, config={"sweep_samples": 5, "rollout_seconds": 0.02}
    )
    result = json.loads(
        px.artifacts.get_bytes(px.assets.refine_articulated_collision(assembly, check))
    )
    assert result["status"] == "PASSED"
    assert result["collision_geometry_refined"] is False
    assert result["candidate_assembly_input"] == source.model_dump(mode="json")


def test_native_parent_contacts_match_explicit_all_pairs(tmp_path, monkeypatch):
    import xml.etree.ElementTree as ET
    from itertools import combinations
    from types import SimpleNamespace

    import mujoco

    px = PhysRealX.open(tmp_path)
    request, _, _, _ = fixture(px, True)
    assembly = px.assets.assemble(
        px.artifacts.put_json(request), method="reference_asset_assembly"
    ).output("asset_assembly")
    config = {"sweep_samples": 31, "rollout_seconds": 0.02}
    native = json.loads(
        px.artifacts.get_bytes(px.assets.check_physics_feasibility(assembly, config=config))
    )
    assert native["pair_broadphase"] == {
        "method": "mujoco_native",
        "parent_child_filter_disabled": True,
        "explicit_pairs": 0,
    }
    assert native["max_penetration_m"] > 0
    model_class = mujoco.MjModel

    def all_pairs(xml, **kwargs):
        root = ET.fromstring(xml)
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        geoms = [
            (body.get("name"), g.get("name"))
            for body in root.iter("body")
            for g in body.findall("geom")
            if g.get("contype") == "1"
        ]
        for (a, ga), (b, gb) in combinations(geoms, 2):
            if a != b:
                ET.SubElement(contact, "pair", geom1=ga, geom2=gb)
        return model_class.from_xml_string(ET.tostring(root, encoding="unicode"), **kwargs)

    monkeypatch.setattr(mujoco, "MjModel", SimpleNamespace(from_xml_string=all_pairs))
    explicit = json.loads(
        px.artifacts.get_bytes(px.assets.check_physics_feasibility(assembly, config=config))
    )
    assert explicit["pair_broadphase"]["explicit_pairs"] > 0
    np.testing.assert_allclose(
        [r["penetration_m"] for r in native["sweep"]],
        [r["penetration_m"] for r in explicit["sweep"]],
        atol=1e-10,
    )
    assert native["collision_pairs"] == explicit["collision_pairs"]


def test_native_contact_detection_remains_enabled_outside_joint_limits(tmp_path, monkeypatch):
    import mujoco

    px = PhysRealX.open(tmp_path)
    request, _, _, _ = fixture(px, True)
    assembly = px.assets.assemble(
        px.artifacts.put_json(request), method="reference_asset_assembly"
    ).output("asset_assembly")
    seen = []

    def escaped_step(model, data):
        assert model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
        assert model.npair == 0
        data.qpos[0] = model.jnt_range[0, 1] + np.ptp(model.jnt_range[0])
        mujoco.mj_forward(model, data)
        seen.append(float(data.qpos[0]))

    monkeypatch.setattr(mujoco, "mj_step", escaped_step)
    result = json.loads(
        px.artifacts.get_bytes(
            px.assets.check_physics_feasibility(
                assembly, config={"sweep_samples": 3, "rollout_seconds": 0.02}
            )
        )
    )
    assert seen and all(q > result["locked_range"][1] for q in seen)
    assert len(result["rollouts"]) == 3
    assert all(row["stable"] for row in result["rollouts"])


def test_finite_dynamic_rollout_cannot_pass_with_colliding_poses(tmp_path, monkeypatch):
    from mujoco import rollout

    px = PhysRealX.open(tmp_path)
    request, _, _, _ = fixture(px, False, joint_origin=(0, 0, 0.04))
    assembly = px.assets.assemble(
        px.artifacts.put_json(request), method="reference_asset_assembly"
    ).output("asset_assembly")
    original = rollout.Rollout.rollout

    def colliding_states(self, *args, **kwargs):
        states, sensors = original(self, *args, **kwargs)
        address = int(args[0].joint("asset/hinge").qposadr[0])
        states[:, :, 1 + address] = 2.2
        return states, sensors

    monkeypatch.setattr(rollout.Rollout, "rollout", colliding_states)
    result = json.loads(
        px.artifacts.get_bytes(
            px.assets.check_physics_feasibility(
                assembly, config={"sweep_samples": 9, "rollout_seconds": 0.02}
            )
        )
    )
    assert result["minimum_clearance_lower_bound_m"] >= result["required_clearance_m"]
    assert all(row["stable"] for row in result["rollouts"])
    assert all(row["contact_poses_evaluated"] > 0 for row in result["rollouts"])
    assert any(not row["clearance_passed"] for row in result["rollouts"])
    assert result["status"] == "FAILED"
