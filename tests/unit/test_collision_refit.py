"""Counterexamples for source coverage and bounded collision-only refinement."""

import numpy as np
import pytest
import trimesh
from pydantic import ValidationError

from physrealx.integrations._collision_refit import (
    SurfaceCoverage,
    SurfacePatch,
    bind_surface,
    clip_patch_support,
    fit_patches,
)
from physrealx.integrations.articulated_geometry import (
    ArticulatedCollisionRefinementConfig,
    _meets_penetration_target,
)
from physrealx.integrations.physical_feasibility import FeasibilityConfig


def test_default_sweep_and_refinement_margin_cover_sub_grid_collision_peaks():
    assert FeasibilityConfig().sweep_samples == 361
    assert ArticulatedCollisionRefinementConfig().penetration_safety_margin_m == 5e-5
    assert _meets_penetration_target(0.000950036, 0.00095)
    assert not _meets_penetration_target(0.0009502, 0.00095)


def test_small_missing_contact_patch_cannot_hide_in_global_p95():
    large = trimesh.creation.box([0.3, 0.2, 0.02])
    contact = trimesh.creation.box([0.008, 0.008, 0.008])
    contact.apply_translation([0.17, 0, 0])
    visual = trimesh.util.concatenate([large, contact])
    initial = bind_surface(visual, [large, contact])
    audit = SurfaceCoverage(visual, initial, {1}, cell_size_m=0.01, seed=11)
    result, _ = audit.evaluate(
        [initial[0]], max_error_m=0.003, coverage_distance_m=0.002, min_coverage=0.99
    )
    assert result["surface_p95_m"] < 1e-6
    assert result["bidirectional_sample_max_m"] > 0.015
    assert result["contact_coverage"] == 0
    assert result["worst_cell_coverage"] == 0
    assert result["status"] == "FAILED"


def test_refit_retains_all_source_faces_without_mutating_visual_geometry():
    visual = trimesh.creation.icosphere(subdivisions=1, radius=0.03)
    visual.visual.vertex_colors = [123, 45, 67, 255]
    vertices, faces, colors = (
        visual.vertices.copy(),
        visual.faces.copy(),
        visual.visual.vertex_colors.copy(),
    )
    patch = bind_surface(visual, [visual.convex_hull])[0]
    fitted = fit_patches(visual, patch, skin_m=0.00025)
    assert fitted is not None
    assert len(fitted) == 2
    np.testing.assert_array_equal(
        np.sort(np.concatenate([p.faces for p in fitted])), np.arange(len(faces))
    )
    assert all(p.mesh.is_convex and p.mesh.is_volume for p in fitted)
    np.testing.assert_array_equal(visual.vertices, vertices)
    np.testing.assert_array_equal(visual.faces, faces)
    np.testing.assert_array_equal(visual.visual.vertex_colors, colors)


def test_degenerate_patch_rejects_split_without_dropping_source_faces():
    visual = trimesh.Trimesh(
        vertices=[[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    patch = SurfacePatch(trimesh.creation.box([0.02, 0.001, 0.001]), np.array([0]))
    source_faces = patch.faces.copy()
    assert fit_patches(visual, patch, skin_m=0.00025) is None
    np.testing.assert_array_equal(patch.faces, source_faces)


@pytest.mark.parametrize(
    "config",
    [
        {"max_removed_volume_fraction": 0.2},
        {"max_removed_proxy_surface_fraction": 0.15},
        {"max_surface_error_m": 0.008},
        {"proxy_skin_m": 0.002, "coverage_distance_m": 0.001},
        {"shape_trust_region_m": 0.003, "coverage_distance_m": 0.002},
        {"max_local_error_m": float("nan")},
        {"max_rounds": 0},
    ],
)
def test_legacy_deletion_and_inconsistent_configs_are_rejected(config):
    with pytest.raises(ValidationError):
        ArticulatedCollisionRefinementConfig.model_validate(config)


def test_support_plane_candidates_are_bounded_and_preserve_source_geometry():
    from types import SimpleNamespace

    from physrealx.domain import GeometryRole, JointType, PoseSpec
    from physrealx.integrations.articulated_geometry import _shape_refit_candidates

    body = trimesh.creation.box([0.01, 0.01, 0.01])
    cover = body.copy()
    visuals = {"body": body, "cover": cover}
    patches = {
        link: [SurfacePatch(mesh.convex_hull, np.arange(len(mesh.faces)))]
        for link, mesh in visuals.items()
    }
    coverage = {
        link: SurfaceCoverage(mesh, patches[link], {0}, cell_size_m=0.01, seed=1)
        for link, mesh in visuals.items()
    }
    asset = SimpleNamespace(
        root_link="body",
        root_pose=PoseSpec(),
        joints=[
            SimpleNamespace(
                name="hinge",
                parent="body",
                child="cover",
                parent_to_joint=PoseSpec(),
                joint_type=JointType.REVOLUTE,
                axis=(0, 0, 1),
                lower_limit=-0.1,
                upper_limit=0.1,
            )
        ],
        geometries=[
            SimpleNamespace(
                link=link,
                role=GeometryRole.COLLISION,
                local_pose=PoseSpec(translation_m=(-0.00425 if link == "body" else 0.00425, 0, 0)),
                scale=(1, 1, 1),
            )
            for link in visuals
        ],
    )
    pair = {
        "max_penetration_m": 0.0015,
        "peak_q": 0,
        "normal_a_to_b": [1, 0, 0],
        "contact_position_m": [0, 0, 0],
        "link_a": "body",
        "link_b": "cover",
        "geom_a": "asset/geometry/0/0",
        "geom_b": "asset/geometry/1/0",
    }
    source_vertices = {link: mesh.vertices.copy() for link, mesh in visuals.items()}
    proxy_vertices = {link: patches[link][0].mesh.vertices.copy() for link in patches}
    candidates = _shape_refit_candidates(
        asset,
        patches,
        visuals,
        coverage,
        {"collision_pairs": [pair]},
        0.001,
        ArticulatedCollisionRefinementConfig(min_proxy_volume_ratio=0.997),
    )
    assert {candidate["policy"] for candidate in candidates} == {"support_plane_trust_region"}
    assert len(candidates) == 1
    for candidate in candidates:
        proposal = candidate["patches"]
        assert candidate["fraction"] == 0.0625
        assert candidate["step_m"] <= 0.0018 + 1e-12
        assert candidate["max_adjacency_gap_m"] == 0
        assert candidate["min_volume_ratio"] >= 0.1
        assert candidate["minimum_thickness_m"] >= 0.00025
        assert candidate["constraint_count"] == 1
        assert candidate["support_plane_count"] == 2
        assert candidate["linear_constraint_residual_m"] <= 1e-10
        assert proposal["body"][0].mesh.bounds[1, 0] < proxy_vertices["body"][:, 0].max()
        assert proposal["cover"][0].mesh.bounds[0, 0] > proxy_vertices["cover"][:, 0].min()
        for link in visuals:
            np.testing.assert_array_equal(visuals[link].vertices, source_vertices[link])
            np.testing.assert_array_equal(proposal[link][0].faces, patches[link][0].faces)
            assert proposal[link][0].mesh.is_volume
    for link in patches:
        np.testing.assert_array_equal(patches[link][0].mesh.vertices, proxy_vertices[link])


def test_support_plane_clip_preserves_ownership_convexity_and_thickness():
    visual = trimesh.Trimesh(
        vertices=[[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    proxy = trimesh.creation.box([0.01, 0.01, 0.0005])
    result, metrics = clip_patch_support(
        SurfacePatch(proxy, np.array([0])),
        visual,
        [(np.array([1.0, 0, 0]), 0.003)],
        trust_region_m=0.0021,
        min_thickness_ratio=0.25,
        min_volume_ratio=0.5,
    )
    assert result is not None
    assert result.faces.tolist() == [0]
    assert result.mesh.is_volume and result.mesh.is_convex
    assert np.ptp(result.mesh.vertices[:, 2]) == pytest.approx(0.0005)
    assert metrics["volume_ratio"] == pytest.approx(0.8)
    assert metrics["max_vertex_displacement_m"] == pytest.approx(0.002)
    np.testing.assert_array_equal(visual.vertices, [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])


def test_halfspace_polyhedron_merges_coplanar_facets_before_glb_roundtrip():
    import io

    from physrealx.integrations._collision_refit import _halfspace_polyhedron

    box = trimesh.creation.box([0.01, 0.006, 0.002])
    points = np.vstack(
        [
            box.vertices,
            box.triangles_center,
            [[0.005, 0, 0], [-0.005, 0, 0], [0, 0.003, 0], [0, -0.003, 0]],
        ]
    )
    mesh = _halfspace_polyhedron(points)
    assert mesh.is_volume and mesh.is_convex and mesh.euler_number == 2
    decoded = trimesh.load(io.BytesIO(mesh.export(file_type="glb")), file_type="glb", process=True)
    geometries = list(decoded.geometry.values())
    assert len(geometries) == 1
    assert geometries[0].is_volume and geometries[0].is_convex
