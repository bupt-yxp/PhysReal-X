"""Counterexamples for source coverage and bounded collision-only refinement."""

import numpy as np
import pytest
import trimesh
from pydantic import ValidationError

from physrealx.errors import MethodExecutionError
from physrealx.integrations._collision_refit import (
    SurfaceCoverage,
    SurfacePatch,
    bind_surface,
    clip_patch_support,
    fit_one_sided_patch,
    fit_patches,
)
from physrealx.integrations._refinement_checkpoint import RefinementCheckpointStore
from physrealx.integrations._surface_query import CpuSurfaceQuery
from physrealx.integrations.articulated_geometry import (
    ArticulatedCollisionRefinementConfig,
    _conflicting_parts,
    _glb_stable_convex_proxy,
    _maximum_partition_seam_error,
    _meets_penetration_target,
    _partition_current_envelope,
    _publish_parts,
    _shape_candidate_order,
    _split_with_current_envelope,
    _stabilize_contact_normals,
    _support_line_search_fractions,
    _topology_refit_exploration_candidates,
)
from physrealx.integrations.physical_feasibility import FeasibilityConfig
from physrealx.storage import LocalArtifactStore


def test_default_sweep_and_refinement_margin_cover_sub_grid_collision_peaks():
    assert FeasibilityConfig().sweep_samples == 361
    refinement = ArticulatedCollisionRefinementConfig()
    assert refinement.penetration_safety_margin_m == 5e-5
    assert refinement.max_local_error_m - refinement.surface_fidelity_reserve_m == 0.00275
    assert _meets_penetration_target(0.0009504, 0.00095)
    assert not _meets_penetration_target(0.0009506, 0.00095)


def test_support_line_search_over_relaxes_only_near_clearance_target():
    assert max(_support_line_search_fractions(0.0003, None)) == 1.0
    assert _support_line_search_fractions(0.0001, None)[:3] == [2.0, 1.5, 1.25]
    assert _support_line_search_fractions(0.0001, 1.0)[:3] == [2.0, 1.5, 1.0]
    assert max(_support_line_search_fractions(0.0003, 0.03125)) == pytest.approx(0.046875)


def test_collision_selection_uses_the_supplied_optimization_target():
    evaluation = {
        "collision_pairs": [
            {
                "max_penetration_m": 0.00097,
                "link_a": "body",
                "geom_a": "body/3",
                "link_b": "cover",
                "geom_b": "cover/4",
            }
        ]
    }
    prefixes = {"body": "body/", "cover": "cover/"}

    assert _conflicting_parts(evaluation, prefixes, 0.001) == {
        "body": set(),
        "cover": set(),
    }
    assert _conflicting_parts(evaluation, prefixes, 0.00095) == {
        "body": {3},
        "cover": {4},
    }


def test_contact_normal_stabilization_aligns_sign_ambiguous_adjacent_samples():
    contacts = [
        {
            "link_a": "body",
            "geom_a": "body/2",
            "link_b": "cover",
            "geom_b": "cover/7",
            "penetration_m": 0.0018,
            "q": 0.15,
            "normal_a_to_b": [0.01, 0.04, -0.999],
        },
        {
            "link_a": "body",
            "geom_a": "body/2",
            "link_b": "cover",
            "geom_b": "cover/7",
            "penetration_m": 0.0017,
            "q": 0.16,
            "normal_a_to_b": [-0.004, -0.07, 0.997],
        },
        {
            "link_a": "cover",
            "geom_a": "cover/7",
            "link_b": "body",
            "geom_b": "body/2",
            "penetration_m": 0.0016,
            "q": 0.17,
            "normal_a_to_b": [0.003, 0.06, -0.998],
        },
        {
            "link_a": "body",
            "geom_a": "body/2",
            "link_b": "cover",
            "geom_b": "cover/7",
            "penetration_m": 0.0012,
            "q": 0.18,
            "normal_a_to_b": [0.999, 0.01, 0.0],
        },
    ]

    stabilized, diagnostics = _stabilize_contact_normals(contacts)

    canonical = []
    for contact in stabilized[:3]:
        normal = np.asarray(contact["normal_a_to_b"])
        if contact["link_a"] == "cover":
            normal = -normal
        canonical.append(normal / np.linalg.norm(normal))
    assert all(float(canonical[0] @ value) > 0.99 for value in canonical[1:])
    assert diagnostics == {
        "contacts": 4,
        "geometry_pairs": 1,
        "direction_clusters": 2,
        "sign_flips": 2,
        "cluster_cosine": 0.82,
    }
    assert contacts[1]["normal_a_to_b"] == [-0.004, -0.07, 0.997]


def test_parallel_cpu_surface_query_matches_exact_trimesh_distance():
    mesh = trimesh.creation.box([0.1, 0.2, 0.3])
    points = np.asarray([[0, 0, 0], [0.2, 0, 0], [0, 0, 0.4], [-0.2, 0.2, 0]])
    query = CpuSurfaceQuery(workers=2, batch_size=2)
    try:
        distance, faces = query.nearest(mesh, points)
    finally:
        query.close()
    expected = trimesh.proximity.closest_point(mesh, points)
    np.testing.assert_allclose(distance, expected[1])
    np.testing.assert_array_equal(faces, expected[2])
    signed_query = CpuSurfaceQuery(workers=2, batch_size=2)
    try:
        signed = signed_query.signed_distance(mesh, points)
    finally:
        signed_query.close()
    np.testing.assert_allclose(signed, trimesh.proximity.signed_distance(mesh, points))


def test_parallel_surface_query_copies_topology_without_visual_state():
    source = trimesh.creation.box([0.1, 0.2, 0.3])

    class GeometryOnly:
        vertices = source.vertices
        faces = source.faces

        def copy(self):
            raise AssertionError("visual-aware mesh copy must not run")

    points = np.asarray([[0.2, 0, 0], [0, 0, 0.4]])
    query = CpuSurfaceQuery(workers=2, batch_size=1)
    try:
        distance, _ = query.nearest(GeometryOnly(), points)
    finally:
        query.close()
    np.testing.assert_allclose(distance, trimesh.proximity.closest_point(source, points)[1])


def test_refinement_checkpoint_restores_face_ownership_atomically(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_assembly = store.put_json({"source": "assembly"})
    source_feasibility = store.put_json({"source": "feasibility"})
    visuals = {
        "body": trimesh.creation.box([0.1, 0.1, 0.01]),
        "cover": trimesh.creation.box([0.1, 0.01, 0.1]),
    }
    patches = {
        link: [SurfacePatch(mesh.convex_hull, np.arange(len(mesh.faces)), depth=2)]
        for link, mesh in visuals.items()
    }
    patches["body"][0].baseline_volume_m3 = float(patches["body"][0].mesh.volume)
    patches["body"][0].baseline_minimum_thickness_m = 0.004
    unsupported = trimesh.creation.box([0.01, 0.01, 0.01])
    unsupported.apply_translation([0.2, 0, 0])
    patches["body"].append(SurfacePatch(unsupported, np.empty(0, dtype=int), depth=0))
    references = {}
    for link, values in patches.items():
        scene = trimesh.Scene()
        for index, value in enumerate(values):
            scene.add_geometry(value.mesh, geom_name=f"convex_{index:05d}")
        references[link] = store.put_bytes(
            scene.export(file_type="glb"), media_type="model/gltf-binary"
        )
    contact_faces = {link: np.asarray([0, 1, 2]) for link in visuals}
    checkpoint = RefinementCheckpointStore(
        store,
        source_assembly=source_assembly,
        source_feasibility=source_feasibility,
        config={"seed": 1},
        mode="required",
        resume_from=None,
    )
    reference = checkpoint.save(
        patches=patches,
        contact_faces=contact_faces,
        references=references,
        state={"next_round": 3, "evaluation": {}, "fidelity": {}},
        best={
            "patches": patches,
            "references": references,
            "state": {"evaluation": {"max_penetration_m": 0.0012}, "score": 1.25},
        },
    )
    assert reference is not None

    restored = RefinementCheckpointStore(
        store,
        source_assembly=source_assembly,
        source_feasibility=source_feasibility,
        config={"seed": 1},
        mode="required",
        resume_from=None,
    ).load(visuals)

    assert restored is not None and restored["next_round"] == 3
    for link, visual in visuals.items():
        np.testing.assert_array_equal(
            restored["patches"][link][0].faces, np.arange(len(visual.faces))
        )
        np.testing.assert_array_equal(restored["contact_faces"][link], contact_faces[link])
    assert len(restored["patches"]["body"]) == 2
    assert not len(restored["patches"]["body"][1].faces)
    assert restored["patches"]["body"][0].baseline_volume_m3 == pytest.approx(
        patches["body"][0].mesh.volume
    )
    assert restored["patches"]["body"][0].baseline_minimum_thickness_m == pytest.approx(0.004)
    assert restored["best"]["score"] == 1.25
    assert restored["best"]["evaluation"]["max_penetration_m"] == 0.0012
    for link, visual in visuals.items():
        np.testing.assert_array_equal(
            restored["best"]["patches"][link][0].faces, np.arange(len(visual.faces))
        )


def test_refinement_checkpoint_identity_uses_recursive_artifact_content(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    geometry_a = store.put_bytes(b"same geometry", media_type="model/gltf-binary")
    geometry_b = store.put_bytes(b"same geometry", media_type="model/gltf-binary")
    assembly_a = store.put_json(
        {"resource_id": "asset.first", "geometry": geometry_a.model_dump(mode="json")}
    )
    assembly_b = store.put_json(
        {"resource_id": "asset.second", "geometry": geometry_b.model_dump(mode="json")}
    )
    feasibility_a = store.put_json(
        {"source_assembly": assembly_a.model_dump(mode="json"), "status": "FAILED"}
    )
    feasibility_b = store.put_json(
        {"source_assembly": assembly_b.model_dump(mode="json"), "status": "FAILED"}
    )

    first = RefinementCheckpointStore(
        store,
        source_assembly=assembly_a,
        source_feasibility=feasibility_a,
        config={"seed": 1},
        mode="required",
        resume_from=None,
    )
    duplicate = RefinementCheckpointStore(
        store,
        source_assembly=assembly_b,
        source_feasibility=feasibility_b,
        config={"seed": 1},
        mode="required",
        resume_from=None,
    )
    changed = RefinementCheckpointStore(
        store,
        source_assembly=store.put_json(
            {
                "resource_id": "asset.third",
                "geometry": geometry_a.model_dump(mode="json"),
                "revision": 2,
            }
        ),
        source_feasibility=feasibility_a,
        config={"seed": 1},
        mode="required",
        resume_from=None,
    )

    assert first.key == duplicate.key
    assert first.source_identity == duplicate.source_identity
    assert changed.key != first.key


def test_refinement_checkpoint_rejects_concurrent_writers(tmp_path):
    store = LocalArtifactStore(tmp_path / "artifacts")
    assembly = store.put_json({"source": "assembly"})
    feasibility = store.put_json({"source": "feasibility"})
    first = RefinementCheckpointStore(
        store,
        source_assembly=assembly,
        source_feasibility=feasibility,
        config={"seed": 1},
        mode="required",
        resume_from=None,
        lock_writer=True,
    )
    with pytest.raises(MethodExecutionError, match="another articulated collision optimizer"):
        RefinementCheckpointStore(
            store,
            source_assembly=assembly,
            source_feasibility=feasibility,
            config={"seed": 1},
            mode="required",
            resume_from=None,
            lock_writer=True,
        )
    first.close()
    replacement = RefinementCheckpointStore(
        store,
        source_assembly=assembly,
        source_feasibility=feasibility,
        config={"seed": 1},
        mode="required",
        resume_from=None,
        lock_writer=True,
    )
    replacement.close()


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
    assert result["contact_quadrature_probe_count"] == 3 * len(initial[1].faces)
    assert result["surface_p95_m"] < 1e-6
    assert result["bidirectional_sample_max_m"] > 0.015
    assert result["contact_coverage"] == 0
    assert result["worst_cell_coverage"] == 0
    assert result["status"] == "FAILED"


def test_surface_coverage_weights_each_source_triangle_by_area():
    visual = trimesh.creation.box([0.3, 0.02, 0.01])
    patches = bind_surface(visual, [visual.convex_hull])
    audit = SurfaceCoverage(visual, patches, {0}, cell_size_m=0.01, seed=13)

    area_by_face = np.bincount(
        audit.face_ids,
        weights=audit.weights,
        minlength=len(visual.faces),
    )

    np.testing.assert_allclose(area_by_face, visual.area_faces)
    assert audit.weights.sum() == pytest.approx(visual.area)


def test_reverse_surface_error_prioritizes_the_responsible_proxy_patch():
    visual = trimesh.creation.box([0.02, 0.02, 0.002])
    exact = visual.convex_hull
    extended = exact.copy()
    extended.apply_translation([0.01, 0, 0])
    patches = [
        SurfacePatch(extended, np.arange(len(visual.faces))),
        SurfacePatch(exact.copy(), np.empty(0, dtype=int)),
    ]
    audit = SurfaceCoverage(visual, patches, {0}, cell_size_m=0.01, seed=7)

    result, selected, priorities = audit.evaluate_with_priorities(
        patches, max_error_m=0.003, coverage_distance_m=0.002, min_coverage=0.99
    )

    assert result["status"] == "FAILED"
    assert 0 in selected
    assert priorities[0] > 1


def test_disconnected_source_regions_are_split_without_a_bridging_hull():
    left = trimesh.creation.box([0.02, 0.02, 0.002])
    right = left.copy()
    left.apply_translation([-0.1, 0, 0])
    right.apply_translation([0.1, 0, 0])
    visual = trimesh.util.concatenate([left, right])
    source_faces = np.arange(len(visual.faces))
    parent = SurfacePatch(visual.convex_hull, source_faces)

    children = fit_patches(visual, parent, 0.00025)

    assert children is not None and len(children) == 2
    np.testing.assert_array_equal(
        np.sort(np.concatenate([child.faces for child in children])), source_faces
    )
    assert all(child.mesh.extents[0] < 0.03 for child in children)
    assert abs(children[0].mesh.centroid[0] - children[1].mesh.centroid[0]) > 0.15


def test_many_disconnected_regions_are_spatially_balanced_in_one_split():
    components = []
    for index in range(8):
        part = trimesh.creation.box([0.004, 0.004, 0.002])
        part.apply_translation([0.02 * index, 0, 0])
        components.append(part)
    visual = trimesh.util.concatenate(components)
    parent = SurfacePatch(visual.convex_hull, np.arange(len(visual.faces)))

    children = fit_patches(visual, parent, 0.00025)

    assert children is not None and len(children) == 2
    assert [len(child.faces) for child in children] == [48, 48]
    assert all(child.mesh.extents[0] < 0.07 for child in children)


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


def test_one_sided_patch_keeps_source_surface_and_places_skin_on_selected_side():
    visual = trimesh.Trimesh(
        vertices=[[-0.01, -0.01, 0], [0.01, -0.01, 0], [0.01, 0.01, 0], [-0.01, 0.01, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    source_vertices = visual.vertices.copy()
    parent = SurfacePatch(
        trimesh.creation.box([0.02, 0.02, 0.0005]),
        np.arange(len(visual.faces)),
    )

    fitted = fit_one_sided_patch(visual, parent, skin_m=0.00025, polarity=-1)

    assert fitted is not None
    assert fitted.mesh.is_volume and fitted.mesh.is_convex
    assert fitted.mesh.bounds[1, 2] == pytest.approx(0)
    assert fitted.mesh.bounds[0, 2] == pytest.approx(-0.0005)
    np.testing.assert_array_equal(fitted.faces, parent.faces)
    np.testing.assert_array_equal(visual.vertices, source_vertices)
    with pytest.raises(ValueError, match="polarity"):
        fit_one_sided_patch(visual, parent, skin_m=0.00025, polarity=0)


def test_collision_split_inherits_the_current_proxy_envelope():
    from scipy.spatial import ConvexHull

    visual = trimesh.creation.box([0.04, 0.02, 0.01])
    source_vertices = visual.vertices.copy()
    parent_mesh = visual.convex_hull
    parent_mesh.vertices[parent_mesh.vertices[:, 0] > 0, 0] -= 0.002
    parent_mesh = parent_mesh.convex_hull
    parent = SurfacePatch(parent_mesh, np.arange(len(visual.faces)))

    children = _split_with_current_envelope(
        visual,
        parent,
        ArticulatedCollisionRefinementConfig(),
    )

    assert children is not None and len(children) == 2
    np.testing.assert_array_equal(
        np.sort(np.concatenate([child.faces for child in children])), parent.faces
    )
    equations = ConvexHull(parent.mesh.vertices).equations
    for child in children:
        violation = equations[:, :3] @ child.mesh.vertices.T + equations[:, 3, None]
        assert float(violation.max()) <= 1e-7
        assert child.depth == parent.depth + 1
    np.testing.assert_array_equal(visual.vertices, source_vertices)


def test_current_envelope_partition_conserves_volume_and_source_ownership():
    from scipy.spatial import ConvexHull

    visual = trimesh.creation.box([0.04, 0.02, 0.01])
    source_vertices = visual.vertices.copy()
    parent = SurfacePatch(visual.convex_hull, np.arange(len(visual.faces)), depth=3)

    children = _partition_current_envelope(
        visual,
        parent,
        ArticulatedCollisionRefinementConfig(),
    )

    assert children is not None and len(children) == 2
    np.testing.assert_array_equal(
        np.sort(np.concatenate([child.faces for child in children])), parent.faces
    )
    assert sum(child.mesh.volume for child in children) == pytest.approx(
        parent.mesh.volume, rel=5e-5
    )
    equations = ConvexHull(parent.mesh.vertices).equations
    for child in children:
        violation = equations[:, :3] @ child.mesh.vertices.T + equations[:, 3, None]
        assert float(violation.max()) <= 1e-7
        assert child.depth == parent.depth + 1
        assert child.baseline_volume_m3 is None
        assert child.baseline_minimum_thickness_m is None
    np.testing.assert_array_equal(visual.vertices, source_vertices)


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
        {
            "max_local_error_m": 0.0025,
            "surface_fidelity_reserve_m": 0.0006,
            "coverage_distance_m": 0.002,
        },
        {"max_local_error_m": float("nan")},
        {"max_rounds": 0},
        {"surface_query_backend": "cpu", "require_accelerator": True},
        {"gpu_devices": [0, 0]},
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
    assert {candidate["fraction"] for candidate in candidates} == {
        0.0625,
        0.03125,
        0.015625,
    }
    for candidate in candidates:
        proposal = candidate["patches"]
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

    hinted = _shape_refit_candidates(
        asset,
        patches,
        visuals,
        coverage,
        {"collision_pairs": [pair]},
        0.001,
        ArticulatedCollisionRefinementConfig(min_proxy_volume_ratio=0.997),
        fraction_hint=0.03125,
    )
    assert hinted
    assert max(candidate["fraction"] for candidate in hinted) <= 0.046875


def test_support_plane_candidate_keeps_valid_local_cuts_when_one_patch_degenerates(
    monkeypatch,
):
    from types import SimpleNamespace

    import physrealx.integrations.articulated_geometry as articulated_geometry
    from physrealx.domain import GeometryRole, JointType, PoseSpec

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
    original_clip = articulated_geometry.clip_patch_support

    def reject_body(patch, *args, **kwargs):
        if patch is patches["body"][0]:
            return None, {"status": "REJECTED", "reason": "synthetic_degeneracy"}
        return original_clip(patch, *args, **kwargs)

    monkeypatch.setattr(articulated_geometry, "clip_patch_support", reject_body)
    candidates = articulated_geometry._shape_refit_candidates(
        asset,
        patches,
        visuals,
        coverage,
        {"collision_pairs": [pair]},
        0.001,
        ArticulatedCollisionRefinementConfig(min_proxy_volume_ratio=0.997),
    )

    assert candidates
    for candidate in candidates:
        assert candidate["rejected_patch_count"] >= 1
        assert any(value["link"] == "body" for value in candidate["rejected_patches"])
        assert candidate["active_set_rounds"] >= 2
        np.testing.assert_array_equal(
            candidate["patches"]["body"][0].mesh.vertices,
            patches["body"][0].mesh.vertices,
        )
        assert (
            candidate["patches"]["cover"][0].mesh.bounds[0, 0]
            > patches["cover"][0].mesh.bounds[0, 0]
        )
    assert any(candidate["linear_constraint_residual_m"] <= 1e-10 for candidate in candidates)


def test_surface_fidelity_active_set_reroutes_trim_to_opposing_link():
    from types import SimpleNamespace

    from physrealx.domain import GeometryRole, JointType, PoseSpec
    from physrealx.integrations.articulated_geometry import _shape_refit_candidates

    visuals = {
        "body": trimesh.creation.box([0.01, 0.01, 0.01]),
        "cover": trimesh.creation.box([0.01, 0.01, 0.01]),
    }
    patches = {
        link: [SurfacePatch(mesh.convex_hull, np.arange(len(mesh.faces)))]
        for link, mesh in visuals.items()
    }

    class FidelityGate:
        def __init__(self, link):
            self.link = link

        def evaluate_with_priorities(self, proposal, **kwargs):
            changed = not np.array_equal(
                proposal[0].mesh.vertices, patches[self.link][0].mesh.vertices
            )
            failed = self.link == "cover" and changed
            metrics = {
                "status": "FAILED" if failed else "PASSED",
                "bidirectional_sample_max_m": 0.0031 if failed else 0.0,
                "worst_cell_coverage": 1.0,
                "contact_coverage": 1.0,
            }
            return metrics, ({0} if failed else set()), ({0: 1.0} if failed else {})

        def evaluate(self, proposal, **kwargs):
            metrics, selected, _ = self.evaluate_with_priorities(proposal, **kwargs)
            return metrics, selected

    coverage = {link: FidelityGate(link) for link in visuals}
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

    candidates = _shape_refit_candidates(
        asset,
        patches,
        visuals,
        coverage,
        {"collision_pairs": [pair]},
        0.001,
        ArticulatedCollisionRefinementConfig(min_proxy_volume_ratio=0.1),
    )

    assert candidates
    rerouted = next(
        candidate for candidate in candidates if candidate["linear_constraint_residual_m"] <= 1e-10
    )
    assert any(
        value["link"] == "cover" and value["reason"] == "surface_fidelity_gate"
        for value in rerouted["rejected_patches"]
    )
    np.testing.assert_array_equal(
        rerouted["patches"]["cover"][0].mesh.vertices,
        patches["cover"][0].mesh.vertices,
    )
    assert rerouted["patches"]["body"][0].mesh.bounds[1, 0] < patches["body"][0].mesh.bounds[1, 0]


def test_failed_materialized_fidelity_skips_physics_sweep(monkeypatch):
    import physrealx.integrations.articulated_geometry as articulated_geometry

    state = {
        "patches": {"body": [], "cover": []},
        "references": {},
        "asset": object(),
        "fidelity": {
            "body": {"status": "PASSED"},
            "cover": {"status": "FAILED"},
        },
    }

    def unexpected_physics(*args, **kwargs):
        pytest.fail("physics sweep must run only after materialized fidelity passes")

    monkeypatch.setattr(articulated_geometry, "evaluate_asset_feasibility", unexpected_physics)
    result = articulated_geometry._score_materialized_candidate(
        None,
        state,
        None,
        0.001,
        ArticulatedCollisionRefinementConfig(),
    )

    assert result["evaluation"] is None
    assert result["score"] == float("inf")


def test_stalled_shape_progress_escalates_to_topology():
    from physrealx.integrations.articulated_geometry import (
        _shape_progress_requires_topology,
    )

    stalled, improvement, required = _shape_progress_requires_topology(
        "joint_support_plane_refit", 0.006187, 0.006105, 0.00095
    )
    assert stalled
    assert improvement == pytest.approx(0.000082)
    assert required == pytest.approx(0.00026185)

    progressed, _, _ = _shape_progress_requires_topology(
        "joint_support_plane_refit", 0.006187, 0.0058, 0.00095
    )
    assert not progressed
    split, _, _ = _shape_progress_requires_topology(
        "topology_supported_split", 0.006187, 0.00618, 0.00095
    )
    assert not split


def test_shape_candidate_budget_interleaves_current_and_split_topologies():
    def candidate(base, fraction, residual):
        return {
            "base_policy": base,
            "fraction": fraction,
            "linear_constraint_residual_m": residual,
        }

    current_aggressive = candidate("current", 0.06, 4e-6)
    current_low_residual = candidate("current", 0.02, 1e-7)
    current_middle = candidate("current", 0.04, 2e-6)
    split_low_residual = candidate("topology_supported_split", 0.02, 5e-8)

    ordered = _shape_candidate_order(
        [split_low_residual, current_low_residual, current_middle, current_aggressive]
    )

    assert ordered[:3] == [
        current_aggressive,
        split_low_residual,
        current_low_residual,
    ]
    assert sorted(map(id, ordered)) == sorted(
        map(
            id,
            [split_low_residual, current_low_residual, current_middle, current_aggressive],
        )
    )


def test_shape_candidate_budget_keeps_both_current_steps_without_split():
    aggressive = {
        "base_policy": "current",
        "fraction": 0.06,
        "linear_constraint_residual_m": 4e-6,
    }
    low_residual = {
        "base_policy": "current",
        "fraction": 0.02,
        "linear_constraint_residual_m": 1e-7,
    }

    assert _shape_candidate_order([low_residual, aggressive]) == [aggressive, low_residual]


def test_topology_exploration_compares_refit_with_its_raw_split():
    split = {"evaluation": {"max_penetration_m": 0.0022}}
    useful = {
        "policy": "topology_support_plane_refit",
        "evaluation": {"max_penetration_m": 0.0016},
    }
    too_shallow = {
        "policy": "topology_support_plane_refit",
        "evaluation": {"max_penetration_m": 0.00215},
    }
    wrong_topology = {
        "policy": "joint_support_plane_refit",
        "evaluation": {"max_penetration_m": 0.0011},
    }

    assert _topology_refit_exploration_candidates(
        [too_shallow, wrong_topology, useful],
        split,
        current_penetration_m=0.00124,
        optimization_tolerance_m=0.00095,
    ) == [useful]


def test_topology_exploration_rejects_unbounded_metric_jump():
    split = {"evaluation": {"max_penetration_m": 0.003}}
    candidate = {
        "policy": "topology_support_plane_refit",
        "evaluation": {"max_penetration_m": 0.0025},
    }

    assert not _topology_refit_exploration_candidates(
        [candidate],
        split,
        current_penetration_m=0.0012,
        optimization_tolerance_m=0.00095,
    )


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
    assert metrics["trust_region_basis"] == "source_surface_coverage"
    np.testing.assert_array_equal(visual.vertices, [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]])


def test_supported_clip_uses_source_gate_but_unsupported_fragment_keeps_vertex_trust():
    visual = trimesh.Trimesh(
        vertices=[[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    proxy = trimesh.creation.box([0.01, 0.01, 0.0005])
    arguments = {
        "visual": visual,
        "halfspaces": [(np.array([1.0, 0, 0]), 0.003)],
        "trust_region_m": 0.001,
        "min_thickness_ratio": 0.25,
        "min_volume_ratio": 0.5,
    }

    supported, supported_metrics = clip_patch_support(
        SurfacePatch(proxy.copy(), np.array([0])), **arguments
    )
    unsupported, unsupported_metrics = clip_patch_support(
        SurfacePatch(proxy.copy(), np.empty(0, dtype=int)), **arguments
    )

    assert supported is not None
    assert supported_metrics["max_vertex_displacement_m"] == pytest.approx(0.002)
    assert supported_metrics["trust_region_basis"] == "source_surface_coverage"
    assert unsupported is None
    assert unsupported_metrics == {
        "status": "REJECTED",
        "reason": "shape_trust_region",
        "max_vertex_displacement_m": pytest.approx(0.002),
        "trust_region_basis": "unsupported_proxy_vertices",
    }


def test_repeated_support_clips_keep_the_first_volume_and_thickness_baseline():
    visual = trimesh.Trimesh(
        vertices=[[-0.005, -0.005, 0], [0.005, -0.005, 0], [0, 0.005, 0]],
        faces=[[0, 1, 2]],
        process=False,
    )
    original = SurfacePatch(trimesh.creation.box([0.01, 0.01, 0.001]), np.array([0]))

    first, _ = clip_patch_support(
        original,
        visual,
        [(np.array([1.0, 0, 0]), 0.003)],
        trust_region_m=0.001,
        min_thickness_ratio=0.25,
        min_volume_ratio=0.5,
    )
    assert first is not None
    second, metrics = clip_patch_support(
        first,
        visual,
        [(np.array([1.0, 0, 0]), -0.001)],
        trust_region_m=0.001,
        min_thickness_ratio=0.25,
        min_volume_ratio=0.5,
    )

    assert second is None
    assert metrics["reason"] == "minimum_volume_ratio"
    assert metrics["volume_ratio"] == pytest.approx(0.4)
    assert first.baseline_volume_m3 == pytest.approx(original.mesh.volume)


def test_halfspace_polyhedron_ignores_coplanar_interior_points_before_glb_roundtrip():
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
    assert len(mesh.vertices) == 8
    assert len(mesh.faces) == 12
    assert mesh.is_volume and mesh.is_convex and mesh.euler_number == 2
    decoded = trimesh.load(io.BytesIO(mesh.export(file_type="glb")), file_type="glb", process=True)
    geometries = list(decoded.geometry.values())
    assert len(geometries) == 1
    assert geometries[0].is_volume and geometries[0].is_convex


def test_collision_publication_rebuilds_convex_topology_at_glb_precision(tmp_path):
    from physrealx import PhysRealX
    from physrealx.integrations.collision import load_meshes

    px = PhysRealX.open(tmp_path)
    part = trimesh.creation.box([0.04, 0.02, 0.0005])
    vertices = part.vertices.copy()

    reference, order = _publish_parts(px.artifacts, "cover", [part])

    decoded = load_meshes(px.artifacts, reference)
    assert order == [0]
    assert len(decoded) == 1
    assert decoded[0].is_volume and decoded[0].is_convex
    np.testing.assert_array_equal(part.vertices, vertices)


def test_stable_proxy_retriangulates_without_moving_float32_vertices():
    source = trimesh.creation.icosphere(subdivisions=1, radius=0.02).convex_hull
    source_vertices = np.asarray(source.vertices, dtype=np.float32)

    stable = _glb_stable_convex_proxy(source)

    assert stable is not None and stable.is_volume and stable.is_convex
    stable_vertices = np.asarray(stable.vertices, dtype=np.float32)
    assert all(np.any(np.all(source_vertices == vertex, axis=1)) for vertex in stable_vertices)
    np.testing.assert_allclose(stable.bounds, source.bounds, atol=1e-8)


def test_partition_seam_is_measured_against_the_complete_proxy_union():
    visual = trimesh.creation.box([0.04, 0.02, 0.01])
    exact = visual.convex_hull
    upper = exact.copy()
    upper.apply_translation([0, 0, 0.01])
    owned = np.arange(len(visual.faces))
    split = len(owned) // 2
    incomplete = [
        SurfacePatch(upper.copy(), owned[:split]),
        SurfacePatch(upper.copy(), owned[split:]),
    ]
    exact_cover = SurfacePatch(exact, np.empty(0, dtype=int))

    missing_error = _maximum_partition_seam_error(visual, incomplete, {0})
    covered_error = _maximum_partition_seam_error(visual, [*incomplete, exact_cover], {0})

    assert missing_error > 0.005
    assert covered_error < 1e-9
