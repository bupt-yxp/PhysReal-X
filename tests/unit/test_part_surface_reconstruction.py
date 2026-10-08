import numpy as np
import pytest
import torch

try:
    import trimesh
except ModuleNotFoundError as exc:
    if exc.name == "trimesh":
        pytest.skip("Install the physical extra for mesh tests", allow_module_level=True)
    raise

from physrealx.integrations._part_remeshing import (
    manifold_patches,
    refine_observed_crossings,
    restricted_delaunay,
    retriangulate_observed_states,
)
from physrealx.integrations._state_surfaces import MotionConsistency, reconstruct_states
from physrealx.integrations._surface_query import CpuSurfaceQuery
from physrealx.integrations.video_articulation import FramePoints, VideoArticulationRefinementConfig
from physrealx.reconstruction import joint_transform


def transform(points, q, axis, origin):
    q = torch.as_tensor(q, dtype=points.dtype)
    d = points - origin
    return (
        origin
        + d * q.cos()
        + torch.linalg.cross(axis.expand_as(d), d) * q.sin()
        + (d * axis).sum(-1, keepdim=True) * axis * (1 - q.cos())
    )


def test_rdel_preserves_attributes_and_includes_hull_duals():
    source = trimesh.creation.icosphere(subdivisions=1, radius=0.03)
    source.visual.vertex_colors = np.random.default_rng(9).integers(
        0, 255, (len(source.vertices), 4)
    )
    before = source.vertices.copy()
    result, report = restricted_delaunay(source, CpuSurfaceQuery())
    assert report["hull_rays_included"]
    np.testing.assert_array_equal(result.vertices, before)
    np.testing.assert_array_equal(result.visual.vertex_colors, source.visual.vertex_colors)
    assert result.is_watertight
    assert np.isclose(result.volume, source.volume)


def test_video_supported_resampling_keeps_source_surface_and_requires_two_fit_frames():
    source = trimesh.Trimesh(
        vertices=[[0, 0, 0], [0.02, 0, 0], [0.02, 0.02, 0], [0, 0.02, 0]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    source.visual.vertex_colors = np.tile([30, 80, 120, 255], (4, 1))
    original_vertices, original_faces = source.vertices.copy(), source.faces.copy()
    pixel = np.array([[0.01, 0, 0]])
    query = CpuSurfaceQuery()
    unchanged, rejected = refine_observed_crossings(
        source, [pixel], pixel, query, edge_length_m=0.005, support_radius_m=0.001
    )
    assert rejected["status"] == "NO_SUPPORTED_EDGES"
    np.testing.assert_array_equal(unchanged.faces, original_faces)
    result, report = refine_observed_crossings(
        source,
        [pixel, pixel.copy()],
        pixel,
        query,
        edge_length_m=0.005,
        support_radius_m=0.001,
    )
    assert report["status"] == "APPLIED" and report["added_vertices"] == 1
    assert report["heldout_used"] is False and report["source_surface_changed"] is False
    np.testing.assert_allclose(result.vertices[-1], pixel[0])
    np.testing.assert_allclose(result.area, source.area)
    np.testing.assert_array_equal(result.visual.vertex_colors[-1], [30, 80, 120, 255])
    np.testing.assert_array_equal(source.vertices, original_vertices)
    np.testing.assert_array_equal(source.faces, original_faces)


def test_video_resampling_does_not_borrow_support_from_nearby_disconnected_sheet():
    source = trimesh.Trimesh(
        vertices=[
            [0, 0, 0],
            [0.02, 0, 0],
            [0.02, 0.02, 0],
            [0, 0.02, 0],
            [0.0095, -0.0005, 0.0005],
            [0.0105, -0.0005, 0.0005],
            [0.01, 0.0005, 0.0005],
        ],
        faces=[[0, 1, 2], [0, 2, 3], [4, 5, 6]],
        process=False,
    )
    measured = np.array([[0.01, 0, 0.0005]])
    result, report = refine_observed_crossings(
        source,
        [measured, measured.copy()],
        np.array([[0.01, 0, 0]]),
        CpuSurfaceQuery(),
        edge_length_m=0.005,
        support_radius_m=0.001,
    )
    assert report["status"] == "NO_SUPPORTED_EDGES"
    assert len(result.vertices) == len(source.vertices)


def test_video_retriangulation_preserves_shared_vertices_and_rejects_missing_surface(monkeypatch):
    from physrealx.integrations import _part_remeshing

    vertices = np.array([[0, 0, 0], [0.02, 0, 0], [0.02, 0.02, 0], [0, 0.02, 0]], dtype=float)
    source = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    source.visual.vertex_colors = np.array(
        [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255], [255, 255, 0, 255]]
    )
    states = [source.copy(), source.copy()]
    states[0].vertices[2, 2] = 0.0001
    states[1].vertices[2, 2] = 0.0002
    original_faces = source.faces.copy()

    def propose(mesh, query, tolerance, **kwargs):
        candidate = mesh.copy()
        candidate.faces = np.array([[0, 1, 3], [1, 2, 3]])
        return candidate, {"coverage_certified": True}

    monkeypatch.setattr(_part_remeshing, "_restricted_delaunay_once", propose)
    rebuilt, fitted, report = retriangulate_observed_states(
        source, states, CpuSurfaceQuery(), tolerance=0.001
    )
    assert report["status"] == "APPLIED"
    np.testing.assert_array_equal(source.faces, original_faces)
    np.testing.assert_array_equal(rebuilt.vertices, source.vertices)
    np.testing.assert_array_equal(rebuilt.visual.vertex_colors, source.visual.vertex_colors)
    assert all(np.array_equal(item.faces, rebuilt.faces) for item in fitted)
    assert all(
        row["source_to_candidate"]["status"] == "CERTIFIED" for row in report["coverage"].values()
    )

    from physrealx.integrations import _mesh_validity

    original_intersections = _mesh_validity.self_intersections
    monkeypatch.setattr(
        _mesh_validity,
        "self_intersections",
        lambda mesh: (
            bool(np.isclose(mesh.vertices[2, 2], 0.00015, atol=1e-9))
            or original_intersections(mesh)
        ),
    )
    _, _, fallback = retriangulate_observed_states(
        source, states, CpuSurfaceQuery(), tolerance=0.001
    )
    assert fallback["status"] == "APPLIED"
    assert fallback["proposal_basis"] == "state_0"
    assert fallback["attempts"][0] == {
        "basis": "mean",
        "reason": "invalid_proposal_surface",
    }
    monkeypatch.setattr(_mesh_validity, "self_intersections", original_intersections)

    unsupported = [states[0].copy(), states[1].copy()]
    unsupported[1].vertices[2, 2] = 0.01
    same, _, no_video_support = retriangulate_observed_states(
        source, unsupported, CpuSurfaceQuery(), tolerance=0.001
    )
    assert same is source
    assert no_video_support["reason"] == "state_1_surface_coverage"

    def remove_half(mesh, query, tolerance, **kwargs):
        candidate = mesh.copy()
        candidate.faces = np.array([[0, 1, 2]])
        return candidate, {"coverage_certified": True}

    monkeypatch.setattr(_part_remeshing, "_restricted_delaunay_once", remove_half)
    original, unchanged, rejected = retriangulate_observed_states(
        source, states, CpuSurfaceQuery(), tolerance=0.001
    )
    assert original is source and unchanged is states
    assert rejected["status"] == "REJECTED"


def test_video_local_remeshing_uses_bounded_resolution_only_at_crossing(monkeypatch):
    from physrealx.integrations import _part_remeshing

    source = trimesh.Trimesh(
        vertices=[[0, 0, 0], [0.02, 0, 0], [0.02, 0.02, 0], [0, 0.02, 0.005]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    original_faces = source.faces.copy()

    def propose(mesh, query, tolerance, **kwargs):
        candidate = mesh.copy()
        candidate.faces = np.array([[0, 1, 3], [1, 2, 3]])
        return candidate, {"coverage_certified": True}

    monkeypatch.setattr(_part_remeshing, "_restricted_delaunay_once", propose)
    rebuilt, states, report = retriangulate_observed_states(
        source,
        [source.copy(), source.copy()],
        CpuSurfaceQuery(),
        focus_points=source.vertices[[0, 2]],
        local_tolerance=0.003,
    )
    assert report["status"] == "APPLIED"
    assert report["coverage_budget_m"] == 0.003
    assert report["region"] == "local_2_vertices"
    assert any(
        row["reason"] == "source_surface_coverage" and row["coverage_budget_m"] == 0.001
        for row in report["attempts"]
    )
    np.testing.assert_array_equal(source.faces, original_faces)
    assert all(np.array_equal(state.faces, rebuilt.faces) for state in states)


def test_video_retriangulation_retries_only_crossing_cavities(monkeypatch):
    from physrealx.integrations import _part_remeshing

    vertices = np.array([[0, 0, 0], [0.02, 0, 0], [0.02, 0.02, 0], [0, 0.02, 0]])
    source = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    observed = [source.copy(), source.copy()]

    def propose(mesh, query, tolerance, *, focus_vertices=None, **kwargs):
        candidate = mesh.copy()
        if focus_vertices is None:
            return candidate, {
                "coverage_certified": False,
                "source_to_remesh": {"status": "VIOLATION"},
                "remesh_to_source": {"status": "CERTIFIED"},
            }
        candidate.faces = np.array([[0, 1, 3], [1, 2, 3]])
        return candidate, {"coverage_certified": True}

    monkeypatch.setattr(_part_remeshing, "_restricted_delaunay_once", propose)
    result, fitted, report = retriangulate_observed_states(
        source, observed, CpuSurfaceQuery(), focus_points=vertices[[0]]
    )
    assert report["status"] == "APPLIED"
    assert report["region"].startswith("local_")
    assert report["attempts"][0]["reason"] == "proposal_coverage_unresolved"
    assert report["attempts"][0]["forward"]["status"] == "VIOLATION"
    assert all(np.array_equal(item.faces, result.faces) for item in fitted)


def test_manifold_patches_keep_remote_cavity_unchanged():
    vertices = np.array([[0, 0, 0], [0.02, 0, 0], [0.02, 0.02, 0], [0, 0.02, 0]])
    source = trimesh.Trimesh(vertices=vertices, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    candidate = source.copy()
    candidate.faces = np.array([[0, 1, 3], [1, 2, 3]])
    remote, _ = manifold_patches(source, candidate, focus_vertices=np.zeros(4, dtype=bool))
    np.testing.assert_array_equal(remote.faces, source.faces)
    local, report = manifold_patches(
        source, candidate, focus_vertices=np.array([True, False, False, False])
    )
    assert report["accepted_cavities"] == 1
    assert set(map(tuple, np.sort(local.faces, axis=1))) == set(
        map(tuple, np.sort(candidate.faces, axis=1))
    )


def test_observed_cavities_keep_valid_patch_when_another_folds(monkeypatch):
    from physrealx.integrations import _mesh_validity

    square = np.array([[0, 0, 0], [0.02, 0, 0], [0.02, 0.02, 0], [0, 0.02, 0]])
    vertices = np.vstack([square, square + np.array([0.05, 0, 0])])
    source = trimesh.Trimesh(
        vertices=vertices,
        faces=[[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]],
        process=False,
    )
    candidate = source.copy()
    candidate.faces = np.array([[0, 1, 3], [1, 2, 3], [4, 5, 7], [5, 6, 7]])

    def has_fold(mesh):
        faces = set(map(tuple, np.sort(mesh.faces, axis=1)))
        return (0, 1, 3) in faces and (4, 5, 7) in faces

    monkeypatch.setattr(_mesh_validity, "self_intersections", has_fold)
    result, report = manifold_patches(source, candidate, validation_meshes=[source])
    assert report["accepted_cavities"] == 1
    assert report["geometry_rejected_cavities"] == 1
    assert not has_fold(result)
    np.testing.assert_array_equal(source.faces, [[0, 1, 2], [0, 2, 3], [4, 5, 6], [4, 6, 7]])


def test_motion_consistency_uses_observed_second_state_and_inverse_gradients():
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=0.01)
    mesh.apply_translation([0.03, 0, 0])
    motion = {
        "parent": "body",
        "child": "cover",
        "axis": [0, 1, 0],
        "origin_m": [0, 0, 0],
        "range": [0.0, 0.7],
    }
    end = mesh.copy()
    end.apply_transform(joint_transform("hinge", motion["axis"], [0, 0, 0.003], 0.7))
    states = {0.0: {"cover": mesh}, 0.7: {"cover": end}}
    loss = MotionConsistency(
        states, motion, VideoArticulationRefinementConfig(), CpuSurfaceQuery(), "cpu"
    )
    loss.bind({"cover": mesh}, motion)
    vertices = {"cover": torch.tensor(mesh.vertices, requires_grad=True)}
    axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    origin = torch.zeros(3, dtype=torch.float64, requires_grad=True)
    value = loss(vertices, transform, axis, origin)
    value.backward()
    assert value > 0
    assert origin.grad.norm() > 0
    assert vertices["cover"].grad.norm() > 0
    assert {item[2] for item in loss.matches} == {False, True}


def test_independent_state_fit_moves_towards_evidence_without_reading_check():
    a = trimesh.creation.icosphere(subdivisions=1, radius=0.01)
    b = a.copy()
    b.apply_translation([0.04, 0, 0])
    meshes = {"body": a, "cover": b}
    motion = {
        "parent": "body",
        "child": "cover",
        "axis": [0, 1, 0],
        "origin_m": [0, 0, 0],
        "range": [0.0, 0.7],
    }
    observations = {}
    for n, mesh in meshes.items():
        frames = []
        for q in motion["range"]:
            data = mesh.vertices + np.array([0.0003, 0, 0])
            if n == "cover":
                data = trimesh.transform_points(
                    data, joint_transform("hinge", motion["axis"], [0, 0, 0], q)
                )
            frames.append(FramePoints(str(q), q, data))
        observations[n] = {
            "fit": frames,
            "check": [FramePoints("invalid", 0, np.full((3, 3), np.nan))],
        }
    states, report = reconstruct_states(
        meshes, observations, motion, VideoArticulationRefinementConfig(), CpuSurfaceQuery()
    )
    assert not report["heldout_used"]
    assert states[0.0]["body"].centroid[0] > a.centroid[0]
    np.testing.assert_array_equal(a.vertices, meshes["body"].vertices)
