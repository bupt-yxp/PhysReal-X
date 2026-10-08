from unittest.mock import patch

import numpy as np
import pytest

try:
    import trimesh
except ModuleNotFoundError as exc:
    if exc.name == "trimesh":
        pytest.skip("Install the physical extra for mesh tests", allow_module_level=True)
    raise

from physrealx.integrations._surface_clearance import (
    conforming_refine,
    surface_distance_bound,
    swept_clearance,
)
from physrealx.integrations._surface_query import CpuSurfaceQuery


def plane(z=0):
    return trimesh.Trimesh(
        vertices=[[-0.01, -0.01, z], [0.01, -0.01, z], [0.01, 0.01, z], [-0.01, 0.01, z]],
        faces=[[0, 1, 2], [0, 2, 3]],
        process=False,
    )


def test_refine_preserves_source_surface_boundaries_and_colors():
    source = plane()
    source.visual.vertex_colors = np.tile([50, 100, 150, 255], (4, 1))
    refined, report = conforming_refine(source, np.array([0, 1.0, 0]), np.zeros(3), 0.04, 0.006, 3)
    assert report["added_vertices"] > 0
    assert np.all(refined.vertices[:, 2] == 0)
    assert np.all(refined.visual.vertex_colors == [50, 100, 150, 255])
    assert np.isclose(refined.area, source.area)
    # Boundary edges remain on the original boundary; no interior T-junctions.
    counts = np.bincount(refined.edges_unique_inverse)
    edge = refined.vertices[refined.edges_unique[counts == 1]]
    assert np.all(np.any(np.all(np.isclose(abs(edge), 0.01), axis=1), axis=1))
    assert not refined.is_watertight


def test_open_surface_separation_is_certified_and_crossing_is_rejected():
    query = CpuSurfaceQuery()
    source = plane()
    separated = plane(0.002)
    assert surface_distance_bound(source, separated, query, 0.0002)["status"] == "CERTIFIED"
    crossing = plane()
    crossing.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    assert surface_distance_bound(source, crossing, query, 0.0002)["status"] == "VIOLATION"
    assert (
        surface_distance_bound(source, separated, query, 0.0002, max_depth=0)["status"]
        == "UNRESOLVED"
    )


def test_continuous_sweep_detects_collision_between_clear_endpoints():
    source = plane()
    moving = plane()
    moving.apply_translation([0.02, 0, 0])
    motion = {"axis": [0, 1.0, 0], "origin_m": [0.01, 0, 0], "range": [-np.pi / 2, np.pi / 2]}
    result = swept_clearance(
        source,
        moving,
        motion,
        CpuSurfaceQuery(),
        clearance=0.0002,
        samples=3,
        temporal_levels=1,
        max_cells=30000,
    )
    assert result["status"] == "FAILED"
    assert not result["continuous_surface_clearance_certified"]


def test_a_measured_violation_is_terminal_but_covers_the_full_interval():
    motion = {"axis": [0, 1.0, 0], "origin_m": [0, 0, 0], "range": [0.0, 0.7]}
    witness = {"status": "VIOLATION", "lower_bound_m": -0.001, "upper_bound_m": 0.0001}
    with patch(
        "physrealx.integrations._surface_clearance.surface_distance_bound", return_value=witness
    ) as distance:
        result = swept_clearance(
            plane(),
            plane(),
            motion,
            CpuSurfaceQuery(),
            clearance=0.0002,
            samples=3,
            temporal_levels=8,
        )
    assert distance.call_count == 2
    assert result["status"] == "FAILED" and result["violating_intervals"] == 2
    assert [r["interval"] for r in result["intervals"]] == [[0.0, 0.35], [0.35, 0.7]]


def test_closed_containment_fails_despite_positive_surface_distance():
    outer = trimesh.creation.box([0.04, 0.04, 0.04])
    inner = trimesh.creation.box([0.004, 0.004, 0.004])
    motion = {"axis": [0, 1.0, 0], "origin_m": [0, 0, 0], "range": [0.0, 0.2]}
    for invert in (False, True):
        if invert:
            outer.invert()
        result = swept_clearance(
            outer,
            inner,
            motion,
            CpuSurfaceQuery(),
            clearance=0.0002,
            samples=3,
        )
        assert result["minimum_clearance_lower_bound_m"] > 0.0002
        assert result["solid_containment_detected"] and result["status"] == "FAILED"


def test_local_hinge_bounds_include_every_point_of_the_rotation_arc():
    source = plane(0.004)
    target = plane()
    query = CpuSurfaceQuery()
    axis, origin, half = np.array([1.0, 0.0, 0.0]), np.zeros(3), 0.05
    result = surface_distance_bound(
        source, target, query, 0.0002, hinge_motion=(axis, origin, half)
    )
    assert result["status"] == "CERTIFIED"
    for q in np.linspace(-half, half, 21):
        moved = source.copy()
        moved.apply_transform(trimesh.transformations.rotation_matrix(q, axis, origin))
        distances, _ = query.nearest(target, np.vstack([moved.vertices, moved.triangles_center]))
        assert distances.min() >= result["lower_bound_m"]


def test_local_motion_bound_does_not_certify_a_mid_interval_crossing():
    source = plane(0.0003)
    target = plane()
    result = surface_distance_bound(
        source,
        target,
        CpuSurfaceQuery(),
        0.0002,
        max_depth=4,
        hinge_motion=(np.array([1.0, 0.0, 0.0]), np.zeros(3), 0.1),
    )
    assert result["status"] != "CERTIFIED"


def test_subdivided_clearance_witness_retains_original_face_identity():
    source = plane()
    target = trimesh.Trimesh(
        vertices=[[-0.0095, 0.0085, 0.0001], [-0.0085, 0.0085, 0.0001], [-0.009, 0.0095, 0.0001]],
        faces=[[0, 1, 2]],
        process=False,
    )
    result = surface_distance_bound(source, target, CpuSurfaceQuery(), 0.0002)
    assert result["status"] == "VIOLATION"
    assert result["level"] > 0
    assert result["source_face"] == 1 and result["target_face"] == 0
    point = np.array([result["witness"]])
    closest = trimesh.triangles.closest_point(source.triangles[[result["source_face"]]], point)
    np.testing.assert_allclose(closest, point, atol=1e-12)
