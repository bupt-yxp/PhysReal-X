"""Geometric counterexamples: topology, crossing a distance branch, local CCD."""

import numpy as np
import pytest
import torch

try:
    import trimesh
except ModuleNotFoundError as exc:
    if exc.name == "trimesh":
        pytest.skip("Install the physical extra for mesh tests", allow_module_level=True)
    raise

from physrealx.integrations._mesh_validity import (
    collision_mesh,
    deformation_step,
    limit_collision_nodes,
    limit_triangle_area_nodes,
    surface_topology,
)
from physrealx.integrations._part_remeshing import manifold_patches
from physrealx.integrations.video_articulation import (
    FramePoints,
    _resize_metric_depth,
)


def transform(points, q, axis, origin):
    q = torch.as_tensor(q, dtype=points.dtype)
    axis = axis.to(points.dtype)
    d = points - origin
    return (
        origin
        + d * q.cos()
        + torch.linalg.cross(axis.expand_as(d), d) * q.sin()
        + (d * axis).sum(-1, keepdim=True) * axis * (1 - q.cos())
    )


def test_topology_rejects_third_sheet_and_vertex_bowtie_without_repair():
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [1, 1, 1]])
    third = trimesh.Trimesh(vertices, [[0, 1, 2], [1, 0, 3], [0, 1, 4]], process=False)
    report = surface_topology(third)
    assert not report["valid"] and report["nonmanifold_edges"] == 1
    bowtie = trimesh.Trimesh(vertices, [[0, 1, 2], [0, 4, 5]], process=False)
    report = surface_topology(bowtie)
    assert not report["valid"] and report["nonmanifold_edges"] == 0
    assert report["nonmanifold_vertices"] == 1
    duplicate = trimesh.Trimesh(vertices, [[0, 1, 2], [0, 1, 2]], process=False)
    assert not surface_topology(duplicate)["valid"]


def test_rdel_rejects_extra_sheet_without_deleting_source_surface():
    source = trimesh.Trimesh(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]],
        [[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    candidate = source.copy()
    candidate.faces = np.array([[0, 1, 2], [0, 2, 3], [0, 2, 4]])
    result, report = manifold_patches(source, candidate)
    assert report["topology"]["valid"]
    assert report["unmatched_candidate_cavities"] > 0
    np.testing.assert_array_equal(result.vertices, source.vertices)
    assert {tuple(sorted(f)) for f in result.faces} == {tuple(sorted(f)) for f in source.faces}
    assert result.area == source.area


def test_sparse_depth_decimation_retains_measurements_and_preserves_unknown_regions():
    depth = np.zeros((8, 8), dtype=np.float32)
    depth[::2, ::2] = 1.2
    depth[:2, :2] = 0
    depth[0, 0], depth[0, 1] = np.nan, np.inf
    reduced = _resize_metric_depth(depth, (4, 4))
    assert reduced[0, 0] == 0
    assert np.isfinite(reduced).all()
    np.testing.assert_allclose(reduced.ravel()[1:], 1.2)


def test_rgb_supervision_is_not_disabled_when_depth_is_missing():
    from physrealx.integrations._video_render_loss import VideoRenderLoss

    body = trimesh.Trimesh(
        [[-0.01, -0.01, 1], [0.01, -0.01, 1], [0, 0.01, 1]], [[0, 1, 2]], process=False
    )
    body.visual.vertex_colors = np.tile([0, 0, 0, 255], (3, 1))
    cover = body.copy()
    cover.apply_translation([3, 0, 0])
    meshes = dict(body=body, cover=cover)
    motion = dict(parent="body", child="cover", range=[0, 0.5])
    values = []
    for color in [0, 255]:
        image = (
            np.full((32, 32, 3), color, np.uint8),
            np.zeros((32, 32)),
            np.ones((32, 32), bool),
            np.eye(4),
            np.array([500, 500, 16, 16]),
        )
        observations = dict(body={"fit": [FramePoints("rgb_only", 0, np.empty((0, 3)), image)]})
        renderer = VideoRenderLoss(meshes, observations, "cpu", motion)
        vertices = {n: torch.tensor(m.vertices, requires_grad=True) for n, m in meshes.items()}
        loss = renderer(
            vertices,
            transform,
            torch.tensor([1.0, 0, 0], dtype=torch.float64),
            torch.zeros(3),
            "cover",
            0,
        )
        loss.backward()
        assert vertices["body"].grad.isfinite().all()
        values.append(loss.item())
    assert values[1] > values[0] + 0.5


def test_local_ccd_preserves_unrelated_motion_and_checks_the_full_step():
    lower = np.array([[-0.01, -0.01, 0], [0.01, -0.01, 0], [0, 0.01, 0]])
    vertices = np.concatenate(
        [lower, lower + np.array([0, 0, 0.001]), lower + np.array([0.05, 0, 0])]
    )
    mesh = trimesh.Trimesh(vertices, [[0, 1, 2], [3, 4, 5], [6, 7, 8]], process=False)
    old = torch.zeros((9, 3), dtype=torch.float64)
    delta = old.clone()
    delta[3:6, 2] = -0.002
    delta[6:, 0] = 0.002
    graph = dict(
        base=torch.tensor(vertices),
        delta=delta,
        idx=torch.arange(9)[:, None],
        weights=torch.ones((9, 1)),
    )
    ipc = collision_mesh(mesh)
    report = limit_collision_nodes(ipc, graph, old)
    assert report["local_nodes"] > 0
    assert report["fallback_scale"] == 1
    torch.testing.assert_close(delta[6:, 0], torch.full((3,), 0.002, dtype=torch.float64))
    assert deformation_step(ipc, vertices, vertices + delta.numpy()) == 1


def test_near_fold_swept_witness_does_not_freeze_a_separate_patch():
    # Two incident faces from a failing local update: the blocking nonincident
    # edges are only 0.6 micrometres apart. A near-impact spatial query missed
    # this witness even though continuous CCD correctly restricted the step.
    folded = (
        np.array(
            [
                [0.10722503066062927, -0.3820713609457016, 0.2394006005488336],
                [-1.1619441211223602, -1.471085473895073, -0.5374781903810799],
                [0.766020268201828, 2.025419846177101, -0.03164453664794564],
                [0.28869882225990295, -0.1722630113363266, 0.32972212648019195],
            ]
        )
        / 1000
    )
    change = (
        np.array(
            [
                [0.5000040531152883, -0.4999966023170044, -0.25565061174908416],
                [-0.5000040471982575, -0.49999659706567723, -0.2909247785523412],
                [0, 0, 0],
                [-0.4999966023397362, -0.500004053113276, -0.34872790167130363],
            ]
        )
        / 1000
    )
    vertices = np.vstack([folded, [[0.03, 0, 0], [0.04, 0, 0], [0.03, 0.01, 0]]])
    mesh = trimesh.Trimesh(vertices, [[3, 2, 1], [1, 2, 0], [4, 5, 6]], process=False)
    previous = torch.zeros((7, 3), dtype=torch.float64)
    delta = torch.tensor(np.vstack([change, np.tile([0.002, 0, 0], (3, 1))]))
    graph = dict(
        base=torch.tensor(vertices),
        delta=delta,
        idx=torch.arange(7)[:, None],
        weights=torch.ones((7, 1)),
    )
    ipc = collision_mesh(mesh)
    assert deformation_step(ipc, vertices, vertices + delta.numpy()) < 0.01
    report = limit_collision_nodes(ipc, graph, previous)
    assert report["local_nodes"] == 4
    assert report["fallback_scale"] == 1.0
    torch.testing.assert_close(delta[4:, 0], torch.full((3,), 0.002, dtype=torch.float64))
    assert deformation_step(ipc, vertices, vertices + delta.numpy()) == 1.0


def test_orientation_checks_intermediate_collapse_despite_valid_endpoint():
    base = torch.tensor([[0.0, 0, 0], [1.0, 0, 0], [0.0, 1, 0]], dtype=torch.float64)
    previous = torch.zeros_like(base)
    delta = -2 * base
    graph = dict(
        base=base,
        delta=delta,
        faces=torch.tensor([[0, 1, 2]]),
        idx=torch.arange(3)[:, None],
        weights=torch.ones((3, 1)),
    )
    report = limit_triangle_area_nodes(graph, previous)
    assert report["backtracked"] > 0
    for t in np.linspace(0, 1, 101):
        triangle = (base + t * delta).numpy()
        assert np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])[2] > 0.1


def test_a_valid_triangle_can_unfold_by_rotating_its_normal_through_180_degrees():
    base = torch.tensor([[0.0, 0, 0], [0.01, 0, 0], [0.0, 0.01, 0]], dtype=torch.float64)
    delta = torch.zeros_like(base)
    graph = dict(
        base=base,
        delta=delta,
        faces=torch.tensor([[0, 1, 2]]),
        idx=torch.arange(3)[:, None],
        weights=torch.ones((3, 1)),
    )
    for angle in np.linspace(0, np.pi, 19)[1:]:
        previous = delta.clone()
        matrix = torch.tensor(
            [[1.0, 0, 0], [0, np.cos(angle), -np.sin(angle)], [0, np.sin(angle), np.cos(angle)]],
            dtype=torch.float64,
        )
        proposed = base @ matrix.T - base
        delta.copy_(proposed)
        report = limit_triangle_area_nodes(graph, previous)
        assert report["backtracked"] == 0
        torch.testing.assert_close(delta, proposed)


def test_independent_image_checks_detect_missing_geometry_and_color_mismatch():
    from physrealx.integrations._video_render_loss import check_images

    body = trimesh.Trimesh(
        [[-0.01, -0.01, 1], [0.01, -0.01, 1], [0, 0.01, 1]], [[0, 1, 2]], process=False
    )
    body.visual.vertex_colors = np.tile([0, 0, 0, 255], (3, 1))
    cover = body.copy()
    cover.apply_translation([3, 0, 0])
    motion = dict(
        parent="body", child="cover", range=[0, 0.5], axis=[1.0, 0, 0], origin_m=[0.0, 0, 0]
    )
    image = (
        np.full((32, 32, 3), 255, np.uint8),
        np.ones((32, 32)),
        np.ones((32, 32), bool),
        np.eye(4),
        np.array([500, 500, 16, 16]),
    )
    obs = dict(body={"fit": [], "check": [FramePoints("heldout", 0, np.empty((0, 3)), image)]})
    measured = check_images(dict(body=body, cover=cover), obs, motion, "cpu")[0]
    assert measured["rgb_mae"] == 1.0
    assert measured["depth_p90_m"] < 1e-6
    assert measured["overlap_pixels"] > 0
    body.apply_translation([3, 0, 0])
    missing = check_images(dict(body=body, cover=cover), obs, motion, "cpu")[0]
    assert missing["mask_iou"] == 0
    assert missing["rgb_mae"] is None and missing["depth_p90_m"] is None


def test_link_rendering_respects_other_parts_for_hard_and_soft_visibility():
    from physrealx.integrations._video_render_loss import VideoRenderLoss, check_images

    body = trimesh.Trimesh(
        [[-0.1, -0.1, 1], [0.1, -0.1, 1], [0.1, 0.1, 1], [-0.1, 0.1, 1]],
        [[0, 1, 2], [0, 2, 3]],
        process=False,
    )
    cover = body.copy()
    cover.vertices *= 0.8  # Same projection, entirely in front of the body.
    body.visual.vertex_colors = np.tile([255, 0, 0, 255], (4, 1))
    cover.visual.vertex_colors = np.tile([0, 255, 0, 255], (4, 1))
    meshes = dict(body=body, cover=cover)
    motion = dict(
        parent="body", child="cover", range=[0, 0.4], axis=[1.0, 0, 0], origin_m=[0.0, 0, 0]
    )
    image = (
        np.zeros((32, 32, 3), np.uint8),
        np.full((32, 32), 0.8),
        np.ones((32, 32), bool),
        np.eye(4),
        np.array([100, 100, 15.3, 15.1]),
    )
    frame = FramePoints("occluded", 0, np.empty((0, 3)), image)
    obs = {n: {"fit": [frame], "check": [frame]} for n in meshes}
    renderer = VideoRenderLoss(meshes, obs, "cpu", motion)
    vertices = {n: torch.tensor(m.vertices, requires_grad=True) for n, m in meshes.items()}
    for soft in (False, True):
        pix = renderer._raster(vertices, renderer.frames[0], soft=soft)[0]
        assert (pix[0, 10:20, 10:20] < 0).all()
        pix = renderer._raster(vertices, renderer.frames[1], soft=soft)[0]
        assert ((pix[0, 10:20, 10:20, 0] >= 0) & (pix[0, 10:20, 10:20, 0] < 2)).all()
    rows = check_images(meshes, obs, motion, "cpu")
    assert rows[0]["overlap_pixels"] == 0
    assert rows[0]["depth_p90_m"] is None
    assert rows[1]["depth_p90_m"] < 1e-6


def test_dense_sloped_surface_keeps_interior_pixels_and_metric_depth():
    from physrealx.integrations._video_render_loss import check_images

    # Subpixel triangles have blurred neighbours closer to the camera than the
    # actual covering face. Visibility must still match the analytic plane.
    side = 81
    x, y = np.meshgrid(np.linspace(-0.18, 0.18, side), np.linspace(-0.18, 0.18, side))
    vertices = np.column_stack([x.ravel(), y.ravel(), (1 + 0.2 * x).ravel()])
    a = (np.arange(side - 1)[:, None] * side + np.arange(side - 1)).ravel()
    faces = np.vstack(
        [np.column_stack([a, a + 1, a + side]), np.column_stack([a + 1, a + side + 1, a + side])]
    )
    body = trimesh.Trimesh(vertices, faces, process=False)
    body.visual.vertex_colors = np.tile([128, 64, 192, 255], (len(vertices), 1))
    cover = body.copy()
    cover.apply_translation([3, 0, 0])
    u, v = np.meshgrid(np.arange(64), np.arange(64))
    # Offset the principal point so this visibility test does not depend on
    # ambiguous exact pixel/edge coincidences along the tessellation grid.
    depth = 1 / (1 - 0.2 * (u - 31.3) / 100)
    mask = (np.abs((u - 31.3) / 100 * depth) < 0.18) & (np.abs((v - 30.7) / 100 * depth) < 0.18)
    image = (
        np.broadcast_to(np.array([128, 64, 192], np.uint8), (64, 64, 3)),
        depth,
        mask,
        np.eye(4),
        np.array([100, 100, 31.3, 30.7]),
    )
    motion = dict(
        parent="body", child="cover", range=[0, 0.5], axis=[1.0, 0, 0], origin_m=[0.0, 0, 0]
    )
    obs = dict(body={"fit": [], "check": [FramePoints("plane", 0, np.empty((0, 3)), image)]})
    measured = check_images(dict(body=body, cover=cover), obs, motion, "cpu")[0]
    assert measured["mask_iou"] > 0.99
    # Float rasterization may omit a few pixels exactly on shared edges.
    assert measured["metric_pixels"] >= 0.99 * mask.sum()
    assert measured["depth_p90_m"] < 1e-6
    assert measured["rgb_mae"] < 1e-6


def test_many_distinct_impact_times_preserve_an_unrelated_surface_patch():
    vertices, faces, increments = [], [], []
    for i in range(12):
        lower = np.array([[0.03 * i, 0, 0], [0.03 * i + 0.01, 0, 0], [0.03 * i, 0.01, 0]])
        upper = lower + np.array([0, 0, 1e-6 * 2.3**i])
        offset = len(vertices)
        vertices.extend(np.vstack([lower, upper]))
        faces.extend([[offset, offset + 1, offset + 2], [offset + 3, offset + 4, offset + 5]])
        increments.extend(np.vstack([np.zeros((3, 3)), np.tile([0, 0, -0.02], (3, 1))]))
    offset = len(vertices)
    vertices.extend([[1, 0, 0], [1.01, 0, 0], [1, 0.01, 0]])
    faces.append([offset, offset + 1, offset + 2])
    increments.extend(np.tile([0.002, 0, 0], (3, 1)))
    vertices = np.array(vertices)
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    old = torch.zeros_like(torch.tensor(vertices))
    delta = torch.tensor(np.array(increments))
    graph = dict(
        base=torch.tensor(vertices),
        delta=delta,
        idx=torch.arange(len(vertices))[:, None],
        weights=torch.ones((len(vertices), 1)),
    )
    ipc = collision_mesh(mesh)
    report = limit_collision_nodes(ipc, graph, old)
    assert report["fallback_scale"] == 1.0
    torch.testing.assert_close(delta[-3:, 0], torch.full((3,), 0.002, dtype=torch.float64))
    assert deformation_step(ipc, vertices, vertices + delta.numpy()) == 1.0
