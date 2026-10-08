"""Independent gradient and initially-crossed surface tests."""

import numpy as np
import pytest

from physrealx.integrations._surface_untangling import (
    SurfaceUntangling,
    gaussian_energy,
    minkowski_gap,
    surface_quadrature,
)
from physrealx.integrations._triangle_contacts import swept_constraints

torch = pytest.importorskip("torch")
trimesh = pytest.importorskip("trimesh")


def test_gaussian_gradient_matches_finite_difference_and_repels_both_parts():
    a = torch.tensor([[0.0, 0, 0], [0.002, 0, 0]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[0.0, 0, 0.001], [0.001, 0, 0.002]], dtype=torch.float64, requires_grad=True)
    pairs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    areas = torch.full((2,), 1e-6, dtype=a.dtype)

    def energy(x, y):
        return gaussian_energy(x, y, pairs, areas, areas, 0.005)

    assert torch.autograd.gradcheck(energy, (a, b), eps=1e-7, atol=1e-5)
    loss = energy(a, b)
    loss.backward()
    torch.testing.assert_close(a.grad.sum(0) + b.grad.sum(0), torch.zeros(3, dtype=a.dtype))
    assert a.grad[:, 2].sum() > 0 and b.grad[:, 2].sum() < 0
    assert energy(a - 1e-5 * a.grad, b - 1e-5 * b.grad) < loss


def test_minkowski_crossing_coplanarity_and_gradients():
    a = torch.tensor([[[-0.02, -0.02, 0], [0.02, -0.02, 0], [0, 0.02, 0]]], dtype=torch.float64)
    b = torch.tensor(
        [[[-0.007, -0.01, -0.003], [0.008, -0.009, 0.002], [0.001, 0.011, 0.004]]], dtype=a.dtype
    )
    a.requires_grad_()
    b.requires_grad_()
    assert minkowski_gap(a, b).item() < 0
    assert torch.autograd.gradcheck(minkowski_gap, (a, b), eps=1e-8, atol=1e-5)
    assert minkowski_gap(a, b + torch.tensor([0, 0, 0.02])).item() > 0
    coplanar = a.detach().clone().requires_grad_()
    loss = torch.relu(0.0002 - minkowski_gap(a, coplanar)).square().sum()
    assert loss > 0
    loss.backward()
    assert torch.isfinite(a.grad).all() and torch.isfinite(coplanar.grad).all()
    assert torch.linalg.vector_norm(coplanar.grad) > 0
    assert minkowski_gap(a - a.grad, coplanar - coplanar.grad).item() > 0.0001


def test_spawn_pose_workers_match_threaded_contact_results():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    body = trimesh.creation.box(extents=[0.04, 0.02, 0.002])
    cover = body.copy()
    cover.apply_translation([0.005, 0, 0.001])
    motion = dict(axis=[0, 1.0, 0], origin_m=[0, 0, 0])
    args = (body, cover, motion, [0.0, 0.2, 0.5])
    expected = swept_constraints(*args, margin=0.001, budget=100, workers=1)
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        actual = swept_constraints(*args, margin=0.001, budget=100, executor=pool)
    for key, value in expected.items():
        np.testing.assert_array_equal(actual[key], value)
    assert actual["intersecting_pairs"] > 0


def test_quadrature_integrates_large_triangles_without_changing_geometry():
    mesh = trimesh.Trimesh([[0, 0, 0], [0.1, 0, 0], [0, 0.1, 0]], [[0, 1, 2]], process=False)
    source = mesh.vertices.copy()
    faces, bary, area = surface_quadrature(mesh, 0.005)
    samples = (mesh.vertices[faces] * bary[..., None]).sum(1)
    assert len(samples) > 100
    assert area.sum() == pytest.approx(mesh.area)
    np.testing.assert_allclose((samples * area[:, None]).sum(0) / area.sum(), mesh.centroid)
    assert bary.min() >= 0
    np.testing.assert_array_equal(mesh.vertices, source)


def test_batched_self_ccd_covers_ipc_crossings_and_matches_serial():
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    import ipctk

    from physrealx.integrations._mesh_validity import collision_mesh, self_collision_witnesses

    points = np.array([[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0]], dtype=float)
    before = np.r_[points, points + np.array([0.001, 0.001, 0.01])]
    after = before.copy()
    after[3:, 2] = -0.01
    mesh = collision_mesh(trimesh.Trimesh(before, [[0, 1, 2], [3, 4, 5]], process=False))
    candidates = ipctk.Candidates()
    candidates.build(mesh, before, after)
    edges, faces = np.asarray(mesh.edges), np.asarray(mesh.faces)
    expected = []
    for family, split in (
        (candidates.fv_candidates, 1),
        (candidates.ee_candidates, 2),
        (candidates.ev_candidates, 1),
        (candidates.vv_candidates, 1),
    ):
        for stencil in family:
            ids = np.asarray(stencil.vertex_ids(edges, faces))
            ids = ids[ids >= 0]
            hit, time = stencil.ccd(before[ids].ravel(), after[ids].ravel(), tmax=1.0)
            if hit:
                expected.append((tuple(ids[:split]), tuple(ids[split:]), time))
    assert expected
    with ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        result = self_collision_witnesses(mesh, before, after, executor=pool)
    actual = [(tuple(a), tuple(b), time) for a, b, time in result]
    # These planar trajectories have their first exact contact at t=0.5.
    # IPC and our intervals may choose different conservative stopping times.
    assert {(a, b) for a, b, _ in expected} <= {(a, b) for a, b, _ in actual}
    assert all(0 <= time < 0.5 for _a, _b, time in actual)
    serial = self_collision_witnesses(mesh, before, after)
    assert actual == [(tuple(a), tuple(b), time) for a, b, time in serial]


@pytest.mark.parametrize("offset", [0.0, 1e3])
def test_bounded_ccd_detects_coplanar_tunnelling_with_clear_endpoints(offset):
    from physrealx.integrations._mesh_validity import _separated_prefix

    triangle = [[-0.01, -0.01, 0], [0.01, -0.01, 0], [0, 0.01, 0]]
    before = np.array([[[-0.02, 0, 0], *triangle]], dtype=float) + offset
    after = before.copy()
    after[0, 0, 0] += 0.04
    time = _separated_prefix(before, after, 1)[0]
    assert 0 <= time < 0.375  # enters triangle at x=-0.005
    # A genuinely separated tangent path can use the complete interval.
    before[:, 0, 2] += 0.0002
    after[:, 0, 2] += 0.0002
    assert _separated_prefix(before, after, 1)[0] == 1


def test_bounded_ccd_tiny_gaps_fail_closed_and_detect_endpoint_contact():
    from physrealx.integrations._mesh_validity import _separated_prefix

    triangle = [[-0.01, -0.01, 0], [0.01, -0.01, 0], [0, 0.01, 0]]
    before = np.array([[[0, 0, 1e-15], *triangle]], dtype=float)
    after = before.copy()
    after[0, 0, 2] = -1e-15
    assert _separated_prefix(before, after, 1)[0] == 0
    before[0, 0, 2] = 0.001
    after[0, 0, 2] = 0
    assert 0 < _separated_prefix(before, after, 1)[0] < 1


@pytest.mark.parametrize("kind", ["vv", "ev", "ee"])
def test_bounded_ccd_certifies_prefix_before_analytic_feature_crossing(kind):
    from physrealx.integrations._mesh_validity import _separated_prefix

    if kind == "vv":
        before = np.array([[[-0.01, 0, 0], [0.01, 0, 0]]])
        after = -before
        split = 1
    elif kind == "ev":
        before = np.array([[[-0.02, 0, 0], [0, -0.01, 0], [0, 0.01, 0]]])
        after = before.copy()
        after[0, 0, 0] = 0.02
        split = 1
    else:
        before = np.array([[[-0.01, 0, 0], [0.01, 0, 0], [0, -0.01, 0.01], [0, 0.01, 0.01]]])
        after = before.copy()
        after[0, 2:, 2] = -0.01
        split = 2
    assert 0 < _separated_prefix(before, after, split)[0] < 0.5
    # Certify the entire safe first quarter using the independent analytic time.
    assert _separated_prefix(before, before + 0.25 * (after - before), split)[0] == 1


@pytest.mark.parametrize("devices", [["cpu", "cpu"], ["cuda:0", "cuda:1"]])
def test_parallel_pose_energy_and_joint_gradients_match_serial(devices):
    from types import SimpleNamespace

    if devices[0].startswith("cuda") and torch.cuda.device_count() < 2:
        pytest.skip("Two GPUs required for the CUDA reduction check")
    meshes = {
        name: trimesh.creation.icosphere(subdivisions=0, radius=0.003) for name in ("body", "cover")
    }
    meshes["cover"].apply_translation([0.001, 0.002, 0.003])
    motion = dict(parent="body", child="cover", axis=[1.0, 0, 0], origin_m=[0, 0, 0])
    config = SimpleNamespace(
        surface_support_m=0.003, minimum_clearance_m=0.0002, contact_reserve_m=0.0005
    )
    poses = [0.0, 0.2, 0.5]
    contacts = swept_constraints(
        meshes["body"], meshes["cover"], motion, poses, margin=0.002, budget=100
    )

    def evaluate(devices):
        objective = SurfaceUntangling(meshes, motion, config, devices)
        try:
            objective.bind(meshes, motion, contacts, poses)
            vertices = {
                n: torch.tensor(m.vertices, dtype=torch.float64, requires_grad=True)
                for n, m in meshes.items()
            }
            axis = torch.tensor([1.0, 0, 0], dtype=torch.float64, requires_grad=True)
            origin = torch.zeros(3, dtype=torch.float64, requires_grad=True)

            def transform(points, q, axis, origin):
                axis = axis / torch.linalg.vector_norm(axis)
                d = points - origin
                return (
                    origin
                    + d * np.cos(q)
                    + torch.linalg.cross(axis.expand_as(d), d) * np.sin(q)
                    + (d * axis).sum(-1, keepdim=True) * axis * (1 - np.cos(q))
                )

            terms = objective(vertices, axis, origin, transform)
            loss = sum(terms)
            loss.backward()
            assert torch.linalg.vector_norm(origin.grad) > 0
            return loss.detach(), [v.grad for v in [*vertices.values(), axis, origin]]
        finally:
            objective.close()

    reference = evaluate(["cpu"])
    actual = evaluate(devices)
    torch.testing.assert_close(actual[0], reference[0])
    for left, right in zip(actual[1], reference[1], strict=True):
        torch.testing.assert_close(left, right)
