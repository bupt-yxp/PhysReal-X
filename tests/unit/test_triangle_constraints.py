"""Independent triangle-interior and coplanar intersection predicates."""

import numpy as np
import pytest

try:
    import trimesh
except ModuleNotFoundError as exc:
    if exc.name == "trimesh":
        pytest.skip("Install the physical extra for mesh tests", allow_module_level=True)
    raise

from physrealx.integrations._triangle_contacts import separating_axes, triangle_pairs


def test_separating_axes_detect_interior_crossing_and_coplanar_separation():
    a = np.array([[[-1.0, -1, 0], [1, -1, 0], [0, 1, 0]]])
    b = np.array([[[0.0, -0.5, -1], [0, -0.5, 1], [0, 0.5, 0]]])
    gap, _ = separating_axes(a, b)
    assert gap[0] < 0
    gap, _ = separating_axes(a, a + np.array([3, 0, 0]))
    assert gap[0] > 0


def test_active_budget_never_omits_an_existing_intersection():
    import trimesh

    from physrealx.integrations._triangle_contacts import swept_constraints

    a = np.array([[-0.01, -0.01, 0], [0.01, -0.01, 0], [0, 0.01, 0]])
    b = np.array([[0, -0.005, -0.01], [0, -0.005, 0.01], [0, 0.005, 0]])
    faces = np.arange(9).reshape(3, 3)
    parent = trimesh.Trimesh(
        np.vstack([a + np.array([x, 0, 0]) for x in [0, 0.1, 0.2]]), faces, process=False
    )
    child = trimesh.Trimesh(
        np.vstack([b + np.array([x, 0, 0]) for x in [0, 0.1, 0.2]]), faces, process=False
    )
    r = swept_constraints(
        parent, child, dict(axis=[1, 0, 0], origin_m=[0, 0, 0]), [0.0], margin=0.001, budget=1
    )
    assert r["intersecting_pairs"] == 3
    assert len(r["parent_faces"]) == 3


@pytest.mark.parametrize(
    "left,right,expected",
    [
        ([[0, 0, 0], [-1, 0, 0], [0, -1, 0]], [[1, 2, 1], [2, 2, 1], [1, 3, 1]], np.sqrt(6)),
        ([[-1, 0, 0], [1, 0, 0], [0, -1, 0]], [[0, 2, 1], [0.2, 3, 1], [-0.2, 3, 1]], np.sqrt(5)),
    ],
)
def test_closest_vertex_and_edge_features_supply_tight_separation(left, right, expected):
    gap, _ = separating_axes(np.array([left], float), np.array([right], float))
    assert gap[0] == pytest.approx(expected)


def test_parallel_broad_phase_matches_all_pairs_with_mixed_triangle_sizes():
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(21)
    triangles = [
        rng.normal(size=(n, 1, 3)) * 0.01
        + rng.normal(size=(n, 3, 3)) * 10 ** rng.uniform(-5, -1, size=(n, 1, 1))
        for n in (31, 43)
    ]
    meshes = [
        trimesh.Trimesh(v.reshape(-1, 3), np.arange(v.size // 3).reshape(-1, 3), process=False)
        for v in triangles
    ]
    i, j = np.indices((31, 43)).reshape(2, -1)
    distances, _ = separating_axes(triangles[0][i], triangles[1][j])
    expected = set(zip(i[distances < 0.0002], j[distances < 0.0002], strict=True))

    def query(_):
        a, b, _, _ = triangle_pairs(*meshes, margin=0.0002)
        return set(zip(a, b, strict=True))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(query, range(16)))
    assert all(result == expected for result in results)
