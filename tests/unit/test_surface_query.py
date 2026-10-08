from __future__ import annotations

import numpy as np
import pytest

try:
    import trimesh
except ModuleNotFoundError as exc:
    if exc.name == "trimesh":
        pytest.skip("Install the physical extra for mesh tests", allow_module_level=True)
    raise

from physrealx.integrations._surface_query import CpuSurfaceQuery
from physrealx.integrations._warp_surface_query import WarpSurfaceQuery


def test_parallel_cpu_query_invalidates_mutated_mesh_cache() -> None:
    mesh = trimesh.creation.box()
    query = CpuSurfaceQuery(workers=2, batch_size=1)
    try:
        initial, _ = query.nearest(mesh, np.asarray([[0.5, 0.0, 0.0]] * 2))
        assert np.allclose(initial, 0.0)

        mesh.apply_translation([10.0, 0.0, 0.0])
        translated, _ = query.nearest(mesh, np.asarray([[10.5, 0.0, 0.0]] * 2))
        assert np.allclose(translated, 0.0)
    finally:
        query.close()


def test_warp_query_uses_current_center_and_bounded_device_batches(monkeypatch) -> None:
    mesh = trimesh.creation.box()
    query = WarpSurfaceQuery(("cuda:0", "cuda:1"), batch_size=2)
    calls: list[tuple[str, int, np.ndarray]] = []
    prepared: list[tuple[str, ...]] = []

    def fake_query(
        device: str,
        _key: str,
        _vertices: np.ndarray,
        _faces: np.ndarray,
        center: np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append((device, len(points), center.copy()))
        return points[:, 0].copy(), np.zeros(len(points), dtype=np.int64)

    monkeypatch.setattr(query, "_query_device", fake_query)
    monkeypatch.setattr(
        query,
        "_prepare_meshes",
        lambda devices, _key, _vertices, _faces: prepared.append(devices),
    )
    points = np.arange(21, dtype=float).reshape(7, 3)
    try:
        distances, _ = query.nearest(mesh, points)
        assert np.array_equal(distances, points[:, 0])
        assert {call[0] for call in calls} == {"cuda:0", "cuda:1"}
        assert prepared == [("cuda:0", "cuda:1")]
        assert max(call[1] for call in calls) <= 2
        assert all(np.allclose(call[2], [0.0, 0.0, 0.0]) for call in calls)

        calls.clear()
        prepared.clear()
        mesh.apply_translation([10.0, 0.0, 0.0])
        translated = points + np.asarray([10.0, 0.0, 0.0])
        translated_distances, _ = query.nearest(mesh, translated)
        assert np.array_equal(translated_distances, translated[:, 0])
        assert prepared == [("cuda:0", "cuda:1")]
        assert all(np.allclose(call[2], [10.0, 0.0, 0.0]) for call in calls)
    finally:
        query.close()
