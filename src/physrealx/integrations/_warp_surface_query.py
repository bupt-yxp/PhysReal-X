# mypy: ignore-errors
"""Warp CUDA implementation of unsigned closest-surface queries.

Warp kernel annotations are a runtime DSL and are not valid static Python types.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import warp as wp

# Warp kernels are device-scoped and may run concurrently.  Mesh allocation and
# destruction use process-global native allocator state in Warp 1.17, so build
# or evict cached meshes serially before dispatching work to device threads.
_MESH_CACHE_LOCK = threading.Lock()


@wp.kernel
def _nearest_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    maximum_distance: float,
    distances: wp.array(dtype=wp.float32),
    faces: wp.array(dtype=wp.int32),
):
    index = wp.tid()
    point = points[index]
    query = wp.mesh_query_point_no_sign(mesh_id, point, maximum_distance)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distances[index] = wp.length(point - closest)
        faces[index] = query.face
    else:
        distances[index] = maximum_distance
        faces[index] = -1


@wp.kernel
def _signed_distance_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    maximum_distance: float,
    distances: wp.array(dtype=wp.float32),
):
    index = wp.tid()
    point = points[index]
    query = wp.mesh_query_point_sign_parity(mesh_id, point, maximum_distance)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        # Warp reports a negative sign inside; the platform convention is positive.
        distances[index] = -query.sign * wp.length(point - closest)
    else:
        distances[index] = -maximum_distance


class WarpSurfaceQuery:
    """Shard independent point queries across visible CUDA devices."""

    def __init__(
        self,
        devices: tuple[str, ...],
        *,
        batch_size: int,
        cache_entries_per_device: int = 8,
    ) -> None:
        if not devices:
            raise ValueError("at least one CUDA device is required")
        self.devices = devices
        self.batch_size = batch_size
        self.cache_entries_per_device = cache_entries_per_device
        if batch_size < 1:
            raise ValueError("surface query batch size must be positive")
        self._cache: dict[str, OrderedDict[str, Any]] = {
            device: OrderedDict() for device in devices
        }
        self._pool = ThreadPoolExecutor(
            max_workers=len(devices), thread_name_prefix="warp-surface-query"
        )

    @staticmethod
    def available_devices() -> tuple[str, ...]:
        return tuple(device.alias for device in wp.get_cuda_devices())

    @staticmethod
    def _fingerprint(mesh: Any) -> tuple[str, np.ndarray, np.ndarray, np.ndarray]:
        vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
        faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
        center = (
            (vertices.min(axis=0) + vertices.max(axis=0)) * np.float32(0.5)
            if len(vertices)
            else np.zeros(3, dtype=np.float32)
        )
        centered = np.ascontiguousarray(vertices - center, dtype=np.float32)
        digest = hashlib.blake2b(digest_size=16)
        digest.update(memoryview(centered))
        digest.update(memoryview(faces))
        return digest.hexdigest(), centered, faces, center

    def _mesh(
        self,
        device: str,
        key: str,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> Any:
        with _MESH_CACHE_LOCK:
            cache = self._cache[device]
            cached = cache.pop(key, None)
            if cached is not None:
                cache[key] = cached
                return cached
            with wp.ScopedDevice(device):
                value = wp.Mesh(
                    points=wp.array(vertices, dtype=wp.vec3, device=device),
                    indices=wp.array(faces.reshape(-1), dtype=wp.int32, device=device),
                )
            cache[key] = value
            while len(cache) > self.cache_entries_per_device:
                cache.popitem(last=False)
            return value

    def _prepare_meshes(
        self,
        devices: tuple[str, ...],
        key: str,
        vertices: np.ndarray,
        faces: np.ndarray,
    ) -> None:
        """Materialize device meshes serially before parallel kernel launches."""
        for device in devices:
            self._mesh(device, key, vertices, faces)

    def _query_device(
        self,
        device: str,
        key: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        center: np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        mesh = self._mesh(device, key, vertices, faces)
        local_points = np.ascontiguousarray(points - center, dtype=np.float32)
        maximum_distance = float(
            max(
                1.0,
                np.linalg.norm(local_points, axis=1).max(initial=0.0)
                + np.linalg.norm(np.ptp(vertices, axis=0)),
            )
        )
        with wp.ScopedDevice(device):
            query = wp.array(local_points, dtype=wp.vec3, device=device)
            distances = wp.empty(len(local_points), dtype=wp.float32, device=device)
            triangle_ids = wp.empty(len(local_points), dtype=wp.int32, device=device)
            wp.launch(
                _nearest_kernel,
                dim=len(local_points),
                inputs=[mesh.id, query, maximum_distance],
                outputs=[distances, triangle_ids],
                device=device,
            )
            wp.synchronize_device(device)
            return distances.numpy().astype(float), triangle_ids.numpy().astype(np.int64)

    def _signed_query_device(
        self,
        device: str,
        key: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        center: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        mesh = self._mesh(device, key, vertices, faces)
        local_points = np.ascontiguousarray(points - center, dtype=np.float32)
        maximum_distance = float(
            max(
                1.0,
                np.linalg.norm(local_points, axis=1).max(initial=0.0)
                + np.linalg.norm(np.ptp(vertices, axis=0)),
            )
        )
        with wp.ScopedDevice(device):
            query = wp.array(local_points, dtype=wp.vec3, device=device)
            distances = wp.empty(len(local_points), dtype=wp.float32, device=device)
            wp.launch(
                _signed_distance_kernel,
                dim=len(local_points),
                inputs=[mesh.id, query, maximum_distance],
                outputs=[distances],
                device=device,
            )
            wp.synchronize_device(device)
            return distances.numpy().astype(float)

    def _nearest_shard(
        self,
        device: str,
        key: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        center: np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run bounded batches serially on one device."""
        values = [
            self._query_device(
                device,
                key,
                vertices,
                faces,
                center,
                points[start : start + self.batch_size],
            )
            for start in range(0, len(points), self.batch_size)
        ]
        return (
            np.concatenate([value[0] for value in values]),
            np.concatenate([value[1] for value in values]),
        )

    def _signed_shard(
        self,
        device: str,
        key: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        center: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        """Run bounded signed-distance batches serially on one device."""
        return np.concatenate(
            [
                self._signed_query_device(
                    device,
                    key,
                    vertices,
                    faces,
                    center,
                    points[start : start + self.batch_size],
                )
                for start in range(0, len(points), self.batch_size)
            ]
        )

    def nearest(self, mesh: Any, points: Any) -> tuple[Any, Any]:
        values = np.asarray(points, dtype=float)
        if not len(values):
            return np.empty(0, dtype=float), np.empty(0, dtype=np.int64)
        key, vertices, faces, center = self._fingerprint(mesh)
        count = min(len(self.devices), len(values))
        active_devices = self.devices[:count]
        self._prepare_meshes(active_devices, key, vertices, faces)
        chunks = np.array_split(np.arange(len(values)), count)
        futures = [
            self._pool.submit(
                self._nearest_shard,
                self.devices[index],
                key,
                vertices,
                faces,
                center,
                values[indices],
            )
            for index, indices in enumerate(chunks)
        ]
        result = [future.result() for future in futures]
        return (
            np.concatenate([value[0] for value in result]),
            np.concatenate([value[1] for value in result]),
        )

    def signed_distance(self, mesh: Any, points: Any) -> Any:
        values = np.asarray(points, dtype=float)
        if not len(values):
            return np.empty(0, dtype=float)
        if not mesh.is_watertight or not mesh.is_winding_consistent:
            raise ValueError("signed Warp queries require a watertight consistently wound mesh")
        key, vertices, faces, center = self._fingerprint(mesh)
        count = min(len(self.devices), len(values))
        active_devices = self.devices[:count]
        self._prepare_meshes(active_devices, key, vertices, faces)
        chunks = np.array_split(np.arange(len(values)), count)
        futures = [
            self._pool.submit(
                self._signed_shard,
                self.devices[index],
                key,
                vertices,
                faces,
                center,
                values[indices],
            )
            for index, indices in enumerate(chunks)
        ]
        return np.concatenate([future.result() for future in futures])

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "warp_cuda",
            "devices": list(self.devices),
            "workers": len(self.devices),
            "batch_size": self.batch_size,
            "accelerated": True,
            "precision": "float32_centered",
        }

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        with _MESH_CACHE_LOCK:
            self._cache.clear()
