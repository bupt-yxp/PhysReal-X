"""Nearest-surface queries with explicit accelerator and CPU policies."""

from __future__ import annotations

import os
import threading
import weakref
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

import numpy as np
import trimesh

from physrealx.errors import MethodExecutionError


class SurfaceQuery(Protocol):
    """Read-only closest-surface service used by collision refinement."""

    def nearest(self, mesh: Any, points: Any) -> tuple[Any, Any]: ...

    def signed_distance(self, mesh: Any, points: Any) -> Any: ...

    def describe(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


@dataclass
class CpuSurfaceQuery:
    """Bounded parallel Trimesh queries over one immutable surface."""

    workers: int = 1
    batch_size: int = 4096

    def __post_init__(self) -> None:
        if self.workers < 1:
            raise ValueError("surface query workers must be positive")
        if self.batch_size < 1:
            raise ValueError("surface query batch size must be positive")
        self._pool = (
            ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="surface-query")
            if self.workers > 1
            else None
        )
        self._thread_state = threading.local()

    def _thread_mesh(self, mesh: Any) -> Any:
        """Give every worker its own Rtree; Trimesh spatial indexes are not thread-safe."""
        if self._pool is None:
            return mesh
        cache = cast(
            OrderedDict[int, tuple[Any, tuple[int, int], Any]],
            getattr(self._thread_state, "meshes", OrderedDict()),
        )
        self._thread_state.meshes = cache
        key = id(mesh)
        revision = (hash(mesh.vertices), hash(mesh.faces))
        cached = cache.pop(key, None)
        if cached is not None and cached[0]() is mesh and cached[1] == revision:
            cache[key] = cached
            return cached[2]
        # Proximity queries need topology only. ``Trimesh.copy`` also evaluates
        # lazy colour state, which can be undefined on concatenated collision
        # proxies and must never make a geometry-only query fail.
        copy = trimesh.Trimesh(
            vertices=np.asarray(mesh.vertices).copy(),
            faces=np.asarray(mesh.faces).copy(),
            process=False,
        )
        cache[key] = (weakref.ref(mesh), revision, copy)
        while len(cache) > 4:
            cache.popitem(last=False)
        return copy

    def _query(self, mesh: Any, chunk: Any) -> tuple[Any, Any]:
        local = self._thread_mesh(mesh)
        _, distance, face = trimesh.proximity.closest_point(  # type: ignore[no-untyped-call]
            local, chunk
        )
        return distance, face

    def nearest(self, mesh: Any, points: Any) -> tuple[Any, Any]:
        values = np.asarray(points, dtype=float)
        if not len(values):
            return np.empty(0, dtype=float), np.empty(0, dtype=np.int64)
        chunks = [
            values[start : start + self.batch_size]
            for start in range(0, len(values), self.batch_size)
        ]
        if self._pool is None or len(chunks) == 1:
            result = [self._query(mesh, chunk) for chunk in chunks]
        else:
            # Trimesh/Rtree queries are read-only here. Each task owns its output,
            # and map preserves input ordering for deterministic reductions.
            result = list(self._pool.map(lambda chunk: self._query(mesh, chunk), chunks))
        return (
            np.concatenate([value[0] for value in result]),
            np.concatenate([value[1] for value in result]),
        )

    def signed_distance(self, mesh: Any, points: Any) -> Any:
        values = np.asarray(points, dtype=float)
        if not len(values):
            return np.empty(0, dtype=float)
        chunks = [
            values[start : start + self.batch_size]
            for start in range(0, len(values), self.batch_size)
        ]

        def query(chunk: Any) -> Any:
            # Trimesh uses positive distance for points inside a watertight mesh.
            return trimesh.proximity.signed_distance(  # type: ignore[no-untyped-call]
                self._thread_mesh(mesh), chunk
            )

        if self._pool is None or len(chunks) == 1:
            result = [query(chunk) for chunk in chunks]
        else:
            result = list(self._pool.map(query, chunks))
        return np.concatenate(result)

    def describe(self) -> dict[str, Any]:
        return {
            "backend": "trimesh_cpu",
            "devices": ["cpu"],
            "workers": self.workers,
            "batch_size": self.batch_size,
            "accelerated": False,
        }

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)


def create_surface_query(
    *,
    backend: Literal["auto", "cpu", "warp_cuda"],
    gpu_devices: tuple[int, ...],
    require_accelerator: bool,
    cpu_workers: int,
    batch_size: int,
) -> SurfaceQuery:
    """Create one query engine; accelerator requirements never fall back silently."""
    if backend != "cpu":
        try:
            from physrealx.integrations._warp_surface_query import WarpSurfaceQuery

            available = WarpSurfaceQuery.available_devices()
            selected = tuple(f"cuda:{index}" for index in gpu_devices) if gpu_devices else available
            if selected:
                missing = sorted(set(selected).difference(available))
                if missing:
                    raise MethodExecutionError(
                        f"requested CUDA surface-query devices are unavailable: {missing}"
                    )
                return cast(SurfaceQuery, WarpSurfaceQuery(selected, batch_size=batch_size))
        except ImportError as error:
            if backend == "warp_cuda" or require_accelerator:
                raise MethodExecutionError(
                    "Warp CUDA surface queries require the physical GPU dependencies"
                ) from error
        except MethodExecutionError:
            raise
        except Exception as error:
            if backend == "warp_cuda" or require_accelerator:
                raise MethodExecutionError(
                    f"Warp CUDA surface-query initialization failed: {error}"
                ) from error
        if backend == "warp_cuda" or require_accelerator:
            raise MethodExecutionError(
                "CUDA surface queries were required but no CUDA device is visible"
            )

    workers = cpu_workers or min(12, max(1, (os.cpu_count() or 1) // 2))
    return CpuSurfaceQuery(workers=workers, batch_size=batch_size)
