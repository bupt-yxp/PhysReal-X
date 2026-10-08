"""Adaptive surface-supported convex shells with an explicit clearance budget."""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from physrealx.domain import ArtifactRef
from physrealx.errors import DomainValidationError, MethodExecutionError

from ._artifacts import read_json
from ._collision_refit import SurfacePatch, fit_patches
from ._surface_query import create_surface_query
from .collision import load_meshes

_MIN_SHELL_ERROR_M = 8e-5


def clearance_error_budget(result: dict[str, Any], required: float | None = None) -> float:
    sweep = result["swept_surface_clearance"]
    if result["status"] != "PASSED" or not sweep.get("continuous_surface_clearance_certified"):
        raise DomainValidationError(
            "surface separation must pass before collision shell construction"
        )
    required = max(float(result["config"]["minimum_clearance_m"]), required or 0.0)
    gap = float(sweep["minimum_clearance_lower_bound_m"])
    tolerance = min(0.003, (gap - required) / 4)
    if not np.isfinite([gap, required, tolerance]).all() or tolerance < _MIN_SHELL_ERROR_M:
        raise DomainValidationError("insufficient certified clearance for valid collision shells")
    return tolerance


def certified_surface_budget(
    store: Any, result: dict[str, Any], required: float | None = None
) -> tuple[float, dict[str, Any]]:
    """Tighten a loose successful certificate before allocating proxy error.

    The original surface is rechecked; neither geometry nor the physical gap is
    relaxed. A certificate just above the acceptance threshold need not be a
    tight estimate of the available clearance.
    """
    sweep = result["swept_surface_clearance"]
    requested = max(float(result["config"]["minimum_clearance_m"]), required or 0.0)
    gap = sweep["minimum_clearance_lower_bound_m"]
    if (
        result["status"] != "PASSED"
        or not sweep.get("continuous_surface_clearance_certified")
        or not np.isfinite(gap)
        or gap < result["config"]["minimum_clearance_m"]
        or gap >= requested + 4 * _MIN_SHELL_ERROR_M
    ):
        return clearance_error_budget(result, requested), sweep
    from ._surface_clearance import swept_clearance

    motion = result["optimized_motion"]
    meshes = {
        n: load_meshes(store, ArtifactRef.model_validate(ref), process=False)[0]
        for n, ref in result["geometries"].items()
    }
    query = create_surface_query(
        backend="auto", gpu_devices=(), require_accelerator=False, cpu_workers=4, batch_size=65536
    )
    try:
        sweep = swept_clearance(
            meshes[motion["parent"]],
            meshes[motion["child"]],
            motion,
            query,
            clearance=requested + 4 * _MIN_SHELL_ERROR_M + 2e-6,
            samples=361,
            temporal_levels=5,
        )
    finally:
        query.close()
    tightened = {**result, "swept_surface_clearance": sweep}
    return clearance_error_budget(tightened, requested), sweep


def approximation_bound(
    proxy: Any, surface: Any, query: Any, tolerance: float, *, max_cells: int = 50000
) -> dict[str, Any]:
    """Bound *maximum* proxy-to-source error, including face interiors and holes."""
    triangles = np.asarray(proxy.triangles)
    worst, count = 0.0, 0
    error = max(2e-7, 16 * np.finfo(np.float32).eps * max(surface.scale, proxy.scale))
    while len(triangles):
        center = triangles.mean(1)
        radius = np.linalg.norm(triangles - center[:, None], axis=2).max(1)
        distance, ids = query.nearest(surface, center)
        if np.any(ids < 0) or not np.isfinite(distance).all():
            raise MethodExecutionError("invalid collision surface distance")
        if np.any(distance - error > tolerance):
            return {
                "status": "VIOLATION",
                "witness_error_m": float(distance.max()),
                "witness_point_m": center[int(np.argmax(distance))].tolist(),
            }
        upper = distance + radius + error
        # Distance to one convex source triangle is convex: its maximum on
        # a proxy triangle is bounded by its values at the three vertices.
        # This tightens large coplanar cells without assuming the whole mesh
        # is convex or overlooking holes between source triangles.
        target = np.repeat(np.asarray(surface.triangles)[ids], 3, axis=0)
        points = triangles.reshape(-1, 3)
        closest = trimesh.triangles.closest_point(target, points)  # type: ignore[no-untyped-call]
        same_face_bound = np.linalg.norm(points - closest, axis=1).reshape(-1, 3).max(1)
        upper = np.minimum(upper, same_face_bound + error)
        safe = upper <= tolerance
        worst = max(worst, float(upper[safe].max(initial=0)))
        count += len(triangles)
        triangles = triangles[~safe]
        if not len(triangles):
            return {"status": "CERTIFIED", "upper_bound_m": worst, "queries": count}
        if count + 4 * len(triangles) > max_cells:
            return {"status": "UNRESOLVED", "queries": count}
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        triangles = np.concatenate(
            [np.stack(t, axis=1) for t in [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]]
        )
    raise AssertionError("unreachable")


def rebuild_collision_shells(
    store: Any, refinement: ArtifactRef, *, max_parts: int = 8192
) -> tuple[ArtifactRef, ...]:
    result = read_json(store, refinement, "physrealx.video-articulation-refinement/2")
    # Half the remaining gap stays as a reserve. Each of two links consumes at
    # most one quarter; this replaces the unrelated old 3 mm fidelity threshold.
    tolerance, certificate = certified_surface_budget(store, result)
    if not 1 <= max_parts <= 50000:
        raise DomainValidationError("collision part budget must be between 1 and 50000")
    skin = tolerance / 4
    query = create_surface_query(
        backend="auto", gpu_devices=(), require_accelerator=False, cpu_workers=4, batch_size=65536
    )
    reports = []
    try:
        for name, ref in result["geometries"].items():
            reference = ArtifactRef.model_validate(ref)
            visual = load_meshes(store, reference, process=False)[0]
            queue = [SurfacePatch(visual, np.arange(len(visual.faces)))]
            accepted: list[SurfacePatch] = []
            bounds = []
            while queue:
                patch = queue.pop()
                children = fit_patches(visual, patch, skin)
                if not children:
                    raise MethodExecutionError("source-supported patch has no valid convex shell")
                for candidate in children:
                    # Check at the exported precision before accepting any hull.
                    candidate.mesh = trimesh.convex.convex_hull(
                        candidate.mesh.vertices.astype(np.float32).astype(float)
                    )
                    check = approximation_bound(candidate.mesh, visual, query, tolerance)
                    owned_surface = trimesh.Trimesh(
                        vertices=visual.vertices, faces=visual.faces[candidate.faces], process=False
                    )
                    coverage = approximation_bound(owned_surface, candidate.mesh, query, tolerance)
                    if check["status"] == coverage["status"] == "CERTIFIED":
                        accepted.append(candidate)
                        bounds.append(max(check["upper_bound_m"], coverage["upper_bound_m"]))
                    elif len(candidate.faces) > 1:
                        queue.append(candidate)
                    else:
                        raise MethodExecutionError(
                            "single source triangle exceeds collision error budget"
                        )
                if len(queue) + len(accepted) > max_parts:
                    raise MethodExecutionError(
                        "collision part budget exhausted; no source faces were dropped"
                    )
            ownership = np.concatenate([p.faces for p in accepted])
            if not np.array_equal(np.sort(ownership), np.arange(len(visual.faces))):
                raise MethodExecutionError(
                    "collision shells must retain every source triangle exactly once"
                )
            scene = trimesh.Scene()
            for i, patch in enumerate(accepted):
                scene.add_geometry(patch.mesh, geom_name=f"{name}_{i}", node_name=f"{name}_{i}")
            collision = store.put_bytes(
                scene.export(file_type="glb"),  # type: ignore[no-untyped-call]
                media_type="model/gltf-binary",
            )
            decoded = load_meshes(store, collision)
            if len(decoded) != len(accepted) or any(
                not p.is_volume or not p.is_convex for p in decoded
            ):
                raise MethodExecutionError("exported collision hulls failed roundtrip validation")
            reports.append(
                store.put_json(
                    {
                        "schema": "physrealx.collision-build-report/1",
                        "status": "PASSED",
                        "unit": "m",
                        "link": name,
                        "source_geometry": ref,
                        "collision_geometry": collision.model_dump(mode="json"),
                        "method": {"name": "surface_supported_convex_shell", "version": "1.0.0"},
                        "source_refinement": refinement.model_dump(mode="json"),
                        "construction_surface_clearance": certificate,
                        "preprocessing": {
                            "mode": "adaptive_source_face_partition",
                            "skin_m": skin,
                            "source_faces_retained": len(ownership),
                            "source_faces_removed": 0,
                        },
                        "metrics": {
                            "convex_parts": len(accepted),
                            "maximum_surface_error_bound_m": max(bounds),
                            "surface_error_budget_m": tolerance,
                            "source_triangle_ownership_coverage": 1.0,
                        },
                        "gates": {
                            "convex": True,
                            "closed_positive_volume": True,
                            "surface_error": True,
                            "source_face_coverage": True,
                            "roundtrip": True,
                        },
                    }
                )
            )
    finally:
        query.close()
    return tuple(reports)
