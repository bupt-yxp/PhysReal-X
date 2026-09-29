"""Surface-supported convex patch fitting and local coverage diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from physrealx.integrations.collision import _exterior_samples


def nearest(mesh: Any, points: Any) -> tuple[Any, Any]:
    """Bound temporary proximity-query memory for dense reconstructed surfaces."""
    distances, faces = [], []
    for chunk in np.array_split(points, max(1, int(np.ceil(len(points) / 512)))):
        _, distance, face = trimesh.proximity.closest_point(mesh, chunk)  # type: ignore[no-untyped-call]
        distances.append(distance)
        faces.append(face)
    return np.concatenate(distances), np.concatenate(faces)


@dataclass
class SurfacePatch:
    mesh: Any
    faces: Any
    depth: int = 0


def bind_surface(visual: Any, parts: list[Any]) -> list[SurfacePatch]:
    """Every source triangle has one owner; refinement never drops supported faces."""
    proxy = trimesh.util.concatenate(parts)
    owner = np.repeat(np.arange(len(parts)), [len(part.faces) for part in parts])
    _, closest = nearest(proxy, visual.triangles_center)
    assigned = owner[closest]
    return [SurfacePatch(part, np.flatnonzero(assigned == i)) for i, part in enumerate(parts)]


def fit_patches(visual: Any, patch: SurfacePatch, skin_m: float) -> list[SurfacePatch] | None:
    """Bisect source face sets and fit convex shells, or reject the whole split.

    The fixed skin gives observed open/planar surfaces a numerically valid collision
    volume. It is never reduced in response to penetration. Unsupported old proxy
    fragments can disappear only subject to the independent full-surface gates.
    A numerically degenerate child rejects the subdivision so no source faces vanish.
    """
    ids = patch.faces
    if not len(ids):
        return []
    if len(ids) == 1:
        groups = [ids]
    else:
        centers = visual.triangles_center[ids]
        dimension = int(np.argmax(np.ptp(centers, axis=0)))
        order = np.argsort(centers[:, dimension], kind="stable")
        groups = list(np.array_split(ids[order], 2))
    result = []
    for group in groups:
        triangles = visual.triangles[group]
        normals = visual.face_normals[group]
        points = np.concatenate(
            [triangles + normals[:, None, :] * skin_m, triangles - normals[:, None, :] * skin_m]
        ).reshape(-1, 3)
        points = np.unique(points.astype(np.float32).astype(np.float64), axis=0)
        if len(points) < 4 or np.linalg.matrix_rank(points[1:] - points[0], tol=1e-10) < 3:
            return None
        try:
            mesh = trimesh.convex.convex_hull(points)
        except (ValueError, RuntimeError):
            return None
        if not mesh.is_volume or not mesh.is_convex:
            return None
        result.append(SurfacePatch(mesh, group, patch.depth + 1))
    return result


class SurfaceCoverage:
    """Fixed probes and fixed contact regions, independent of candidate geometry."""

    def __init__(
        self,
        visual: Any,
        patches: list[SurfacePatch],
        contact_parts: set[int],
        *,
        cell_size_m: float,
        seed: int,
    ) -> None:
        self.visual = visual
        random, faces = trimesh.sample.sample_surface(visual, 4096, seed=seed)
        self.points = np.concatenate([visual.vertices, visual.triangles_center, random])
        vertex_faces = np.maximum(visual.vertex_faces[:, 0], 0)
        self.face_ids = np.concatenate([vertex_faces, np.arange(len(visual.faces)), faces])
        self.contact_faces = (
            np.concatenate([p.faces for i, p in enumerate(patches) if i in contact_parts])
            if contact_parts
            else np.empty(0, dtype=int)
        )
        self.contact = np.isin(self.face_ids, self.contact_faces)
        _, self.cells = np.unique(
            np.floor(self.points / cell_size_m).astype(np.int64), axis=0, return_inverse=True
        )
        self.seed = seed

    def evaluate(
        self,
        patches: list[SurfacePatch],
        *,
        max_error_m: float,
        coverage_distance_m: float,
        min_coverage: float,
    ) -> tuple[dict[str, Any], set[int]]:
        parts = [p.mesh for p in patches]
        proxy = trimesh.util.concatenate(parts)
        distance, _ = nearest(proxy, self.points)
        # Exclude buried convex partition faces when checking reverse fidelity.
        outside = _exterior_samples(parts, proxy, self.seed, include_vertices=True)
        reverse, _ = nearest(self.visual, outside)
        covered = distance <= coverage_distance_m
        cell_coverage = np.bincount(self.cells, weights=covered) / np.bincount(self.cells)
        contact_distance = distance[self.contact]
        bad = (distance > max_error_m) | (cell_coverage[self.cells] < min_coverage)
        bad_faces = set(self.face_ids[bad].tolist())
        selected = {i for i, p in enumerate(patches) if bad_faces.intersection(p.faces)}
        if np.any(reverse > max_error_m):
            _, closest = nearest(proxy, outside[reverse > max_error_m])
            owners = np.repeat(np.arange(len(parts)), [len(p.faces) for p in parts])
            selected.update(owners[closest].tolist())
        contact_coverage = float(np.mean(covered[self.contact])) if np.any(self.contact) else 1.0
        maximum = max(float(distance.max()), float(reverse.max()))
        passed = (
            maximum <= max_error_m
            and float(cell_coverage.min()) >= min_coverage
            and contact_coverage >= min_coverage
        )
        return {
            "status": "PASSED" if passed else "FAILED",
            "probe_count": len(distance),
            "contact_probe_count": len(contact_distance),
            "cell_count": len(cell_coverage),
            "bidirectional_sample_max_m": maximum,
            "surface_p95_m": max(
                float(np.quantile(distance, 0.95)), float(np.quantile(reverse, 0.95))
            ),
            "surface_p99_m": max(
                float(np.quantile(distance, 0.99)), float(np.quantile(reverse, 0.99))
            ),
            "surface_coverage": float(np.mean(covered)),
            "worst_cell_coverage": float(cell_coverage.min()),
            "contact_coverage": contact_coverage,
            "contact_sample_max_m": float(contact_distance.max()) if len(contact_distance) else 0.0,
            "uncovered_source_faces": len(bad_faces),
        }, selected


def _halfspace_polyhedron(points: Any) -> Any:
    """Triangulate a bounded half-space intersection without moving its vertices."""
    from scipy.spatial import ConvexHull

    # GLB stores positions as float32. Fit the hull at that exact precision so
    # serialization cannot merge vertices behind the validation layer's back.
    points = np.unique(np.asarray(points, dtype=np.float32).astype(np.float64), axis=0)
    center = points.mean(axis=0)
    scale = max(float(np.ptp(points, axis=0).max()), 1e-12)
    normalized = (points - center) / scale
    hull = ConvexHull(normalized, qhull_options="Qt Qx")
    facets: dict[tuple[float, ...], tuple[Any, set[int]]] = {}
    for simplex, equation in zip(hull.simplices, hull.equations, strict=True):
        key = tuple(np.round(equation, 9).tolist())
        if key not in facets:
            facets[key] = (equation[:3], set())
        facets[key][1].update(int(index) for index in simplex)
    faces: list[list[int]] = []
    for normal, indices in facets.values():
        ids = np.asarray(sorted(indices), dtype=int)
        unit = normal / np.linalg.norm(normal)
        seed = np.eye(3)[int(np.argmin(np.abs(unit)))]
        basis_x = np.cross(unit, seed)
        basis_x /= np.linalg.norm(basis_x)
        basis_y = np.cross(unit, basis_x)
        facet_points = normalized[ids]
        centered = facet_points - facet_points.mean(axis=0)
        coordinates = np.column_stack([centered @ basis_x, centered @ basis_y])
        boundary = ids[ConvexHull(coordinates).vertices]
        for position in range(1, len(boundary) - 1):
            face = [int(boundary[0]), int(boundary[position]), int(boundary[position + 1])]
            triangle = normalized[face]
            cross = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            if float(cross @ unit) < 0:
                face[1], face[2] = face[2], face[1]
            faces.append(face)
    used = np.asarray(sorted({index for face in faces for index in face}), dtype=int)
    remap = np.full(len(points), -1, dtype=int)
    remap[used] = np.arange(len(used))
    return trimesh.Trimesh(
        vertices=points[used],
        faces=remap[np.asarray(faces, dtype=int)],
        process=False,
    )


def clip_patch_support(
    patch: SurfacePatch,
    visual: Any,
    halfspaces: list[tuple[Any, float]],
    *,
    trust_region_m: float,
    min_thickness_ratio: float,
    min_volume_ratio: float,
) -> tuple[SurfacePatch | None, dict[str, Any]]:
    """Intersect one convex proxy with contact planes under hard shape gates.

    The original convex hull contributes immutable half-spaces; contact planes may
    only remove unsupported proxy volume. A Chebyshev-center linear program finds a
    strict interior point before half-space intersection. Source geometry is read
    only to enforce the original surface-normal thickness.
    """
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

    vertices = np.asarray(patch.mesh.vertices, dtype=float)
    try:
        original_hull = ConvexHull(vertices)
    except QhullError:
        return None, {"status": "REJECTED", "reason": "invalid_original_hull"}
    equations = [np.asarray(row, dtype=float) for row in original_hull.equations]
    for normal, bound in halfspaces:
        direction = np.asarray(normal, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            continue
        equations.append(np.r_[direction / norm, -float(bound) / norm])
    unique_planes: dict[tuple[float, float, float], Any] = {}
    for equation in equations:
        normal = equation[:3]
        norm = float(np.linalg.norm(normal))
        normalized = np.r_[normal / norm, float(equation[3]) / norm]
        key = tuple(np.round(normalized[:3], 7).tolist())
        previous = unique_planes.get(key)
        # For equal outward normals, the larger offset is the tighter inequality.
        if previous is None or normalized[3] > previous[3]:
            unique_planes[key] = normalized
    matrix = np.asarray(list(unique_planes.values()))
    normals, offsets = matrix[:, :3], matrix[:, 3]
    norms = np.linalg.norm(normals, axis=1)
    program = linprog(
        np.array([0.0, 0.0, 0.0, -1.0]),
        A_ub=np.column_stack([normals, norms]),
        b_ub=-offsets,
        bounds=[(None, None), (None, None), (None, None), (1e-9, None)],
        method="highs",
    )
    if not program.success or float(program.x[3]) <= 1e-9:
        return None, {"status": "REJECTED", "reason": "empty_halfspace_interior"}
    interior = program.x[:3]
    coordinate_scale = max(float(np.ptp(vertices, axis=0).max()), 1e-9)
    normalized_matrix = np.column_stack([normals * coordinate_scale, normals @ interior + offsets])
    try:
        normalized_intersection = HalfspaceIntersection(
            normalized_matrix, np.zeros(3)
        ).intersections
    except (QhullError, ValueError):
        try:
            normalized_intersection = HalfspaceIntersection(
                normalized_matrix, np.zeros(3), qhull_options="QJ"
            ).intersections
        except (QhullError, ValueError) as error:
            return None, {
                "status": "REJECTED",
                "reason": "halfspace_intersection_failure",
                "detail": str(error).splitlines()[0],
            }
    intersection = interior + coordinate_scale * normalized_intersection
    if len(intersection) < 4:
        return None, {"status": "REJECTED", "reason": "degenerate_intersection"}
    try:
        mesh = _halfspace_polyhedron(intersection)
    except (QhullError, ValueError) as error:
        return None, {
            "status": "REJECTED",
            "reason": "polyhedron_triangulation_failure",
            "detail": str(error).splitlines()[0],
        }
    if not mesh.is_volume or not mesh.is_convex or mesh.volume <= 0:
        return None, {
            "status": "REJECTED",
            "reason": "invalid_convex_volume",
            "intersection_vertices": len(intersection),
            "is_volume": bool(mesh.is_volume),
            "is_convex": bool(mesh.is_convex),
            "is_watertight": bool(mesh.is_watertight),
            "is_winding_consistent": bool(mesh.is_winding_consistent),
            "euler_number": int(mesh.euler_number),
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "volume_m3": float(mesh.volume),
            "extents_m": np.asarray(mesh.extents).tolist(),
        }
    volume_ratio = float(mesh.volume / patch.mesh.volume)
    if volume_ratio < min_volume_ratio:
        return None, {
            "status": "REJECTED",
            "reason": "minimum_volume_ratio",
            "volume_ratio": volume_ratio,
        }
    source_normals = np.asarray(visual.face_normals[patch.faces], dtype=float)
    if not len(source_normals):
        # An inherited proxy can be unsupported by the reconstructed surface.
        # Keep it subject to a geometric thickness gate rather than deleting it.
        source_normals = np.asarray(original_hull.equations[:, :3], dtype=float)
    original_widths = np.ptp(vertices @ source_normals.T, axis=0)
    widths = np.ptp(np.asarray(mesh.vertices) @ source_normals.T, axis=0)
    original_minimum_thickness = float(original_widths.min())
    minimum_thickness = float(widths.min())
    minimum_allowed_thickness = min_thickness_ratio * original_minimum_thickness
    if minimum_thickness < minimum_allowed_thickness:
        return None, {
            "status": "REJECTED",
            "reason": "minimum_source_normal_thickness",
            "minimum_thickness_m": minimum_thickness,
            "original_minimum_thickness_m": original_minimum_thickness,
            "minimum_allowed_thickness_m": minimum_allowed_thickness,
        }
    displacement, _ = nearest(mesh, vertices)
    max_displacement = float(displacement.max())
    if max_displacement > trust_region_m:
        return None, {
            "status": "REJECTED",
            "reason": "shape_trust_region",
            "max_vertex_displacement_m": max_displacement,
        }
    violation = float(np.max(normals @ np.asarray(mesh.vertices).T + offsets[:, None]))
    if violation > 1e-7:
        return None, {
            "status": "REJECTED",
            "reason": "halfspace_violation",
            "halfspace_violation_m": violation,
        }
    return SurfacePatch(mesh, patch.faces, patch.depth), {
        "status": "PASSED",
        "volume_ratio": volume_ratio,
        "minimum_thickness_m": minimum_thickness,
        "original_minimum_thickness_m": original_minimum_thickness,
        "minimum_allowed_thickness_m": minimum_allowed_thickness,
        "max_vertex_displacement_m": max_displacement,
        "chebyshev_radius_m": float(program.x[3]),
    }
