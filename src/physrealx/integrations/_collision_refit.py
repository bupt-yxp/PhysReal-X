"""Surface-supported convex patch fitting and local coverage diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import trimesh
from numpy.typing import NDArray

from physrealx.integrations.collision import _exterior_samples


def nearest(mesh: Any, points: Any) -> tuple[Any, Any]:
    """Bound temporary proximity-query memory for dense reconstructed surfaces."""
    distances, faces = [], []
    for chunk in np.array_split(points, max(1, int(np.ceil(len(points) / 512)))):
        _, distance, face = trimesh.proximity.closest_point(  # type: ignore[no-untyped-call]
            mesh, chunk
        )
        distances.append(distance)
        faces.append(face)
    return np.concatenate(distances), np.concatenate(faces)


@dataclass
class SurfacePatch:
    mesh: Any
    faces: Any
    depth: int = 0
    baseline_volume_m3: float | None = None
    baseline_minimum_thickness_m: float | None = None
    _clip_hull_equations: Any | None = field(default=None, repr=False, compare=False)


NearestQuery = Callable[[Any, Any], tuple[Any, Any]]


def bind_surface(
    visual: Any,
    parts: list[Any],
    *,
    query: NearestQuery = nearest,
) -> list[SurfacePatch]:
    """Every source triangle has one owner; refinement never drops supported faces."""
    proxy = trimesh.util.concatenate(parts)
    owner = np.repeat(np.arange(len(parts)), [len(part.faces) for part in parts])
    _, closest = query(proxy, visual.triangles_center)
    assigned = owner[closest]
    return [SurfacePatch(part, np.flatnonzero(assigned == i)) for i, part in enumerate(parts)]


def fit_patches(visual: Any, patch: SurfacePatch, skin_m: float) -> list[SurfacePatch] | None:
    """Split a source-supported region and fit convex shells, or reject the split.

    The fixed skin gives observed open/planar surfaces a numerically valid collision
    volume. It is never reduced in response to penetration. Unsupported old proxy
    fragments may disappear only when they own no module-one source face; every
    supported face remains independently audited. Disconnected source regions are
    separated before a deterministic PCA split so a convex hull cannot bridge
    unrelated module-one surfaces.
    A numerically degenerate child rejects the subdivision so no source faces vanish.
    """
    ids = patch.faces
    if not len(ids):
        return []
    if len(ids) == 1:
        groups = [ids]
    else:
        selected = {int(face) for face in ids}
        neighbors: dict[int, set[int]] = {face: set() for face in selected}
        for left, right in visual.face_adjacency:
            a, b = int(left), int(right)
            if a in selected and b in selected:
                neighbors[a].add(b)
                neighbors[b].add(a)
        pending = set(selected)
        components: list[NDArray[Any]] = []
        while pending:
            stack = [min(pending)]
            pending.remove(stack[0])
            component = []
            while stack:
                face = stack.pop()
                component.append(face)
                adjacent = neighbors[face].intersection(pending)
                pending.difference_update(adjacent)
                stack.extend(sorted(adjacent, reverse=True))
            components.append(np.asarray(sorted(component), dtype=int))
        components.sort(key=lambda value: (-len(value), int(value[0])))
        if len(components) > 1:
            # Reconstructed triangle soups can contain hundreds of tiny disconnected
            # islands. Peeling one component per round leaves a large hull bridging
            # distant surfaces. Spatially order whole components and make a balanced
            # binary split; no component is cut and convergence becomes logarithmic.
            centers = np.asarray(
                [visual.triangles_center[value].mean(axis=0) for value in components]
            )
            centered = centers - centers.mean(axis=0)
            _, _, basis = np.linalg.svd(centered, full_matrices=False)
            order = np.argsort(centered @ basis[0], kind="stable")
            ordered_components = [components[int(index)] for index in order]
            cumulative = np.cumsum([len(value) for value in ordered_components])
            target = cumulative[-1] / 2
            boundary = 1 + int(np.argmin(np.abs(cumulative[:-1] - target)))
            groups = [
                np.concatenate(ordered_components[:boundary]),
                np.concatenate(ordered_components[boundary:]),
            ]
        else:
            centers = visual.triangles_center[ids]
            centered = centers - centers.mean(axis=0)
            _, _, basis = np.linalg.svd(centered, full_matrices=False)
            direction = basis[0]
            order = np.argsort(centered @ direction, kind="stable")
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


def fit_one_sided_patch(
    visual: Any,
    patch: SurfacePatch,
    *,
    skin_m: float,
    polarity: int,
) -> SurfacePatch | None:
    """Refit an open source patch as a one-sided convex shell.

    Symmetric thickening is a poor model at an articulated seam: two source
    surfaces that legitimately meet at the hinge are expanded into each other.
    This construction keeps every module-one source triangle on the proxy and
    places the numerical thickness on only one side.  The caller must choose the
    polarity from measured contact directions and independently re-run surface
    fidelity and physics gates; source normals alone do not define an inside for
    an open reconstruction.
    """
    if polarity not in {-1, 1}:
        raise ValueError("one-sided patch polarity must be -1 or 1")
    if not len(patch.faces):
        return patch
    triangles = np.asarray(visual.triangles[patch.faces], dtype=float)
    normals = np.asarray(visual.face_normals[patch.faces], dtype=float)
    points = np.concatenate(
        [
            triangles,
            triangles + float(polarity) * normals[:, None, :] * (2.0 * skin_m),
        ]
    ).reshape(-1, 3)
    points = np.unique(points.astype(np.float32).astype(np.float64), axis=0)
    if len(points) < 4 or np.linalg.matrix_rank(points[1:] - points[0], tol=1e-10) < 3:
        return None
    try:
        mesh = trimesh.convex.convex_hull(points)
    except (ValueError, RuntimeError):
        return None
    if not mesh.is_volume or not mesh.is_convex or mesh.volume <= 0:
        return None
    return SurfacePatch(mesh, patch.faces, patch.depth)


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
        query: NearestQuery = nearest,
        contact_faces: Any | None = None,
    ) -> None:
        self.visual = visual
        self.contact_faces = (
            np.asarray(contact_faces, dtype=int)
            if contact_faces is not None
            else (
                np.concatenate([p.faces for i, p in enumerate(patches) if i in contact_parts])
                if contact_parts
                else np.empty(0, dtype=int)
            )
        )
        random, faces = trimesh.sample.sample_surface(visual, 4096, seed=seed)
        # Resolve the 99% contact-coverage gate with deterministic triangle
        # quadrature. Sparse cells otherwise jump from 100% to 95.24% after one
        # missed probe and cannot represent a small boundary loss faithfully.
        weights = np.asarray(
            [
                [2 / 3, 1 / 6, 1 / 6],
                [1 / 6, 2 / 3, 1 / 6],
                [1 / 6, 1 / 6, 2 / 3],
            ]
        )
        contact_triangles = visual.triangles[self.contact_faces]
        quadrature = np.einsum("qv,nvc->nqc", weights, contact_triangles).reshape(-1, 3)
        quadrature_faces = np.repeat(self.contact_faces, len(weights))
        self.points = np.concatenate([visual.vertices, visual.triangles_center, random, quadrature])
        vertex_faces = np.maximum(visual.vertex_faces[:, 0], 0)
        self.face_ids = np.concatenate(
            [vertex_faces, np.arange(len(visual.faces)), faces, quadrature_faces]
        )
        # Coverage is a surface-area statement, not a vertex-density statement.
        # Reconstructed meshes are highly non-uniform near the hinge; counting
        # probes equally lets one densely tessellated sliver dominate an entire
        # spatial cell.  Distribute each triangle's area across all probes bound
        # to that triangle so remeshing cannot silently tighten or loosen gates.
        probes_per_face = np.bincount(self.face_ids, minlength=len(visual.faces))
        self.weights = np.asarray(visual.area_faces, dtype=float)[self.face_ids] / np.maximum(
            probes_per_face[self.face_ids], 1
        )
        self.contact = np.isin(self.face_ids, self.contact_faces)
        self.contact_quadrature_probe_count = len(quadrature)
        _, self.cells = np.unique(
            np.floor(self.points / cell_size_m).astype(np.int64), axis=0, return_inverse=True
        )
        self.seed = seed
        self._nearest = query

    def evaluate(
        self,
        patches: list[SurfacePatch],
        *,
        max_error_m: float,
        coverage_distance_m: float,
        min_coverage: float,
    ) -> tuple[dict[str, Any], set[int]]:
        metrics, selected, _ = self.evaluate_with_priorities(
            patches,
            max_error_m=max_error_m,
            coverage_distance_m=coverage_distance_m,
            min_coverage=min_coverage,
        )
        return metrics, selected

    def evaluate_with_priorities(
        self,
        patches: list[SurfacePatch],
        *,
        max_error_m: float,
        coverage_distance_m: float,
        min_coverage: float,
    ) -> tuple[dict[str, Any], set[int], dict[int, float]]:
        """Measure fidelity and rank the exact proxy patches causing violations."""
        parts = [p.mesh for p in patches]
        proxy = trimesh.util.concatenate(parts)
        distance, _ = self._nearest(proxy, self.points)
        # Exclude buried convex partition faces when checking reverse fidelity.
        outside = _exterior_samples(parts, proxy, self.seed, include_vertices=True)
        reverse, _ = self._nearest(self.visual, outside)
        covered = distance <= coverage_distance_m
        cell_weight = np.bincount(self.cells, weights=self.weights)
        cell_coverage = np.bincount(self.cells, weights=self.weights * covered) / np.maximum(
            cell_weight, np.finfo(float).tiny
        )
        contact_distance = distance[self.contact]
        bad = (distance > max_error_m) | (cell_coverage[self.cells] < min_coverage)
        bad_faces = set(self.face_ids[bad].tolist())
        selected = {i for i, p in enumerate(patches) if bad_faces.intersection(p.faces)}
        priorities: dict[int, float] = {}
        source_owner = np.empty(len(self.visual.faces), dtype=int)
        for index, patch in enumerate(patches):
            source_owner[patch.faces] = index
        bad_indices = np.flatnonzero(bad)
        for probe_index in bad_indices:
            owner = int(source_owner[int(self.face_ids[probe_index])])
            distance_score = float(distance[probe_index]) / max_error_m
            local_coverage = float(cell_coverage[int(self.cells[probe_index])])
            coverage_score = 1.0 + max(0.0, min_coverage - local_coverage) / min_coverage
            priorities[owner] = max(priorities.get(owner, 0.0), distance_score, coverage_score)
        reverse_bad = reverse > max_error_m
        if np.any(reverse_bad):
            _, closest = self._nearest(proxy, outside[reverse_bad])
            proxy_owner = np.repeat(np.arange(len(parts)), [len(p.faces) for p in parts])
            bad_owners = proxy_owner[closest]
            for owner, error in zip(bad_owners, reverse[reverse_bad], strict=True):
                index = int(owner)
                selected.add(index)
                priorities[index] = max(priorities.get(index, 0.0), float(error) / max_error_m)
        contact_weight = self.weights[self.contact]
        contact_coverage = (
            float(np.sum(contact_weight * covered[self.contact]) / np.sum(contact_weight))
            if len(contact_weight) and float(np.sum(contact_weight)) > 0
            else 1.0
        )
        maximum = max(float(distance.max()), float(reverse.max()))
        passed = (
            maximum <= max_error_m
            and float(cell_coverage.min()) >= min_coverage
            and contact_coverage >= min_coverage
        )
        return (
            {
                "status": "PASSED" if passed else "FAILED",
                "probe_count": len(distance),
                "contact_probe_count": len(contact_distance),
                "contact_quadrature_probe_count": self.contact_quadrature_probe_count,
                "cell_count": len(cell_coverage),
                "bidirectional_sample_max_m": maximum,
                "surface_p95_m": max(
                    float(np.quantile(distance, 0.95)), float(np.quantile(reverse, 0.95))
                ),
                "surface_p99_m": max(
                    float(np.quantile(distance, 0.99)), float(np.quantile(reverse, 0.99))
                ),
                "surface_coverage": float(np.sum(self.weights * covered) / np.sum(self.weights)),
                "worst_cell_coverage": float(cell_coverage.min()),
                "contact_coverage": contact_coverage,
                "contact_sample_max_m": float(contact_distance.max())
                if len(contact_distance)
                else 0.0,
                "uncovered_source_faces": len(bad_faces),
            },
            selected,
            priorities,
        )


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
    # Qhull already emits a closed triangular boundary. Rebuilding every coplanar
    # polygon with a second 2-D hull made late refinement rounds quadratic in the
    # number of support planes. Orient all Qhull facets against their matching
    # outward equations in one vectorized pass; this keeps the exact float32
    # vertices and remains watertight after GLB round-trip.
    faces = np.asarray(hull.simplices, dtype=int).copy()
    triangles = normalized[faces]
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    flip = np.einsum("ij,ij->i", cross, hull.equations[:, :3]) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    used = np.unique(faces)
    remap = np.full(len(points), -1, dtype=int)
    remap[used] = np.arange(len(used))
    return trimesh.Trimesh(
        vertices=points[used],
        faces=remap[faces],
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
    query: NearestQuery = nearest,
) -> tuple[SurfacePatch | None, dict[str, Any]]:
    """Intersect one convex proxy with contact planes under hard shape gates.

    The original convex hull contributes immutable half-spaces; contact planes may
    only remove unsupported proxy volume. A Chebyshev-center linear program finds a
    strict interior point before half-space intersection. Source-supported patches
    are governed by the caller's fixed source probes and local coverage gate.  A
    nearest-vertex trust region is meaningful only for inherited proxy fragments
    with no source-face owner; applying it to a long, thin source triangle confuses
    tangential corner removal with normal surface error and can reject a bounded
    support cut even though the immutable source remains fully covered.
    """
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError

    vertices = np.asarray(patch.mesh.vertices, dtype=float)
    if patch._clip_hull_equations is None:
        try:
            patch._clip_hull_equations = np.asarray(ConvexHull(vertices).equations, dtype=float)
        except QhullError:
            return None, {"status": "REJECTED", "reason": "invalid_original_hull"}
    original_equations = patch._clip_hull_equations
    directions = np.asarray([value[0] for value in halfspaces], dtype=float).reshape(-1, 3)
    direction_norms = np.linalg.norm(directions, axis=1)
    valid = direction_norms > 1e-12
    extra_equations = np.column_stack(
        [
            directions[valid] / direction_norms[valid, None],
            -np.asarray([value[1] for value in halfspaces], dtype=float)[valid]
            / direction_norms[valid],
        ]
    )
    equations = np.vstack([original_equations, extra_equations])
    equation_norms = np.linalg.norm(equations[:, :3], axis=1)
    normalized = equations / equation_norms[:, None]
    rounded_normals, groups = np.unique(np.round(normalized[:, :3], 7), axis=0, return_inverse=True)
    # Sort by normal group and offset; the last row in every group is its tightest
    # inequality. This replaces a Python loop over every inherited support plane.
    order = np.lexsort((normalized[:, 3], groups))
    ordered_groups = groups[order]
    selected = order[np.r_[np.flatnonzero(np.diff(ordered_groups)), len(order) - 1]]
    matrix = normalized[selected]
    if len(matrix) != len(rounded_normals):
        raise RuntimeError("half-space normal grouping lost a constraint")
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
    baseline_volume = (
        float(patch.baseline_volume_m3)
        if patch.baseline_volume_m3 is not None
        else float(patch.mesh.volume)
    )
    volume_ratio = float(mesh.volume / baseline_volume)
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
        source_normals = np.asarray(original_equations[:, :3], dtype=float)
    original_widths = np.ptp(vertices @ source_normals.T, axis=0)
    widths = np.ptp(np.asarray(mesh.vertices) @ source_normals.T, axis=0)
    current_minimum_thickness = float(original_widths.min())
    baseline_minimum_thickness = (
        float(patch.baseline_minimum_thickness_m)
        if patch.baseline_minimum_thickness_m is not None
        else current_minimum_thickness
    )
    minimum_thickness = float(widths.min())
    minimum_allowed_thickness = min_thickness_ratio * baseline_minimum_thickness
    if minimum_thickness < minimum_allowed_thickness:
        return None, {
            "status": "REJECTED",
            "reason": "minimum_source_normal_thickness",
            "minimum_thickness_m": minimum_thickness,
            "current_minimum_thickness_m": current_minimum_thickness,
            "baseline_minimum_thickness_m": baseline_minimum_thickness,
            "minimum_allowed_thickness_m": minimum_allowed_thickness,
        }
    displacement, _ = query(mesh, vertices)
    max_displacement = float(displacement.max())
    source_supported = bool(len(patch.faces))
    if not source_supported and max_displacement > trust_region_m:
        return None, {
            "status": "REJECTED",
            "reason": "shape_trust_region",
            "max_vertex_displacement_m": max_displacement,
            "trust_region_basis": "unsupported_proxy_vertices",
        }
    violation = float(np.max(normals @ np.asarray(mesh.vertices).T + offsets[:, None]))
    if violation > 1e-7:
        return None, {
            "status": "REJECTED",
            "reason": "halfspace_violation",
            "halfspace_violation_m": violation,
        }
    return SurfacePatch(
        mesh,
        patch.faces,
        patch.depth,
        baseline_volume_m3=baseline_volume,
        baseline_minimum_thickness_m=baseline_minimum_thickness,
    ), {
        "status": "PASSED",
        "volume_ratio": volume_ratio,
        "minimum_thickness_m": minimum_thickness,
        "current_minimum_thickness_m": current_minimum_thickness,
        "baseline_minimum_thickness_m": baseline_minimum_thickness,
        "minimum_allowed_thickness_m": minimum_allowed_thickness,
        "max_vertex_displacement_m": max_displacement,
        "trust_region_basis": (
            "source_surface_coverage" if source_supported else "unsupported_proxy_vertices"
        ),
        "chebyshev_radius_m": float(program.x[3]),
    }
