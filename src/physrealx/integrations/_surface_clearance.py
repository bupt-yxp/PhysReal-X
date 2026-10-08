"""Conservative surface-distance bounds for open articulated triangle meshes.

A triangle is enclosed by its centroid ball. Distance to any surface is
1-Lipschitz, hence d(center, surface) - radius is a lower bound over the
whole triangle. Unresolved cells are subdivided, never silently accepted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh

from physrealx.reconstruction import joint_transform


def conforming_refine(
    mesh: Any,
    axis: Any,
    origin: Any,
    band: float,
    edge: float,
    rounds: int = 3,
    *,
    near: Any = None,
    edge_mask: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Bisect marked edges on *both* incident faces, retaining open boundaries/colors."""
    if edge_mask is not None and rounds != 1:
        raise ValueError("an explicit edge mask applies to exactly one refinement round")
    result = mesh.copy()
    original_faces = len(mesh.faces)
    for _ in range(rounds):
        v, f = np.asarray(result.vertices), np.asarray(result.faces)
        delta = v - np.asarray(origin)
        radial = np.linalg.norm(delta - np.outer(delta @ axis, axis), axis=1)
        if near is not None:
            from scipy.spatial import cKDTree

            radial = cKDTree(near).query(v)[0]
        edges = result.edges_unique
        marked = (radial[edges].min(axis=1) < band) & (
            np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1) > edge
        )
        if edge_mask is not None:
            selected_mask = np.asarray(edge_mask, dtype=bool)
            if selected_mask.shape != (len(edges),):
                raise ValueError("edge mask must match the input mesh edges")
            marked &= selected_mask
        selected = edges[marked]
        if not len(selected):
            break
        lookup = {tuple(e): len(v) + i for i, e in enumerate(selected)}
        vertices = np.concatenate([v, v[selected].mean(axis=1)])
        colors = np.asarray(result.visual.vertex_colors)
        colors = np.concatenate([colors, np.rint(colors[selected].mean(axis=1)).astype(np.uint8)])
        faces = []
        for a, b, c in f:
            ab, bc, ca = (lookup.get(tuple(sorted(e))) for e in ((a, b), (b, c), (c, a)))
            count = sum(x is not None for x in (ab, bc, ca))
            if count == 0:
                faces.append((a, b, c))
            elif count == 3:
                faces.extend([(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)])
            else:
                # Rotate to make edge a-b split; the cases retain face winding.
                while ab is None or (count == 2 and bc is None):
                    a, b, c, ab, bc, ca = b, c, a, bc, ca, ab
                if count == 1:
                    faces.extend([(a, ab, c), (ab, b, c)])
                else:
                    faces.extend([(ab, b, bc), (a, ab, c), (ab, bc, c)])
        result = trimesh.Trimesh(
            vertices=vertices, faces=faces, vertex_colors=colors, process=False
        )
    return result, {
        "method": "conforming_edge_refinement",
        "source_faces": original_faces,
        "faces": len(result.faces),
        "added_vertices": len(result.vertices) - len(mesh.vertices),
        "source_surface_changed": False,
        "remesh_distance_budget_m": 0.0,
        "holes_filled": False,
        "cross_part_triangles_created": False,
    }


def surface_distance_bound(
    source: Any,
    target: Any,
    query: Any,
    required: float,
    *,
    max_depth: int = 10,
    max_cells: int = 200000,
    hinge_motion: tuple[Any, Any, float] | None = None,
) -> dict[str, Any]:
    """Prove separation or return a violating point / unresolved lower bound.

    This tests surfaces. Solid containment is checked separately for closed links.
    Float32 query error is subtracted; it can never improve the lower bound.
    """
    triangles = np.asarray(source.triangles)
    source_faces = np.arange(len(triangles))
    witness_faces: dict[str, int] = {}
    if not len(triangles) or not len(target.faces) or required <= 0:
        raise ValueError("clearance requires nonempty surfaces and a positive distance")
    if not np.isfinite(triangles).all() or not np.isfinite(target.vertices).all():
        raise ValueError("clearance surfaces must have finite coordinates")
    if hinge_motion is not None:
        axis, origin, half_angle = hinge_motion
        axis, origin = np.asarray(axis, dtype=float), np.asarray(origin, dtype=float)
        if (
            axis.shape != (3,)
            or origin.shape != (3,)
            or not np.isfinite(axis).all()
            or not np.isfinite(origin).all()
            or not np.isfinite(half_angle)
            or half_angle < 0
            or not np.isclose(np.linalg.norm(axis), 1, atol=1e-8, rtol=0)
        ):
            raise ValueError("invalid hinge motion bound")
    certified = np.inf
    upper = np.inf
    lowest = np.inf
    queries = 0
    witness = None
    error = max(
        2e-7,
        16
        * float(np.finfo(np.float32).eps)
        * max(
            float(source.scale),
            float(target.scale),
            float(np.abs(source.vertices).max()),
            float(np.abs(target.vertices).max()),
        ),
    )
    for level in range(max_depth + 1):
        centers = triangles.mean(axis=1)
        radius = np.linalg.norm(triangles - centers[:, None], axis=2).max(axis=1)
        dist, ids = query.nearest(target, centers)
        if np.any(ids < 0) or not np.isfinite(dist).all():
            raise ValueError("invalid closest-surface query")
        queries += len(centers)
        motion_bound = 0.0
        if hinge_motion is not None:
            # Work in the fixed target frame. Every point of a source triangle
            # rotates about this hinge; its radial norm is convex, hence bounded
            # by the largest VERTEX radius. r*|dq| bounds its entire arc. Refined
            # cells inherit/recompute the bound; no sampled motion is assumed.
            relative = triangles - origin
            radial = relative - (relative @ axis)[..., None] * axis
            motion_bound = np.linalg.norm(radial, axis=2).max(axis=1) * half_angle
        bounds = dist - radius - error - motion_bound
        lowest = min(float(bounds.min()), float(certified))
        i = int(np.argmin(dist))
        if float(dist[i]) + error < upper:
            upper = float(dist[i]) + error
            witness = centers[i].tolist()
            witness_faces = {"source_face": int(source_faces[i]), "target_face": int(ids[i])}
        if np.any(dist + error < required):
            return {
                "status": "VIOLATION",
                "lower_bound_m": lowest,
                "upper_bound_m": upper,
                "witness": witness,
                **witness_faces,
                "queries": queries,
                "level": level,
            }
        safe = bounds >= required
        if np.any(safe):
            certified = min(certified, float(bounds[safe].min()))
        triangles = triangles[~safe]
        source_faces = source_faces[~safe]
        if not len(triangles):
            return {
                "status": "CERTIFIED",
                "lower_bound_m": float(certified),
                "upper_bound_m": upper,
                "queries": queries,
                "level": level,
            }
        if level == max_depth or len(triangles) * 4 + queries > max_cells:
            return {
                "status": "UNRESOLVED",
                "lower_bound_m": lowest,
                "upper_bound_m": upper,
                "queries": queries,
                "level": level,
                "witness": witness,
                **witness_faces,
            }
        source_faces = np.tile(source_faces, 4)
        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        ab, bc, ca = (a + b) / 2, (b + c) / 2, (c + a) / 2
        triangles = np.concatenate(
            [np.stack(x, axis=1) for x in [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]]
        )
    raise AssertionError("unreachable")


def swept_clearance(
    body: Any,
    child: Any,
    motion: dict[str, Any],
    query: Any,
    *,
    clearance: float,
    samples: int = 61,
    temporal_levels: int = 3,
    max_cells: int = 200000,
) -> dict[str, Any]:
    """Bound surface separation over the entire hinge interval with motion balls.

    For a rigid hinge, displacement over delta-q is <= radius * abs(delta-q).
    The mid-pose bound minus this displacement certifies every pose in the interval.
    No claim about solid volume containment is made for open surfaces.
    """
    axis = np.asarray(motion["axis"])
    origin = np.asarray(motion["origin_m"])
    lo, hi = motion["range"]
    evaluated = []
    unresolved = 0
    violated = 0
    queue = [
        (float(a), float(b), 0)
        for a, b in zip(
            np.linspace(lo, hi, samples)[:-1], np.linspace(lo, hi, samples)[1:], strict=True
        )
    ]
    minimum = np.inf
    while queue:
        a, b, level = queue.pop()
        q = (a + b) / 2
        transform = joint_transform("hinge", axis, origin, q)
        # Keep the BVH mesh constant: transform the body into the child frame.
        moved = body.copy()
        moved.apply_transform(np.linalg.inv(transform))
        result = surface_distance_bound(
            moved,
            child,
            query,
            clearance,
            max_cells=max_cells,
            hinge_motion=(axis, origin, (b - a) / 2),
        )
        lower = result["lower_bound_m"]
        if result["status"] == "CERTIFIED":
            minimum = min(minimum, lower)
        elif result["upper_bound_m"] < clearance:
            # A measured violating point already disproves clearance on this
            # interval. Refining its time cells cannot turn that evidence into
            # a certificate, and needlessly multiplies failed-asset validation.
            minimum = min(minimum, lower)
            violated += 1
        elif level < temporal_levels:
            queue.extend([(a, q, level + 1), (q, b, level + 1)])
            continue
        else:
            minimum = min(minimum, lower)
            violated += int(result["upper_bound_m"] < clearance)
            unresolved += int(result["upper_bound_m"] >= clearance)
        evaluated.append(
            {
                "q": q,
                "interval": [a, b],
                "lower_bound_m": lower,
                "upper_bound_m": result["upper_bound_m"],
                "status": result["status"],
                **{key: result[key] for key in ("source_face", "target_face") if key in result},
            }
        )
    containment = False
    for q in (lo, (lo + hi) / 2, hi):
        M = joint_transform("hinge", axis, origin, q)
        if body.is_watertight and body.is_winding_consistent:
            containment |= bool(
                np.any(query.signed_distance(body, trimesh.transform_points(child.vertices, M)) > 0)
            )
        if child.is_watertight and child.is_winding_consistent:
            containment |= bool(
                np.any(
                    query.signed_distance(
                        child, trimesh.transform_points(body.vertices, np.linalg.inv(M))
                    )
                    > 0
                )
            )
    passed = not (unresolved or violated or containment)
    return {
        "status": "PASSED" if passed else "FAILED",
        "method": "triangle_ball_and_local_hinge_motion_bounds",
        "minimum_clearance_lower_bound_m": float(minimum),
        "required_clearance_m": clearance,
        "continuous_surface_clearance_certified": passed,
        "solid_containment_detected": containment,
        "violating_intervals": violated,
        "unresolved_intervals": unresolved,
        "intervals": sorted(evaluated, key=lambda x: x["q"]),
        "range": list(motion["range"]),
        "open_surface_supported": True,
        "solid_collision_proxy_validation_required": True,
    }
