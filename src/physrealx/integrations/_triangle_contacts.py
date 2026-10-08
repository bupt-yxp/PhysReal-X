"""Triangle-pair separation constraints, including initially intersecting surfaces.

Unlike a centroid-to-plane sign, the separating-axis interval is defined by all
six vertices. It therefore detects edge/face crossings and works on open meshes.
These are optimization constraints; continuous certification is independent.
"""

from __future__ import annotations

from concurrent.futures import Executor, ThreadPoolExecutor
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from physrealx.reconstruction import joint_transform


def separating_axes(
    left: Any, right: Any, preferred: Any = None, *, minimum_alignment: Any = 0.3
) -> tuple[Any, Any]:
    """Return the largest signed interval gap and its direction (B towards A).

    Includes in-plane edge normals for coplanar triangles. A positive result is
    a sufficient separation bound, not an estimate of penetration depth.
    """
    a, b = np.asarray(left), np.asarray(right)
    ea, eb = np.roll(a, -1, axis=1) - a, np.roll(b, -1, axis=1) - b
    na, nb = np.cross(ea[:, 0], ea[:, 1]), np.cross(eb[:, 0], eb[:, 1])
    features = [(a[:, :, None] - b[:, None, :]).reshape(-1, 9, 3)]
    for vertices, starts, edges in ((a, b, eb), (b, a, ea)):
        delta = vertices[:, :, None] - starts[:, None, :]
        fraction = np.clip(
            (delta * edges[:, None]).sum(3) / np.maximum((edges * edges).sum(2)[:, None], 1e-30),
            0,
            1,
        )
        features.append((delta - fraction[..., None] * edges[:, None]).reshape(-1, 9, 3))
    axes = np.concatenate(
        [
            na[:, None],
            nb[:, None],
            np.cross(ea[:, :, None], eb[:, None, :]).reshape(-1, 9, 3),
            np.cross(na[:, None], ea),
            np.cross(nb[:, None], eb),
            *features,
        ],
        axis=1,
    )
    length = np.linalg.norm(axes, axis=2)
    axes /= np.maximum(length[..., None], 1e-30)
    pa, pb = np.einsum("nvc,nac->nav", a, axes), np.einsum("nvc,nac->nav", b, axes)
    forward = pa.min(2) - pb.max(2)
    backward = pb.min(2) - pa.max(2)
    if preferred is not None:
        preferred = np.asarray(preferred, dtype=float)
        if preferred.shape == (3,):
            preferred = np.broadcast_to(preferred, (len(a), 3))
        if preferred.shape != (len(a), 3):
            raise ValueError("separation direction must be a vector or one vector per pair")
        norm = np.linalg.norm(preferred, axis=1)
        if not np.isfinite(preferred).all() or np.any(norm < 1e-12):
            raise ValueError("separation direction must be finite and nonzero")
        preferred = preferred / norm[:, None]
        # The preferred direction itself is a valid supporting-plane axis.
        # Including it avoids an empty admissible set on nearly degenerate faces.
        axes = np.concatenate([axes, preferred[:, None]], axis=1)
        length = np.column_stack([length, np.ones(len(a))])
        pa, pb = np.einsum("nvc,nac->nav", a, axes), np.einsum("nvc,nac->nav", b, axes)
        forward, backward = pa.min(2) - pb.max(2), pb.min(2) - pa.max(2)
        alignment = np.einsum("nac,nc->na", axes, preferred)
        direction_sign = np.where(alignment >= 0, 1.0, -1.0)
        gaps = np.where(alignment >= 0, forward, backward)
        gaps[(length < 1e-18) | (abs(alignment) < minimum_alignment)] = -np.inf
        index = np.argmax(gaps, axis=1)
        rows = np.arange(len(a))
        return gaps[rows, index], axes[rows, index] * direction_sign[rows, index, None]
    gaps = np.maximum(forward, backward)
    gaps[length < 1e-18] = -np.inf
    index = np.argmax(gaps, axis=1)
    rows = np.arange(len(a))
    direction = (
        axes[rows, index]
        * np.where(forward[rows, index] >= backward[rows, index], 1.0, -1.0)[:, None]
    )
    return gaps[rows, index], direction


def triangle_pairs(left: Any, right: Any, margin: float) -> tuple[Any, Any, Any, Any]:
    """All AABB candidate pairs whose separating-axis bound is below margin."""
    a, b = np.asarray(left.triangles), np.asarray(right.triangles)
    if not len(a) or not len(b):
        return np.empty(0, int), np.empty(0, int), np.empty((0, 3)), np.empty(0)
    lower, upper = a.min(1) - margin, a.max(1) + margin
    selected = np.flatnonzero(
        np.all(upper >= b.min((0, 1)), axis=1) & np.all(lower <= b.max((0, 1)), axis=1)
    )
    # A bucketed L-infinity query is a conservative AABB broad phase. Unlike
    # libspatialindex, cKDTree queries support concurrent pose workers. Size
    # buckets stop a single long triangle inflating every search radius.
    blo, bhi = b.min(1), b.max(1)
    ac, bc = (lower + upper) / 2, (blo + bhi) / 2
    ar, br = ((upper - lower) / 2).max(1), ((bhi - blo) / 2).max(1)
    levels = np.floor(np.log2(np.maximum(br, 1e-15))).astype(int)
    candidate_a, candidate_b = [], []
    for level in np.unique(levels):
        ids = np.flatnonzero(levels == level)
        tree = cKDTree(bc[ids])
        for start in range(0, len(selected), 512):
            rows = selected[start : start + 512]
            radius = np.nextafter(ar[rows] + br[ids].max(), np.inf)
            hits = tree.query_ball_point(ac[rows], radius, p=np.inf, workers=1)
            counts = [len(h) for h in hits]
            if not sum(counts):
                continue
            ai = np.repeat(rows, counts)
            bj = ids[np.concatenate(hits).astype(np.int64)]
            keep = np.all(upper[ai] >= blo[bj], axis=1) & np.all(lower[ai] <= bhi[bj], axis=1)
            candidate_a.append(ai[keep])
            candidate_b.append(bj[keep])
    ai = np.concatenate(candidate_a) if candidate_a else np.empty(0, dtype=int)
    bj = np.concatenate(candidate_b) if candidate_b else np.empty(0, dtype=int)
    indexes, normals, distances = [], [], []
    for start in range(0, len(ai), 8192):
        i, j = ai[start : start + 8192], bj[start : start + 8192]
        gap, direction = separating_axes(a[i], b[j])
        keep = gap < margin
        indexes.append(np.column_stack([i[keep], j[keep]]))
        normals.append(direction[keep])
        distances.append(gap[keep])
    if not indexes:
        return np.empty(0, int), np.empty(0, int), np.empty((0, 3)), np.empty(0)
    pairs = np.concatenate(indexes)
    return pairs[:, 0], pairs[:, 1], np.concatenate(normals), np.concatenate(distances)


def swept_constraints(
    parent: Any,
    child: Any,
    motion: dict[str, Any],
    positions: Any,
    *,
    margin: float,
    budget: int,
    workers: int = 4,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """Parallel pose batches retaining every actual crossing.

    The budget limits additional near contacts, never existing intersections or
    validation. Omitting known crossings makes local repairs chase each other.
    """

    jobs = [(parent, child, motion, float(q), margin, budget) for q in positions]
    if not jobs:
        raise ValueError("at least one contact pose is required")
    if executor is not None:
        rows = list(executor.map(_pose_constraints, jobs))
    else:
        with ThreadPoolExecutor(
            max_workers=max(1, workers), thread_name_prefix="triangle-pose"
        ) as pool:
            rows = list(pool.map(_pose_constraints, jobs))
    return {
        "parent_faces": np.concatenate([r[0] for r in rows]),
        "child_faces": np.concatenate([r[1] for r in rows]),
        "normals": np.concatenate([r[2] for r in rows]),
        "positions": np.concatenate([r[3] for r in rows]),
        "gaps": np.concatenate([r[4] for r in rows]),
        "candidate_pairs": sum(r[5] for r in rows),
        "intersecting_pairs": sum(r[6] for r in rows),
        "affected_parent_vertices": np.unique(np.concatenate([r[7] for r in rows])),
        "affected_child_vertices": np.unique(np.concatenate([r[8] for r in rows])),
    }


def _pose_constraints(job: tuple[Any, ...]) -> tuple[Any, ...]:
    """Picklable, CUDA-free work item for the solve's persistent spawn pool."""
    parent, child, motion, q, margin, budget = job
    moved = child.copy()
    moved.apply_transform(joint_transform("hinge", motion["axis"], motion["origin_m"], q))
    ai, bj, normal, gap = triangle_pairs(parent, moved, margin)
    total = len(ai)
    crossing = int(np.count_nonzero(gap <= 0))
    mandatory = np.flatnonzero(gap <= 0)
    near = np.flatnonzero(gap > 0)
    order = near[np.argsort(gap[near], kind="stable")]
    if len(order) > budget:
        _, first_a = np.unique(ai[order], return_index=True)
        _, first_b = np.unique(bj[order], return_index=True)
        first = np.unique(np.concatenate([first_a, first_b]))
        priority = np.concatenate([first, np.setdiff1d(np.arange(len(order)), first)])
        order = order[priority[:budget]]
    order = np.r_[mandatory, order]
    return (
        parent.faces[ai[order]],
        child.faces[bj[order]],
        normal[order],
        np.full(len(order), q),
        gap[order],
        total,
        crossing,
        np.unique(parent.faces[ai].reshape(-1)),
        np.unique(child.faces[bj].reshape(-1)),
    )
