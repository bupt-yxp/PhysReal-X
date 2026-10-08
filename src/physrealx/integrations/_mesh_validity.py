"""Bounded continuous deformation certificates and independent IPC mesh checks."""

from __future__ import annotations

from typing import Any

import numpy as np


def surface_topology(mesh: Any) -> dict[str, Any]:
    """Audit an open two-manifold without repairing or dropping any triangles.

    Edge incidence alone misses bow-tie vertices. Connected components of the
    incident face fans detect these, including two disks touching at one vertex.
    Open boundaries are legitimate; a boundary vertex must have degree two.
    """
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components

    faces = np.asarray(mesh.faces, dtype=np.int64)
    vertices = np.asarray(mesh.vertices)
    finite = bool(np.isfinite(vertices).all())
    indices_valid = bool(len(faces) and faces.min() >= 0 and faces.max() < len(vertices))
    if not finite or not indices_valid:
        return {"valid": False, "finite": finite, "indices_valid": indices_valid}
    duplicate_faces = len(faces) - len(np.unique(np.sort(faces, axis=1), axis=0))
    tri = vertices[faces]
    twice_area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    scale = max(float(np.ptp(vertices, axis=0).max()), 1e-15)
    degenerate = int(np.count_nonzero(twice_area <= np.finfo(float).eps * scale**2))
    # Each corner is one incident face in the link of its vertex.
    corners = np.arange(faces.size).reshape(-1, 3)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edge_corners = np.concatenate([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]])
    order = np.argsort(edges, axis=1)
    edges = np.take_along_axis(edges, order, axis=1)
    edge_corners = np.take_along_axis(edge_corners, order, axis=1)
    unique, inverse, counts = np.unique(edges, axis=0, return_inverse=True, return_counts=True)
    permutation = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]][counts == 2]
    left = edge_corners[permutation[starts]].ravel()
    right = edge_corners[permutation[starts + 1]].ravel()
    graph = sparse.coo_matrix((np.ones(len(left)), (left, right)), shape=(faces.size, faces.size))
    _, fan = connected_components(graph, directed=False)
    vertex_fans = np.unique(np.column_stack([faces.ravel(), fan]), axis=0)
    fan_counts = np.bincount(vertex_fans[:, 0], minlength=len(vertices))
    boundary_edges = unique[counts == 1]
    boundary_degree = np.bincount(boundary_edges.ravel(), minlength=len(vertices))
    nonmanifold_vertices = int(
        np.count_nonzero((fan_counts > 1) | ~np.isin(boundary_degree, [0, 2]))
    )
    boundary_graph = sparse.coo_matrix(
        (np.ones(len(boundary_edges)), (boundary_edges[:, 0], boundary_edges[:, 1])),
        shape=(len(vertices), len(vertices)),
    )
    _, labels = connected_components(boundary_graph, directed=False)
    boundary_loops = len(np.unique(labels[boundary_degree > 0]))
    used = np.unique(faces)
    components_graph = sparse.coo_matrix(
        (np.ones(len(unique)), (unique[:, 0], unique[:, 1])), shape=(len(vertices), len(vertices))
    )
    _, components = connected_components(components_graph, directed=False)
    nonmanifold_edges = int(np.count_nonzero(counts > 2))
    winding = bool(mesh.is_winding_consistent)
    return {
        "valid": not (duplicate_faces or degenerate or nonmanifold_edges or nonmanifold_vertices)
        and winding,
        "nonmanifold_edges": nonmanifold_edges,
        "nonmanifold_vertices": nonmanifold_vertices,
        "nonmanifold_vertex_ids": np.flatnonzero(
            (fan_counts > 1) | ~np.isin(boundary_degree, [0, 2])
        ).tolist(),
        "duplicate_faces": int(duplicate_faces),
        "degenerate_faces": degenerate,
        "boundary_edges": len(boundary_edges),
        "boundary_loops": boundary_loops,
        "components": len(np.unique(components[used])),
        "euler_characteristic": len(used) - len(unique) + len(faces),
        "winding_consistent": winding,
    }


def collision_mesh(mesh: Any) -> Any:
    import ipctk

    return ipctk.CollisionMesh(
        np.asarray(mesh.vertices),
        np.asarray(mesh.edges_unique, dtype=np.int32),
        np.asarray(mesh.faces, dtype=np.int32),
    )


def self_intersections(mesh: Any) -> bool:
    import ipctk

    return bool(ipctk.has_intersections(collision_mesh(mesh), np.asarray(mesh.vertices)))


def deformation_step(mesh: Any, before: Any, after: Any, *, executor: Any = None) -> float:
    """Certify a linear-vertex prefix; deliberately not used for hinge rotation.

    IPC supplies the complete swept primitive candidates. Supporting planes
    certify entire time intervals, not only their sampled endpoints. Exhausted
    or numerically unresolved intervals return a shorter prefix, never success.
    This avoids Tight Inclusion's unbounded small-time retry on grazing sheets.
    """
    witnesses = self_collision_witnesses(mesh, before, after, executor=executor)
    return min((time for _a, _b, time in witnesses), default=1.0)


def _separated_prefix(before: Any, after: Any, split: int) -> Any:
    """Bounded conservative advancement of FV/EE/EV/VV linear trajectories.

    For any fixed plane, min(A projection)-max(B projection) is concave
    in time. The chord of its endpoint bounds therefore certifies the whole
    interval. A fresh plane can extend the certified prefix after each step.
    Plane selection need not be exact: every finite direction is admissible.
    Roundoff allowance applies to projections and interpolated coordinates.
    """
    from ._triangle_contacts import separating_axes

    before, after = np.asarray(before, dtype=float), np.asarray(after, dtype=float)
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("continuous deformation coordinates must be finite")
    velocity = after - before
    times = np.zeros(len(before))
    # An unchanged pair cannot invalidate its already-checked initial geometry.
    stationary = np.all(before == after, axis=(1, 2))
    times[stationary] = 1.0
    active = np.flatnonzero(~stationary)
    a_ids, b_ids = np.resize(np.arange(split), 3), np.resize(np.arange(split, before.shape[1]), 3)
    scale = np.maximum(1.0, np.maximum(abs(before).max((1, 2)), abs(after).max((1, 2))))

    def bound(points: Any, normal: Any, allowance: Any) -> Any:
        projected = (points * normal[:, None]).sum(2)
        return projected[:, :split].min(1) - projected[:, split:].max(1) - allowance

    for _ in range(32):
        if not len(active):
            break
        now = before[active] + times[active, None, None] * velocity[active]
        _gap, normal = separating_axes(now[:, a_ids], now[:, b_ids])
        allowance = 64 * np.finfo(float).eps * scale[active] * abs(normal).sum(1)

        start_gap, end_gap = bound(now, normal, allowance), bound(after[active], normal, allowance)
        valid = (start_gap > 0) & np.isfinite(start_gap) & np.isfinite(end_gap)
        finished = valid & (end_gap > 0)
        times[active[finished]] = 1.0
        advance = valid & ~finished
        rows = active[advance]
        fraction = 0.8 * start_gap[advance] / (start_gap[advance] - end_gap[advance])
        updated = np.minimum(times[rows] + (1 - times[rows]) * fraction, np.nextafter(1.0, 0.0))
        progressing = updated > times[rows]
        times[rows] = updated
        active = rows[progressing]
    return times


def _ccd_stencil_batch(job: Any) -> list[Any]:
    """Certify packed IPC candidates in a CUDA-free worker with bounded work."""
    kind, ids, before, after = job
    split = 2 if kind == "ee" else 1
    times = _separated_prefix(before, after, split)
    return [(ids[i, :split], ids[i, split:], float(times[i])) for i in np.flatnonzero(times < 1)]


def self_collision_witnesses(
    mesh: Any, before: Any, after: Any, *, executor: Any = None
) -> list[Any]:
    """Return blocking features and certified times, including unresolved pairs.

    A blocker is conservative; it is not itself proof of an intersection.
    Every IPC swept candidate is retained, including contacts absent at endpoints.
    """
    import ipctk

    # All impacts on the proposed path matter. Resolving only the earliest
    # prefix exhausts a fixed iteration budget on independent later impacts,
    # then incorrectly scales unrelated geometry with the global fallback.
    candidates = ipctk.Candidates()
    candidates.build(mesh, before, after)
    edges, faces = np.asarray(mesh.edges), np.asarray(mesh.faces)
    families = (
        ("fv", candidates.fv_candidates, 4),
        ("ee", candidates.ee_candidates, 4),
        ("ev", candidates.ev_candidates, 3),
        ("vv", candidates.vv_candidates, 2),
    )
    jobs = []
    for kind, family, width in families:
        for start in range(0, len(family), 2048):
            ids = np.asarray(
                [c.vertex_ids(edges, faces)[:width] for c in family[start : start + 2048]]
            )
            jobs.append((kind, ids, before[ids], after[ids]))
    rows = (
        executor.map(_ccd_stencil_batch, jobs)
        if executor is not None
        else map(_ccd_stencil_batch, jobs)
    )
    return [pair for batch in rows for pair in batch]


def self_contact_planes(before: Any, after: Any, pairs: list[Any]) -> list[Any]:
    """Batch supporting planes of swept IPC witnesses at the valid start."""
    import trimesh

    rows: list[Any] = []
    for shape in {(len(a), len(b)) for a, b in pairs}:
        family = [(a, b) for a, b in pairs if (len(a), len(b)) == shape]
        for begin in range(0, len(family), 8192):
            batch = family[begin : begin + 8192]
            a = np.asarray([x[0] for x in batch], dtype=int)
            b = np.asarray([x[1] for x in batch], dtype=int)
            left, right = before[a], before[b]
            if b.shape[1] == 3:
                closest = trimesh.triangles.closest_point(right, left[:, 0])  # type: ignore[no-untyped-call]
                normal = left[:, 0] - closest
            elif a.shape[1] == b.shape[1] == 2:
                u, v, w = (
                    left[:, 1] - left[:, 0],
                    right[:, 1] - right[:, 0],
                    left[:, 0] - right[:, 0],
                )
                aa, bb, cc = (u * u).sum(1), (u * v).sum(1), (v * v).sum(1)
                dd, ee = (u * w).sum(1), (v * w).sum(1)
                determinant = aa * cc - bb * bb
                t0 = np.zeros(len(a))
                nonparallel = determinant > 1e-12 * aa * cc
                t0[nonparallel] = (bb * ee - cc * dd)[nonparallel] / determinant[nonparallel]
                t0 = np.clip(t0, 0, 1)
                t1 = (bb * t0 + ee) / np.maximum(cc, 1e-30)
                t0 = np.where(t1 < 0, np.clip(-dd / np.maximum(aa, 1e-30), 0, 1), t0)
                t0 = np.where(t1 > 1, np.clip((bb - dd) / np.maximum(aa, 1e-30), 0, 1), t0)
                t1 = np.clip(t1, 0, 1)
                normal = w + t0[:, None] * u - t1[:, None] * v
            elif b.shape[1] == 2:
                edge = right[:, 1] - right[:, 0]
                direction = left[:, 0] - right[:, 0]
                t = np.clip(
                    (direction * edge).sum(1) / np.maximum((edge * edge).sum(1), 1e-30), 0, 1
                )
                normal = direction - t[:, None] * edge
            else:
                normal = left[:, 0] - right[:, 0]
            length = np.linalg.norm(normal, axis=1)
            normal /= np.maximum(length[:, None], 1e-30)
            gap = (left * normal[:, None]).sum(2).min(1) - (right * normal[:, None]).sum(2).max(1)
            required = np.minimum(1e-6, 0.2 * gap)
            trial = (after[a] * normal[:, None]).sum(2).min(1) - (after[b] * normal[:, None]).sum(
                2
            ).max(1)
            valid = (length > 1e-12) & (gap > 1e-12) & (trial < required)
            rows.extend(
                (ai, bi, n, float(d))
                for ai, bi, n, d in zip(
                    a[valid], b[valid], normal[valid], required[valid], strict=True
                )
            )
    return rows


def limit_collision_nodes(
    mesh: Any, graph: Any, previous: Any, on_blockers: Any = None, *, executor: Any = None
) -> dict[str, Any]:
    """Backtrack blocking primitive supports, then certify the whole linear step.

    A local fold should not freeze unrelated graph nodes. Witnesses only propose
    a local correction; a fresh global CCD is mandatory after every correction.
    The final global fallback is conservative and explicitly reported.
    """
    import torch

    with torch.no_grad():

        def vertices(delta: Any) -> Any:
            return (
                (graph["base"] + (delta[graph["idx"]] * graph["weights"][..., None]).sum(1))
                .cpu()
                .numpy()
            )

        before = vertices(previous)
        touched: set[int] = set()
        first_alpha = 1.0
        for iteration in range(9):
            after = vertices(graph["delta"])
            witnesses = self_collision_witnesses(mesh, before, after, executor=executor)
            alpha = min((time for _a, _b, time in witnesses), default=1.0)
            if iteration == 0:
                first_alpha = alpha
            if alpha >= 1:
                return {
                    "initial_global_step": first_alpha,
                    "fallback_scale": 1.0,
                    "local_nodes": len(touched),
                }
            if iteration == 8:
                break
            pairs = [(a, b) for a, b, _toi in witnesses]
            planes = self_contact_planes(before, after, pairs)
            if on_blockers is not None:
                on_blockers(planes)
            # A grazing/coplanar interval may have no valid initial supporting
            # plane. Its primitive stencil still identifies the local nodes;
            # lack of a plane must not freeze an entire unrelated surface.
            if not witnesses:
                break
            node_map = graph["idx"].cpu().numpy()
            local_scale = np.ones(len(graph["delta"]))
            for a, b, toi in witnesses:
                nodes = node_map[np.r_[a, b]].ravel()
                np.minimum.at(local_scale, nodes, 0.5 * toi)
            support = torch.as_tensor(np.flatnonzero(local_scale < 1), device=graph["delta"].device)
            touched.update(support.cpu().tolist())
            scales = torch.as_tensor(
                local_scale, dtype=graph["delta"].dtype, device=graph["delta"].device
            )[support, None]
            graph["delta"][support] = previous[support] + scales * (
                graph["delta"][support] - previous[support]
            )
        # This uniform remainder is only used if local witness resolution did
        # not converge. Never accept a partly checked deformation.
        graph["delta"].copy_(previous + alpha * (graph["delta"] - previous))
        return {
            "initial_global_step": first_alpha,
            "fallback_scale": alpha,
            "local_nodes": len(touched),
        }


def limit_triangle_area_nodes(graph: Any, previous: Any) -> dict[str, int]:
    """Certify noncollapse throughout a linear step, allowing 3D normal rotation.

    The quadratic normal curve is bounded by the segment joining its endpoint
    normals plus its exact quadratic remainder. No fixed normal hemisphere is
    imposed: a rotated triangle is not an inverted triangle in three dimensions.
    """
    import torch

    with torch.no_grad():
        faces = graph["faces"]
        base = graph["base"][faces]
        normal = torch.linalg.cross(base[:, 1] - base[:, 0], base[:, 2] - base[:, 0])
        old_vertices = graph["base"] + (previous[graph["idx"]] * graph["weights"][..., None]).sum(1)
        old_tri = old_vertices[faces]
        e0, e1 = old_tri[:, 1] - old_tri[:, 0], old_tri[:, 2] - old_tri[:, 0]
        initial_normal = torch.linalg.cross(e0, e1)

        def invalid() -> Any:
            vertices = graph["base"] + (
                graph["delta"][graph["idx"]] * graph["weights"][..., None]
            ).sum(1)
            tri = vertices[faces]
            d0, d1 = tri[:, 1] - tri[:, 0] - e0, tri[:, 2] - tri[:, 0] - e1
            final_normal = torch.linalg.cross(e0 + d0, e1 + d1)
            difference = final_normal - initial_normal
            denominator = difference.square().sum(1)
            t = (-(initial_normal * difference).sum(1) / denominator.clamp_min(1e-30)).clamp(0, 1)
            lower = torch.linalg.vector_norm(initial_normal + t[:, None] * difference, dim=1)
            lower -= 0.25 * torch.linalg.vector_norm(torch.linalg.cross(d0, d1), dim=1)
            return lower <= 0.1 * torch.linalg.vector_norm(normal, dim=1)

        touched: set[int] = set()
        for _ in range(24):
            bad = invalid()
            if not bool(bad.any()):
                return {"backtracked": len(touched), "restored": 0}
            ids = torch.unique(graph["idx"][faces[bad]])
            touched.update(ids.cpu().tolist())
            graph["delta"][ids] = previous[ids] + 0.5 * (graph["delta"][ids] - previous[ids])
        # A tiny face may need more than 24 halvings. Restore its complete node
        # support exactly, propagating only if neighbouring faces become invalid.
        # Each pass restores at least one new node, so this process is bounded.
        restored: set[int] = set()
        while bool((bad := invalid()).any()):
            ids = torch.unique(graph["idx"][faces[bad]])
            affected = set(ids.cpu().tolist())
            if not affected.difference(restored):
                raise ValueError("previous surface already violates triangle-area validity")
            restored.update(affected)
            graph["delta"][ids] = previous[ids]
        return {"backtracked": len(touched), "restored": len(restored)}


def valid_surface_step(
    mesh: Any, base: Any, before: Any, proposed: Any, *, device: str = "cpu", executor: Any = None
) -> tuple[Any, dict[str, Any]]:
    """Project only unsafe vertex supports, then certify the complete update."""
    import torch

    def tensor(x: Any) -> Any:
        return torch.as_tensor(np.array(x), dtype=torch.float64, device=device)

    old = tensor(before - base)
    graph = dict(
        base=tensor(base),
        delta=tensor(proposed - base),
        faces=torch.as_tensor(np.asarray(mesh.faces), device=device),
        idx=torch.arange(len(base), device=device)[:, None],
        weights=torch.ones((len(base), 1), device=device),
    )
    area = limit_triangle_area_nodes(graph, old)
    ipc = collision_mesh(mesh)
    contact = limit_collision_nodes(ipc, graph, old, executor=executor)
    limit_triangle_area_nodes(graph, old)
    trial = base + graph["delta"].cpu().numpy()
    alpha = deformation_step(ipc, before, trial, executor=executor)
    result = before + alpha * (trial - before)
    return result, {"area": area, "contact": contact, "final_ccd_scale": alpha}
