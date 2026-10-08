"""Independently fit observed articulation states before cross-state supervision.

Only fit observations participate. Whole-object observations are assigned by
closest source part with an ambiguity margin; they never become hand-labelled
part masks. Unobserved vertices stay attached through the original edge field.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

from physrealx.reconstruction import joint_transform


def observed_colors(
    mesh: Any, frames: Any, motion: Any, name: str, q: float, tolerance: float
) -> int:
    """Fit state appearance from visible RGB-D pixels; retain unseen attributes."""
    total = np.zeros((len(mesh.vertices), 3))
    count = np.zeros(len(mesh.vertices))
    for frame in frames:
        if frame.image is None or abs(frame.position - q) > 0.05 * np.ptp(motion["range"]):
            continue
        rgb, depth, mask, camera, intrinsics = frame.image
        values = np.asarray(mesh.vertices)
        if name == motion["child"]:
            values = trimesh.transform_points(
                values,
                joint_transform("hinge", motion["axis"], motion["origin_m"], frame.position - q),
            )
        cv = trimesh.transform_points(values, camera)
        fx, fy, cx, cy = intrinsics
        z = np.maximum(cv[:, 2], 1e-9)
        x = np.rint(fx * cv[:, 0] / z + cx).astype(int)
        y = np.rint(fy * cv[:, 1] / z + cy).astype(int)
        valid = (cv[:, 2] > 0) & (x >= 0) & (x < depth.shape[1]) & (y >= 0) & (y < depth.shape[0])
        ids = np.flatnonzero(valid)
        visible = (
            mask[y[ids], x[ids]]
            & (depth[y[ids], x[ids]] > 0)
            & (abs(depth[y[ids], x[ids]] - z[ids]) < tolerance)
        )
        ids = ids[visible]
        total[ids] += rgb[y[ids], x[ids]]
        count[ids] += 1
    colors = mesh.visual.vertex_colors.copy()
    seen = count > 0
    colors[seen, :3] = np.rint(total[seen] / count[seen, None]).astype(np.uint8)
    mesh.visual.vertex_colors = colors
    return int(seen.sum())


def reconstruct_states(
    meshes: dict[str, Any],
    observations: dict[str, Any],
    motion: dict[str, Any],
    config: Any,
    query: Any,
    *,
    executor: Any = None,
) -> tuple[dict[float, dict[str, Any]], dict[str, Any]]:
    from ._mesh_validity import valid_surface_step

    states, report = {}, {}
    observed_positions = sorted({f.position for p in observations.values() for f in p["fit"]})
    if len(observed_positions) < 2:
        raise ValueError("motion consistency requires two distinct observed fit states")
    for q in (observed_positions[0], observed_positions[-1]):
        posed = {name: m.copy() for name, m in meshes.items()}
        posed[motion["child"]].apply_transform(
            joint_transform("hinge", motion["axis"], motion["origin_m"], q)
        )
        names = list(posed)
        points: dict[str, list[Any]] = {n: [] for n in names}
        # Endpoint RGB-D is independently captured. Nearby video states are
        # transformed by the initial hinge solely for initialization.
        frame_count = {n: 0 for n in names}
        for name, partitions in observations.items():
            for frame in partitions["fit"]:
                if abs(frame.position - q) > 0.05 * np.ptp(motion["range"]):
                    continue
                if name == "__object__":
                    distances = np.stack(
                        [query.nearest(posed[n], frame.points_reference)[0] for n in names]
                    )
                    owner = distances.argmin(0)
                    order = np.sort(distances, axis=0)
                    unique = order[1] - order[0] > config.minimum_clearance_m
                    for i, n in enumerate(names):
                        select = (
                            unique
                            & (owner == i)
                            & (order[0] < config.joint_correspondence_distance_m)
                        )
                        if np.any(select):
                            points[n].append(frame.points_reference[select])
                            frame_count[n] += 1
                else:
                    values = frame.points_reference
                    if name == motion["child"]:
                        values = trimesh.transform_points(
                            values,
                            joint_transform(
                                "hinge", motion["axis"], motion["origin_m"], q - frame.position
                            ),
                        )
                    points[name].append(values)
                    frame_count[name] += 1
        fitted, rows = {}, {}
        for name, mesh in posed.items():
            source = np.asarray(mesh.vertices).copy()
            data = np.concatenate(points[name]) if points[name] else np.empty((0, 3))
            if not len(data):
                raise ValueError(f"no independent fit observations for {name} at state {q}")
            edges = mesh.edges_unique
            count = len(source)
            row = np.repeat(np.arange(len(edges)), 2)
            lap = sparse.csr_matrix(
                (np.tile([1.0, -1.0], len(edges)), (row, edges.ravel())), shape=(len(edges), count)
            )
            previous = source.copy()
            observed_vertices = 0
            updates = []
            for _ in range(config.state_reconstruction_steps):
                distances, ids = query.nearest(mesh, data)
                select = distances < config.joint_correspondence_distance_m
                target, ids, distances = data[select], ids[select], distances[select]
                if not len(ids):
                    raise ValueError("state reconstruction has no valid surface correspondences")
                closest = trimesh.triangles.closest_point(mesh.triangles[ids], target)  # type: ignore[no-untyped-call]
                bary = trimesh.triangles.points_to_barycentric(mesh.triangles[ids], closest)  # type: ignore[no-untyped-call]
                weight = np.sqrt(np.minimum(1, 0.003 / np.maximum(distances, 1e-9)))
                A = sparse.csr_matrix(
                    (
                        (bary * weight[:, None]).ravel(),
                        (np.repeat(np.arange(len(ids)), 3), mesh.faces[ids].ravel()),
                    ),
                    shape=(len(ids), count),
                )
                # Surface-to-observation fitting is restricted to vertices of
                # actually observed faces. It supplies direct local degrees of
                # freedom, without attracting unseen backsides to front pixels.
                seen = np.unique(mesh.faces[ids])
                distances_back, nearest = cKDTree(data).query(mesh.vertices[seen])
                keep = distances_back < min(0.003, config.surface_correspondence_distance_m)
                seen, nearest = seen[keep], nearest[keep]
                observed_vertices = len(seen)
                B = sparse.csr_matrix(
                    (np.ones(len(seen)), (np.arange(len(seen)), seen)), shape=(len(seen), count)
                )
                system = sparse.vstack(
                    [A, 0.5 * B, 0.5 * lap, 0.05 * sparse.eye(count)], format="csr"
                )
                rhs = np.vstack(
                    [
                        target * weight[:, None] - A @ source,
                        0.5 * (data[nearest] - B @ source),
                        np.zeros((len(edges) + count, 3)),
                    ]
                )
                # LSQR repeatedly takes short vector norms/dot products.
                # Nested BLAS pools oversubscribe the geometry workers and
                # spend far more time synchronizing than doing arithmetic.
                from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

                with threadpool_limits(limits=1, user_api="blas"):
                    delta = np.column_stack(
                        [
                            lsqr(system, rhs[:, d], atol=1e-6, btol=1e-6, iter_lim=100)[0]
                            for d in range(3)
                        ]
                    )
                norm = np.linalg.norm(delta, axis=1)
                delta *= np.minimum(1, config.max_surface_displacement_m / np.maximum(norm, 1e-12))[
                    :, None
                ]
                proposed = source + delta
                accepted, update = valid_surface_step(
                    mesh,
                    source,
                    previous,
                    proposed,
                    device=query.describe()["devices"][0]
                    if query.describe()["accelerated"]
                    else "cpu",
                    executor=executor,
                )
                mesh.vertices = accepted
                updates.append(update)
                previous = np.asarray(mesh.vertices).copy()
            fitted[name] = mesh
            seen = observed_colors(
                mesh,
                observations[name]["fit"] + observations.get("__object__", {}).get("fit", []),
                motion,
                name,
                q,
                config.surface_correspondence_distance_m,
            )
            rows[name] = {
                "fit_frames": frame_count[name],
                "fit_points": len(data),
                "max_displacement_m": float(np.linalg.norm(mesh.vertices - source, axis=1).max()),
                "observed_appearance_vertices": seen,
                "observed_reverse_fit_vertices": observed_vertices,
                "local_validity_updates": updates,
            }
        states[float(q)] = fitted
        report[str(q)] = rows
    return states, {
        "method": "independent_state_depth_surface_fit",
        "states": report,
        "heldout_used": False,
        "shared_inverse_motion": True,
    }


class MotionConsistency:
    """Same-part forward/backward closest-triangle geometry and appearance losses."""

    def __init__(self, states: Any, motion: Any, config: Any, query: Any, device: str):
        self.states, self.motion, self.config, self.query = states, motion, config, query
        self.device = device
        self.matches: list[Any] = []

    def tensor(self, value: Any, *, integer: bool = False) -> Any:
        import torch

        return torch.as_tensor(
            np.asarray(value), dtype=torch.long if integer else torch.float64, device=self.device
        )

    def bind(self, current: Any, fitted: Any) -> None:
        self.matches = []
        for position, state_meshes in self.states.items():
            for name, mesh in current.items():
                target = state_meshes[name]
                M = joint_transform("hinge", fitted["axis"], fitted["origin_m"], position)
                if name != self.motion["child"]:
                    M = np.eye(4)
                ids = np.linspace(
                    0,
                    len(mesh.vertices) - 1,
                    min(len(mesh.vertices), self.config.points_per_frame),
                    dtype=int,
                )
                transported = trimesh.transform_points(mesh.vertices[ids], M)
                for reverse in (False, True):
                    if reverse:
                        ids = np.linspace(
                            0,
                            len(target.vertices) - 1,
                            min(len(target.vertices), self.config.points_per_frame),
                            dtype=int,
                        )
                        values = trimesh.transform_points(target.vertices[ids], np.linalg.inv(M))
                        destination = mesh
                    else:
                        values, destination = transported, target
                    dist, face = self.query.nearest(destination, values)
                    keep = dist < self.config.surface_correspondence_distance_m
                    triangles = destination.triangles[face[keep]]
                    closest = trimesh.triangles.closest_point(triangles, values[keep])  # type: ignore[no-untyped-call]
                    bary = trimesh.triangles.points_to_barycentric(triangles, closest)  # type: ignore[no-untyped-call]
                    self.matches.append(
                        (
                            name,
                            position,
                            reverse,
                            self.tensor(ids[keep], integer=True),
                            self.tensor(destination.faces[face[keep]], integer=True),
                            self.tensor(bary),
                            self.tensor(target.vertices),
                            self.tensor(target.visual.vertex_colors[:, :3] / 255),
                            self.tensor(mesh.visual.vertex_colors[:, :3] / 255),
                        )
                    )

    def __call__(self, vertices: Any, transform: Any, axis: Any, origin: Any) -> Any:
        import torch

        terms = []
        for (
            name,
            position,
            reverse,
            ids,
            faces,
            bary,
            target,
            target_rgb,
            source_rgb,
        ) in self.matches:
            if not len(ids):
                continue
            q = position if name == self.motion["child"] else 0.0
            if reverse:
                moved = transform(target[ids], -q, axis, origin)
                matched = (vertices[name][faces] * bary[..., None]).sum(1)
                colors = target_rgb[ids], (source_rgb[faces] * bary[..., None]).sum(1)
            else:
                moved = transform(vertices[name][ids], q, axis, origin)
                matched = (target[faces] * bary[..., None]).sum(1)
                colors = source_rgb[ids], (target_rgb[faces] * bary[..., None]).sum(1)
            # Recompute differentiable barycentric coordinates at the transported
            # point so appearance supervises motion, not a detached constant.
            tri = vertices[name][faces] if reverse else target[faces]
            e0, e1, d = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], moved - tri[:, 0]
            aa, ab, bb = (e0 * e0).sum(1), (e0 * e1).sum(1), (e1 * e1).sum(1)
            ad, bd = (e0 * d).sum(1), (e1 * d).sum(1)
            den = (aa * bb - ab.square()).clamp_min(1e-20)
            u, v = (bb * ad - ab * bd) / den, (aa * bd - ab * ad) / den
            weights = torch.stack([1 - u - v, u, v], 1).clamp(0, 1)
            weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-12)
            interpolated = (
                (source_rgb if reverse else target_rgb)[faces] * weights[..., None]
            ).sum(1)
            terms.append(
                torch.nn.functional.huber_loss(
                    (moved - matched) / 0.001, torch.zeros_like(moved), delta=3.0
                )
                + (colors[0] - interpolated).square().mean()
            )
        return torch.stack(terms).mean() if terms else axis.sum() * 0
