"""Observation-compatible, cross-part Mesh-Untangling energies.

Equations follow the authors' Gaussian surface energy and triangle Minkowski
penalty (https://github.com/Simulation-Intelligence/Mesh-Untangling). This is a
PyTorch articulation extension, not a wrapper around their CUDA flow solver.
Areas and bandwidths are fixed during a step: shrinking triangles must not
reduce repulsion. An energy is a search objective, never a clearance certificate.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from physrealx.reconstruction import joint_transform


def minkowski_gap(left: Any, right: Any) -> Any:
    """Signed supporting-facet offset of A-B, including intersecting triangles.

    Eleven distinct facet directions replace the reference's 66 directed-edge
    cross products; opposite directions are both tested. Normals participate in
    autograd. At a coplanar crossing the offset is zero, so a positive-clearance
    hinge loss still acts. This offset is NOT Euclidean penetration depth.
    """
    import torch

    ea, eb = left.roll(-1, 1) - left, right.roll(-1, 1) - right
    axes = torch.cat(
        [
            torch.linalg.cross(ea[:, :1], ea[:, 1:2]),
            torch.linalg.cross(eb[:, :1], eb[:, 1:2]),
            torch.linalg.cross(ea[:, :, None], eb[:, None, :]).reshape(-1, 9, 3),
        ],
        dim=1,
    )
    length = torch.linalg.vector_norm(axes, dim=-1)
    scale = torch.maximum(ea.square().sum(-1).amax(-1), eb.square().sum(-1).amax(-1))
    valid = length > torch.finfo(left.dtype).eps * scale[:, None]
    normals = axes / length.clamp_min(torch.finfo(left.dtype).tiny ** 0.25)[..., None]
    a, b = torch.einsum("nvc,nkc->nkv", left, normals), torch.einsum("nvc,nkc->nkv", right, normals)
    forward, backward = a.amin(-1) - b.amax(-1), b.amin(-1) - a.amax(-1)
    # Select one support branch as the reference implementation does. Averaging
    # equal, opposite branches cancels the entire force at coplanar overlap.
    gap = torch.where(forward >= backward, forward, backward)
    return gap.masked_fill(~valid, -torch.inf).max(-1).values


def gaussian_energy(
    left: Any,
    right: Any,
    pairs: Any,
    areas_a: Any,
    areas_b: Any,
    bandwidth: float,
    *,
    cutoff: float | None = None,
) -> Any:
    """Surface quadrature with bounded memory and the analytic reference gradient.

    Only cross-part pairs are used; per-link self CCD remains an independent
    guard. Areas are immutable reference areas. Neighbor pairs are refreshed
    outside autograd; bandwidth is measured in metres, not vertex count.
    """
    import torch

    class Energy(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, a: Any, b: Any) -> Any:
            if pairs is None:
                if not a.is_cuda or cutoff is None:
                    raise ValueError("implicit Gaussian neighborhoods require a GPU and cutoff")
                from ._warp_untangling import energy_gradient

                value, ga, gb = energy_gradient(a, b, areas_a, areas_b, bandwidth, cutoff)
                ctx.save_for_backward(ga, gb)
                return value
            ga, gb = torch.zeros_like(a), torch.zeros_like(b)
            value = a.new_zeros(())
            denominator = torch.sqrt(areas_a.sum() * areas_b.sum()).clamp_min(1e-20)
            eps2 = bandwidth**2
            for start in range(0, len(pairs), 65536):
                indexes = torch.as_tensor(pairs[start : start + 65536], device=a.device)
                ia, ib = indexes[:, 0], indexes[:, 1]
                delta = a[ia] - b[ib]
                coefficient = areas_a[ia] * areas_b[ib] / (eps2 * denominator)
                kernel = coefficient * torch.exp(-delta.square().sum(1) / eps2)
                value += kernel.sum()
                gradient = (-2 / eps2 * kernel)[:, None] * delta
                ga.index_add_(0, ia, gradient)
                gb.index_add_(0, ib, -gradient)
            ctx.save_for_backward(ga, gb)
            return value

        @staticmethod
        def backward(ctx: Any, gradient: Any) -> tuple[Any, Any]:
            a, b = ctx.saved_tensors
            return a * gradient, b * gradient

    return Energy.apply(left, right)  # type: ignore[no-untyped-call]


def surface_quadrature(mesh: Any, spacing: float) -> tuple[Any, Any, Any]:
    """Centroid quadrature of uniformly subdivided faces, without editing a mesh.

    A large triangle cannot repel another surface using only its distant
    corners. Barycentric samples distribute the integrated force back to its
    original vertices; every face contributes its exact reference area.
    """
    lengths = np.linalg.norm(np.roll(mesh.triangles, -1, axis=1) - mesh.triangles, axis=2).max(1)
    orders = np.maximum(1, np.ceil(lengths / spacing).astype(int))
    faces, bary, areas = [], [], []
    for n in np.unique(orders):
        indexes = np.flatnonzero(orders == n)
        uv = []
        for i in range(n):
            for j in range(n - i):
                uv.append([(i + 1 / 3) / n, (j + 1 / 3) / n])
                if i + j < n - 1:
                    uv.append([(i + 2 / 3) / n, (j + 2 / 3) / n])
        coordinates = np.asarray(uv)
        weights = np.column_stack([1 - coordinates.sum(1), coordinates])
        faces.append(np.repeat(mesh.faces[indexes], len(weights), axis=0))
        bary.append(np.tile(weights, (len(indexes), 1)))
        areas.append(np.repeat(mesh.area_faces[indexes] / len(weights), len(weights)))
    return np.concatenate(faces), np.concatenate(bary), np.concatenate(areas)


class SurfaceUntangling:
    """Shared canonical surfaces, pose-parallel objectives on available GPUs.

    The two bandwidths cover local contact and coherent patch unfolding. Unlike
    the reference's adaptive bandwidth flow, they remain fixed over an outer
    iteration so line search compares the SAME objective. No sampled triangle
    crossing is discarded. CPU fallbacks and single-GPU runs use the same math.
    """

    def __init__(self, meshes: Any, motion: Any, config: Any, devices: list[str]) -> None:
        self.parent, self.child = motion["parent"], motion["child"]
        self.bandwidths = (config.surface_support_m, 2 * config.surface_support_m)
        self.samples = {n: surface_quadrature(m, min(self.bandwidths)) for n, m in meshes.items()}
        self.clearance = config.minimum_clearance_m + config.contact_reserve_m
        self.devices = devices
        self.local_weight = 1.0
        self.pool = ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="untangle-gpu")
        self.jobs: list[Any] = []

    def close(self) -> None:
        self.pool.shutdown(wait=True, cancel_futures=True)

    def bind(self, meshes: Any, motion: Any, constraints: Any, positions: Any) -> None:
        self.jobs = []

        def samples(name: str) -> Any:
            faces, bary, _ = self.samples[name]
            return (np.asarray(meshes[name].vertices)[faces] * bary[..., None]).sum(1)

        a = samples(self.parent)
        for q in positions:
            b = samples(self.child)
            transform = joint_transform("hinge", motion["axis"], motion["origin_m"], q)
            b = b @ transform[:3, :3].T + transform[:3, 3]
            # Four bandwidths bound the omitted kernel by exp(-16).
            # Include the complete shape, not just currently intersecting faces.
            pairs = None
            if not all(d.startswith("cuda") for d in self.devices):
                neighbors = cKDTree(a).sparse_distance_matrix(
                    cKDTree(b), 4 * max(self.bandwidths), output_type="coo_matrix"
                )
                pairs = np.column_stack([neighbors.row, neighbors.col]).astype(np.int64)
            ids = np.asarray(constraints["positions"]) == q
            self.jobs.append(
                (float(q), pairs, constraints["parent_faces"][ids], constraints["child_faces"][ids])
            )

    def __call__(self, vertices: Any, axis: Any, origin: Any, transform: Any) -> tuple[Any, Any]:
        import torch

        # Autograd context is thread-local. Line search must not retain graphs.
        enabled = torch.is_grad_enabled()

        def group(item: tuple[int, list[Any]]) -> tuple[Any, Any]:
            index, jobs = item
            with torch.set_grad_enabled(enabled):
                device = self.devices[index]
                a, b = vertices[self.parent].to(device), vertices[self.child].to(device)
                ax, org = axis.to(device), origin.to(device)
                sampled = {}
                areas = {}
                for name, v in ((self.parent, a), (self.child, b)):
                    faces, bary, area = self.samples[name]
                    f = torch.as_tensor(faces, device=device)
                    w = torch.as_tensor(bary, dtype=a.dtype, device=device)
                    sampled[name] = (v[f] * w[..., None]).sum(1)
                    areas[name] = torch.as_tensor(area, dtype=a.dtype, device=device)
                aa, ab = areas[self.parent], areas[self.child]
                global_loss, local_loss = a.sum() * 0, a.sum() * 0
                for q, pairs, fa, fb in jobs:
                    moved = transform(b, q, ax, org)
                    moved_samples = transform(sampled[self.child], q, ax, org)
                    global_loss = global_loss + sum(
                        gaussian_energy(
                            sampled[self.parent],
                            moved_samples,
                            pairs,
                            aa,
                            ab,
                            h,
                            cutoff=4 * max(self.bandwidths),
                        )
                        for h in self.bandwidths
                    ) / len(self.bandwidths)
                    if len(fa):
                        gaps = minkowski_gap(
                            a[torch.as_tensor(fa, device=device)],
                            moved[torch.as_tensor(fb, device=device)],
                        )
                        deficits = torch.relu((self.clearance - gaps) / 0.001)
                        local_loss = (
                            local_loss + deficits.square().mean() + deficits.square().amax()
                        )
                return global_loss.to(axis.device), local_loss.to(axis.device)

        groups = [(i, self.jobs[i :: len(self.devices)]) for i in range(len(self.devices))]
        results = list(self.pool.map(group, groups)) if len(groups) > 1 else [group(groups[0])]
        return tuple(sum(row[k] for row in results) / max(len(self.jobs), 1) for k in (0, 1))
