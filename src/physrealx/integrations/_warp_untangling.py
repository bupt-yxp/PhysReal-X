# mypy: ignore-errors
"""Memory-linear GPU Gaussian quadrature; no materialized all-neighbor table.

Warp kernel annotations are a runtime DSL, not static Python types.
"""

from typing import Any

import warp as wp

from ._warp_surface_query import _MESH_CACHE_LOCK

wp.init()


@wp.kernel
def _integrate(
    left: wp.array(dtype=wp.vec3d),
    right: wp.array(dtype=wp.vec3d),
    area_left: wp.array(dtype=wp.float64),
    area_right: wp.array(dtype=wp.float64),
    grid: wp.uint64,
    radius: wp.float64,
    padding: float,
    eps2: wp.float64,
    normalizer: wp.float64,
    values: wp.array(dtype=wp.float64),
    gradients: wp.array(dtype=wp.vec3d),
):
    i = wp.tid()
    x = left[i]
    center = wp.vec3(wp.float32(x[0]), wp.float32(x[1]), wp.float32(x[2]))
    total = wp.float64(0.0)
    gradient = wp.vec3d(wp.float64(0.0))
    for j in wp.hash_grid_query(grid, center, wp.float32(radius) + padding):
        delta = x - right[j]
        distance2 = wp.dot(delta, delta)
        if distance2 <= radius * radius:
            value = area_left[i] * area_right[j] * wp.exp(-distance2 / eps2) / (eps2 * normalizer)
            total += value
            gradient -= (wp.float64(2.0) * value / eps2) * delta
    values[i] = total
    gradients[i] = gradient


def energy_gradient(
    a: Any, b: Any, aa: Any, ab: Any, bandwidth: float, cutoff: float
) -> tuple[Any, Any, Any]:
    """Analytic gradients in both directions, avoiding contended atomics.

    The hash is only a broad phase. Distances and energies use float64; query
    padding covers float32 grid-coordinate roundoff. A shared Torch/Warp stream
    preserves tensor lifetimes and execution ordering on each selected GPU.
    """
    import torch

    a, b = a.contiguous(), b.contiguous()
    device = str(a.device)
    normalizer = float(torch.sqrt(aa.sum() * ab.sum()).clamp_min(1e-20))
    magnitude = max(float(a.abs().max()), float(b.abs().max()), cutoff)
    padding = 8 * torch.finfo(torch.float32).eps * magnitude
    ga, gb = torch.empty_like(a), torch.empty_like(b)
    values = torch.empty(len(a), dtype=a.dtype, device=a.device)
    other_values = torch.empty(len(b), dtype=b.dtype, device=b.device)
    # Hash storage is linear in samples plus a fixed cell table, independent
    # of how many millions of neighbors the surfaces have near the hinge.
    torch_stream = torch.cuda.current_stream(a.device)
    stream = wp.stream_from_torch(torch_stream)
    for left, right, area_left, area_right, output, gradient in (
        (a, b, aa, ab, values, ga),
        (b, a, ab, aa, other_values, gb),
    ):
        grid_points = right.to(torch.float32)
        # Warp 1.17 uses process-global native allocator/stream state. Serialize
        # setup and destruction, but wait for GPU execution OUTSIDE this lock.
        with _MESH_CACHE_LOCK, wp.ScopedStream(stream):
            grid = wp.HashGrid(64, 64, 64, device=device)
            grid.build(wp.from_torch(grid_points, dtype=wp.vec3), cutoff / 4)
            wp.launch(
                _integrate,
                dim=len(left),
                device=device,
                inputs=[
                    wp.from_torch(left, dtype=wp.vec3d),
                    wp.from_torch(right, dtype=wp.vec3d),
                    wp.from_torch(area_left, dtype=wp.float64),
                    wp.from_torch(area_right, dtype=wp.float64),
                    grid.id,
                    cutoff,
                    padding,
                    bandwidth**2,
                    normalizer,
                    wp.from_torch(output, dtype=wp.float64),
                    wp.from_torch(gradient, dtype=wp.vec3d),
                ],
            )
        try:
            torch_stream.synchronize()
        finally:
            with _MESH_CACHE_LOCK, wp.ScopedStream(stream):
                del grid
    return (values.sum() + other_values.sum()) / 2, ga, gb
