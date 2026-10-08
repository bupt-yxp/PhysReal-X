"""Read-only articulation audit over immutable reconstructed visual surfaces."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from physrealx.domain import GeometryRole
from physrealx.integrations._asset_preview import link_transforms, pose_matrix
from physrealx.integrations._surface_query import SurfaceQuery


def _probes(mesh: Any, limit: int) -> NDArray[Any]:
    points = np.concatenate([np.asarray(mesh.vertices), np.asarray(mesh.triangles_center)])
    if len(points) <= limit:
        return cast(NDArray[Any], points)
    indices = np.linspace(0, len(points) - 1, limit, dtype=np.int64)
    return cast(NDArray[Any], points[indices])


def _geometry_matrix(link_matrix: Any, geometry: Any) -> NDArray[Any]:
    matrix = np.asarray(link_matrix @ pose_matrix(geometry.local_pose), dtype=float)
    matrix[:3, :3] *= np.asarray(geometry.scale, dtype=float)[None, :]
    return matrix


def _transform(points: NDArray[Any], matrix: NDArray[Any]) -> NDArray[Any]:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    return cast(NDArray[Any], (matrix @ homogeneous.T).T[:, :3])


def audit_visual_motion(
    asset: Any,
    visuals: dict[str, Any],
    links: tuple[str, str],
    query: SurfaceQuery,
    *,
    samples: int,
    probes_per_link: int,
) -> dict[str, Any]:
    """Measure source-surface consistency without treating open meshes as solids."""
    joint = asset.joints[0]
    geometries = {
        geometry.link: geometry
        for geometry in asset.geometries
        if geometry.role is GeometryRole.VISUAL and geometry.link in links
    }
    left, right = links
    local_probes = {link: _probes(visuals[link], probes_per_link) for link in links}
    signed_valid = all(
        visuals[link].is_watertight and visuals[link].is_winding_consistent for link in links
    )
    positions = np.linspace(joint.lower_limit, joint.upper_limit, samples)
    left_batches: list[NDArray[Any]] = []
    right_batches: list[NDArray[Any]] = []
    for q in positions:
        transforms = link_transforms(asset, {joint.name: float(q)})
        matrices = {link: _geometry_matrix(transforms[link], geometries[link]) for link in links}
        left_batches.append(
            _transform(local_probes[left], np.linalg.inv(matrices[right]) @ matrices[left])
        )
        right_batches.append(
            _transform(local_probes[right], np.linalg.inv(matrices[left]) @ matrices[right])
        )
    left_distance, _ = query.nearest(visuals[right], np.concatenate(left_batches))
    right_distance, _ = query.nearest(visuals[left], np.concatenate(right_batches))
    left_distance = left_distance.reshape(samples, -1)
    right_distance = right_distance.reshape(samples, -1)
    left_signed = right_signed = None
    if signed_valid:
        left_signed = query.signed_distance(visuals[right], np.concatenate(left_batches)).reshape(
            samples, -1
        )
        right_signed = query.signed_distance(visuals[left], np.concatenate(right_batches)).reshape(
            samples, -1
        )
    rows: list[dict[str, Any]] = []
    for index, q in enumerate(positions):
        penetration = (
            max(
                0.0,
                float(left_signed[index].max()),
                float(right_signed[index].max()),
            )
            if left_signed is not None and right_signed is not None
            else None
        )
        rows.append(
            {
                "q": float(q),
                "minimum_unsigned_distance_m": min(
                    float(left_distance[index].min()),
                    float(right_distance[index].min()),
                ),
                "sampled_penetration_m": penetration,
            }
        )
    worst = max((float(row["sampled_penetration_m"] or 0.0) for row in rows), default=0.0)
    peak = max(rows, key=lambda row: float(row["sampled_penetration_m"] or 0.0))
    return {
        "status": (
            "INTERSECTING"
            if signed_valid and worst > 1e-7
            else "CLEAR"
            if signed_valid
            else "INDETERMINATE_OPEN_SURFACE"
        ),
        "certified_lower_bound": False,
        "interpretation": (
            "sampled signed visual-surface intersection"
            if signed_valid
            else "unsigned proximity only; an open reconstruction has no reliable inside/outside"
        ),
        "locked_range": [joint.lower_limit, joint.upper_limit],
        "samples": samples,
        "probes_per_link": probes_per_link,
        "links": {
            link: {
                "watertight": bool(visuals[link].is_watertight),
                "winding_consistent": bool(visuals[link].is_winding_consistent),
            }
            for link in links
        },
        "sampled_max_penetration_m": worst if signed_valid else None,
        "peak_q": peak["q"] if signed_valid else None,
        "minimum_unsigned_distance_m": min(
            float(row["minimum_unsigned_distance_m"]) for row in rows
        ),
        "sweep": rows,
    }
