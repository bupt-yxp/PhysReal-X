"""Numerically identifiable joint geometry (PhysReal-X adaptation, MIT)."""

import json

import numpy as np


def initial_half_transform(checkpoint, initialization_path=""):
    """Use validated reference guidance when supplied, otherwise coarse motion geometry."""
    if not initialization_path:
        return estimate_half_transform_from_checkpoint(checkpoint)
    with open(initialization_path, encoding="utf-8") as stream:
        document = json.load(stream)
    if not isinstance(document, dict) or document.get("schema") != (
        "physrealx.artgs-joint-initialization/1"
    ):
        raise ValueError("ArtGS joint initialization has an unsupported schema")
    quaternion = np.asarray(document.get("half_quaternion_wxyz"), dtype=np.float64)
    translation = np.asarray(document.get("half_translation_m"), dtype=np.float64)
    joint_type = document.get("joint_type")
    unit = document.get("observed_span_unit")
    span = document.get("observed_span")
    if (
        quaternion.shape != (4,)
        or translation.shape != (3,)
        or not np.isfinite(quaternion).all()
        or not np.isfinite(translation).all()
        or not isinstance(span, (int, float))
        or not np.isfinite(span)
        or joint_type not in ("hinge", "slide")
        or unit != ("rad" if joint_type == "hinge" else "m")
        or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-6)
    ):
        raise ValueError("ArtGS joint initialization is not a finite rigid transform")
    quaternion /= np.linalg.norm(quaternion)
    return (
        quaternion,
        translation,
        {
            "method": "reference_depth_alignment",
            "joint": document.get("joint"),
            "joint_type": joint_type,
            "observed_span": float(span),
            "observed_span_unit": unit,
        },
    )


def estimate_half_transform_from_checkpoint(checkpoint):
    """Load coarse two-state motion groups and estimate their half transform."""
    from pathlib import Path

    root = Path(checkpoint)
    return estimate_half_transform(
        np.load(root / "xyz_dynamic_0.npy"),
        np.load(root / "xyz_dynamic_1.npy"),
    )


def estimate_half_transform(source, target, *, max_points=10_000):
    """Estimate the symmetric half transform between two rigid point sets.

    PCA supplies the finite set of axis-sign hypotheses. A trimmed symmetric
    nearest-neighbour score selects the rigid hypothesis and rejects rotations
    that do not improve on centroid-aligned translation.
    """
    from itertools import product

    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    clouds = []
    for name, value in (("source", source), ("target", target)):
        points = np.asarray(value, dtype=np.float64)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or len(points) < 32
            or not np.isfinite(points).all()
        ):
            raise ValueError(f"{name} motion cloud must contain at least 32 finite 3-D points")
        if len(points) > max_points:
            points = points[np.linspace(0, len(points) - 1, max_points, dtype=int)]
        clouds.append(points)
    source, target = clouds
    source_center, target_center = source.mean(0), target.mean(0)
    source_values, source_axes = np.linalg.eigh(np.cov((source - source_center).T))
    target_values, target_axes = np.linalg.eigh(np.cov((target - target_center).T))
    if source_values[-1] <= 1e-12 or target_values[-1] <= 1e-12:
        raise ValueError("motion clouds do not span an identifiable rigid part")
    source_tree, target_tree = cKDTree(source), cKDTree(target)

    def score(rotation):
        translation = target_center - rotation @ source_center
        transformed = source @ rotation.T + translation
        inverse = (target - translation) @ rotation
        distances = np.r_[
            target_tree.query(transformed, k=1)[0], source_tree.query(inverse, k=1)[0]
        ]
        keep = max(1, int(0.95 * len(distances)))
        rmse = float(np.sqrt(np.mean(np.partition(distances * distances, keep - 1)[:keep])))
        return rmse, translation

    candidates = []
    for signs in product((-1.0, 1.0), repeat=3):
        rotation = target_axes @ np.diag(signs) @ source_axes.T
        if np.linalg.det(rotation) > 0:
            rmse, translation = score(rotation)
            angle = float(np.linalg.norm(Rotation.from_matrix(rotation).as_rotvec()))
            candidates.append((rmse, angle, rotation, translation))
    best_rmse = min(row[0] for row in candidates)
    # Thin or bilaterally symmetric parts can make a flipped PCA basis score
    # almost identically to the physical motion. Treat fits inside the noise
    # band as geometrically indistinguishable and take the minimum rotation.
    # This makes the two-state estimate stable without consulting reference
    # geometry or imposing an object-specific angle.
    equivalent = [row for row in candidates if row[0] <= 1.2 * best_rmse]
    best = min(equivalent, key=lambda row: (row[1], row[0]))
    translation_rmse, translation = score(np.eye(3))
    if best[0] >= 0.98 * translation_rmse:
        full_rotation = np.eye(3)
        full_translation = translation
        fit_rmse = translation_rmse
    else:
        fit_rmse, _, full_rotation, full_translation = best

    full_rotvec = Rotation.from_matrix(full_rotation).as_rotvec()
    half_rotation = Rotation.from_rotvec(0.5 * full_rotvec)
    half_matrix = half_rotation.as_matrix()
    half_translation, _, _, _ = np.linalg.lstsq(
        np.eye(3) + half_matrix, full_translation, rcond=None
    )
    xyzw = half_rotation.as_quat()
    quaternion = np.r_[xyzw[3], xyzw[:3]]
    diagnostics = {
        "geometry_full_rotation_degrees": float(np.rad2deg(np.linalg.norm(full_rotvec))),
        "geometry_fit_rmse_m": fit_rmse,
        "translation_only_rmse_m": translation_rmse,
        "equivalent_rotation_hypotheses": len(equivalent),
    }
    return quaternion, half_translation, diagnostics


def rotation_axis_position(rotation, translation, direction):
    """Closest axis point from t = (I - R) p, with direction dot p = 0.

    I - R is singular for every proper 3-D rotation. The axis constraint
    removes that nullspace without inventing a zero origin on inversion failure.
    """
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,) or direction.shape != (3,):
        raise ValueError("joint transform requires a 3 by 3 rotation and three-vectors")
    if not all(np.isfinite(value).all() for value in (rotation, translation, direction)):
        raise ValueError("joint transform must be finite")
    norm = np.linalg.norm(direction)
    if norm <= 1e-12 or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError("joint transform requires a nonzero axis and an orthogonal rotation")
    direction = direction / norm
    if not np.isclose(np.linalg.det(rotation), 1, atol=1e-5):
        raise ValueError("joint transform cannot contain a reflection")
    if not np.allclose(rotation @ direction, direction, atol=1e-5):
        raise ValueError("joint axis must be fixed by its rotation")
    matrix = np.vstack((np.eye(3) - rotation, direction))
    position, _, rank, _ = np.linalg.lstsq(matrix, np.r_[translation, 0], rcond=None)
    if rank < 3:
        raise ValueError("zero rotation cannot identify an axis position")
    return position
