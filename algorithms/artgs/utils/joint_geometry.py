"""Numerically identifiable joint geometry (PhysReal-X adaptation, MIT)."""

import numpy as np


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
