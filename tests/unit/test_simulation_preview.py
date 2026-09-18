from __future__ import annotations

import numpy as np

from physrealx.integrations.simulation_preview import (
    _opencv_to_opengl_quaternion,
    _visible_cloud,
    _world_cloud,
)


def test_opencv_camera_axes_convert_to_blender_axes() -> None:
    from scipy.spatial.transform import Rotation

    converted = _opencv_to_opengl_quaternion([0.0, 0.0, 0.0, 1.0])

    np.testing.assert_allclose(
        Rotation.from_quat(converted).as_matrix(),
        np.diag([1.0, -1.0, -1.0]),
        atol=1e-7,
    )


def test_visible_cloud_keeps_valid_points_and_orients_image_height_upward() -> None:
    points = np.array(
        [
            [[-1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            [[7.0, 8.0, 0.0], [10.0, 11.0, 12.0]],
        ],
        dtype=np.float32,
    )
    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    cloud, colours = _visible_cloud(points, rgb)

    np.testing.assert_allclose(
        cloud,
        [[-1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [10.0, 11.0, 12.0]],
    )
    np.testing.assert_allclose(colours, [[1, 0, 0], [0, 1, 0], [1, 1, 1]])


def test_world_cloud_applies_recorded_camera_pose() -> None:
    cloud = _world_cloud([[1.0, 2.0, 3.0]], [0.5, -0.5, 1.0], [0.0, 0.0, 0.0, 1.0])

    np.testing.assert_allclose(cloud, [[1.5, 1.5, 4.0]])
