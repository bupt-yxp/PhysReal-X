from __future__ import annotations

import io
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from physrealx import PhysRealX
from physrealx.domain import DatasetManifest, PoseSpec, VersionedRef
from physrealx.integrations.sensors import CameraSpec
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


@pytest.mark.parametrize("sensor_source", ["blender", "recorded"])
def test_preview_uses_selected_sensor_colors_and_provenance(tmp_path, monkeypatch, sensor_source):
    from physrealx.integrations import simulation_preview as preview

    px = PhysRealX.open(tmp_path, discover_plugins=False)
    camera = CameraSpec(pose=PoseSpec(), width=16, height=16)
    native_rgb = np.full((16, 16, 3), [255, 0, 0], dtype=np.uint8)
    blender_rgb = np.full((16, 16, 3), [0, 255, 0], dtype=np.uint8)

    def data(array):
        array = np.asarray(array)
        return SimpleNamespace(artifact=None, inline=tuple(array.ravel()), shape=array.shape)

    observations = {
        "body.position": data([[0, 0, 0]]),
        "body.orientation": data([[0, 0, 0, 1]]),
        "joint.position": data([0.2]),
        "contact.force": data([0.0]),
        "camera.side.position": data([0, 0, 0]),
        "camera.side.orientation": data([0, 0, 0, 1]),
        "camera.side.rgb": data(native_rgb),
        "camera.side.depth": data(np.ones((16, 16))),
        "camera.side.mask": data(np.full((16, 16), 7)),
        "camera.side.events": data(np.zeros((16, 16, 2))),
        "camera.side.intrinsics": data(camera.intrinsics()),
    }
    episode = SimpleNamespace(
        episode_id="native-episode",
        scene="scene",
        task="task",
        backend=SimpleNamespace(name="isaac_sim"),
        metadata={"sensor_metadata": {"segmentation_labels": {"7": "native/cover"}}},
        steps=[
            SimpleNamespace(
                observations=observations,
                timestamp_seconds=float(i),
                action=data([0.1]),
            )
            for i in range(2)
        ],
    )
    task = SimpleNamespace(
        success_conditions=[SimpleNamespace(signal="joint.position", value=0.3)],
        action=SimpleNamespace(signal=SimpleNamespace(unit="N*m")),
    )
    monkeypatch.setattr(
        preview,
        "load_resource",
        lambda store, ref, cls: (
            SimpleNamespace(episodes=["episode"]) if cls is DatasetManifest else task
        ),
    )
    monkeypatch.setattr(preview, "load_episode", lambda *args: episode)
    monkeypatch.setattr(preview, "_episode_states", lambda *args: [{}, {}])
    monkeypatch.setattr(
        preview,
        "_sensor_camera",
        lambda *args: (
            "blender-sensor",
            [np.zeros(3)] * 2,
            [[0, 0, 0, 1]] * 2,
            camera,
        ),
    )
    overview = Image.new("RGB", (24, 16), "white")
    monkeypatch.setattr(
        preview,
        "render_blender_scene_frames",
        lambda *args, **kwargs: SimpleNamespace(
            overview=[overview] * 2,
            rgb=[blender_rgb] * 2,
            depth=[np.ones((16, 16))] * 2,
            mask=[np.full((16, 16), 3)] * 2,
            labels={"3": "blender/cover"},
        ),
    )
    captured_colors = []

    def clouds(points, colors):
        captured_colors.extend(colors)
        return [overview] * len(points)

    monkeypatch.setattr(preview, "_blender_point_cloud_frames", clouds)
    result = preview.preview_simulation_episode(
        px.artifacts,
        VersionedRef(resource_type="dataset", resource_id="dataset", version="1.0.0"),
        frames=2,
        sensor_source=sensor_source,
    )
    with Image.open(io.BytesIO(px.artifacts.get_bytes(result))) as gif:
        metadata = json.loads(gif.info["comment"])
    expected_rgb = [0, 1, 0] if sensor_source == "blender" else [1, 0, 0]
    np.testing.assert_allclose(captured_colors[0], np.broadcast_to(expected_rgb, (256, 3)))
    assert metadata["camera"] == "side"
    assert metadata["sensor_source"] == sensor_source
    assert metadata["mask_labels"] == (
        {"3": "blender/cover"} if sensor_source == "blender" else {"7": "native/cover"}
    )
    assert metadata["pointcloud_source"] == (
        "blender.camera.side.rgb+metric_depth"
        if sensor_source == "blender"
        else "recorded.camera.side.rgb+depth"
    )
    assert metadata["events_source"] == (
        "log_intensity_crossings_between_preview_samples"
        if sensor_source == "blender"
        else "recorded_episode_observations"
    )
    assert metadata["dataset_modified"] is False
    assert metadata["scope"] == "sampled_visualization_only"
    assert metadata["training_observations_source"] == "recorded_episode"
    assert metadata["preview_frame_count"] == 2
