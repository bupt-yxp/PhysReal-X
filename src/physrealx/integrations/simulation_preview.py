"""Synchronized visualization of one recorded simulation episode."""

from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, DatasetManifest, Episode, TaskSpec, VersionedRef
from physrealx.errors import DomainValidationError, MethodExecutionError, MethodUnavailableError
from physrealx.integrations._artifacts import load_resource, read_array
from physrealx.integrations.blender_scene import (
    OVERVIEW_CAMERA,
    OVERVIEW_TARGET,
    BlenderSensorCamera,
    render_blender_scene_frames,
)
from physrealx.integrations.sensors import CameraSpec, EventAccumulator, project_depth
from physrealx.runtime import load_episode


def _inline(step: Any, name: str) -> float:
    value = step.observations.get(name)
    if value is None or value.inline is None or len(value.inline) != 1:
        raise DomainValidationError(f"simulation preview requires scalar observation {name!r}")
    return float(value.inline[0])


def _observation_array(step: Any, name: str, store: ArtifactStore) -> Any:
    import numpy as np

    value = step.observations.get(name)
    if value is None:
        raise DomainValidationError(f"simulation preview requires array observation {name!r}")
    if value.artifact is not None:
        array = read_array(store, value.artifact)
    elif value.inline is not None:
        array = np.asarray(value.inline).reshape(value.shape)
    else:
        raise DomainValidationError(f"simulation preview observation {name!r} has no data")
    if tuple(array.shape) != value.shape:
        raise DomainValidationError(f"simulation preview observation {name!r} changed shape")
    return array


def _camera_name(episode: Episode, requested: str | None) -> str:
    prefix, suffix = "camera.", ".rgb"
    available = sorted(
        name[len(prefix) : -len(suffix)]
        for name in episode.steps[0].observations
        if name.startswith(prefix) and name.endswith(suffix)
    )
    if not available:
        raise DomainValidationError("simulation preview requires at least one recorded camera")
    if requested is not None and requested not in available:
        raise DomainValidationError(f"simulation preview camera {requested!r} is unavailable")
    return requested or available[0]


def _episode_states(
    episode: Episode, selected: Any, store: ArtifactStore
) -> list[dict[str, dict[str, Any]]]:
    """Read the exact recorded rigid-body poses used by the physics engine."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    metadata = episode.metadata.get("sensor_metadata")
    raw_order = metadata.get("body_pose_order") if isinstance(metadata, dict) else None
    if (
        not isinstance(raw_order, list)
        or not raw_order
        or not all(isinstance(name, str) for name in raw_order)
    ):
        raise DomainValidationError("simulation preview requires the recorded body-pose order")
    order = [str(name) for name in raw_order]
    states = []
    for sample in selected:
        step = episode.steps[int(sample)]
        positions = np.asarray(_observation_array(step, "body.position", store), dtype=float)
        orientations = np.asarray(_observation_array(step, "body.orientation", store), dtype=float)
        if positions.shape != (len(order), 3) or orientations.shape != (len(order), 4):
            raise DomainValidationError("simulation preview body poses do not match their order")
        state: dict[str, dict[str, Any]] = {}
        for name, position, orientation in zip(order, positions, orientations, strict=True):
            if "/" not in name:
                raise DomainValidationError(
                    "simulation preview body names must be entity/link pairs"
                )
            entity, link = name.split("/", 1)
            transform = np.eye(4)
            transform[:3, :3] = Rotation.from_quat(orientation).as_matrix()
            transform[:3, 3] = position
            state.setdefault(entity, {})[link] = transform
        states.append(state)
    return states


def _opencv_to_opengl_quaternion(orientation_xyzw: Any) -> tuple[float, float, float, float]:
    """Convert a camera-to-world OpenCV pose into Blender/OpenGL camera axes."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    orientation = np.asarray(orientation_xyzw, dtype=float)
    if (
        orientation.shape != (4,)
        or not np.isfinite(orientation).all()
        or not np.isclose(np.linalg.norm(orientation), 1.0, atol=1e-6)
    ):
        raise DomainValidationError("simulation preview camera orientation is invalid")
    rotation = Rotation.from_quat(orientation).as_matrix() @ np.diag([1.0, -1.0, -1.0])
    converted = Rotation.from_matrix(rotation).as_quat()
    return (float(converted[0]), float(converted[1]), float(converted[2]), float(converted[3]))


def _sensor_camera(
    episode: Episode,
    selected: Any,
    store: ArtifactStore,
    name: str,
) -> tuple[BlenderSensorCamera, list[Any], list[Any], CameraSpec]:
    """Build a quality-render camera from recorded calibrated per-frame poses."""
    import numpy as np

    cameras = episode.backend.config.get("cameras")
    if not isinstance(cameras, dict) or name not in cameras:
        raise DomainValidationError("simulation preview camera is absent from backend config")
    source = CameraSpec.model_validate(cameras[name])
    width = max(320, source.width)
    height = round(source.height * width / source.width)
    camera = source.model_copy(update={"width": width, "height": height})
    positions = []
    optical_orientations = []
    render_orientations = []
    for sample in selected:
        step = episode.steps[int(sample)]
        position = np.asarray(
            _observation_array(step, f"camera.{name}.position", store), dtype=float
        )
        orientation = np.asarray(
            _observation_array(step, f"camera.{name}.orientation", store), dtype=float
        )
        if position.shape != (3,) or not np.isfinite(position).all():
            raise DomainValidationError("simulation preview camera position is invalid")
        positions.append(position)
        optical_orientations.append(orientation)
        render_orientations.append(_opencv_to_opengl_quaternion(orientation))
    specification = BlenderSensorCamera(
        name=name,
        width=camera.width,
        height=camera.height,
        fovy_degrees=camera.fovy_degrees,
        max_depth_m=camera.max_depth_m,
        positions=tuple((float(item[0]), float(item[1]), float(item[2])) for item in positions),
        rotations_xyzw=tuple(render_orientations),
    )
    return specification, positions, optical_orientations, camera


def _visible_cloud(points: Any, rgb: Any, *, maximum_points: int = 100_000) -> tuple[Any, Any]:
    """Pair every valid camera point with its measured RGB colour."""
    import numpy as np

    points = np.asarray(points)
    rgb = np.asarray(rgb)
    if points.ndim != 3 or points.shape[-1] != 3 or rgb.shape != points.shape:
        raise DomainValidationError("simulation preview RGB and point-cloud shapes must agree")
    selected = np.isfinite(points).all(axis=-1) & (points[..., 2] > 0)
    cloud = points[selected]
    colours = rgb[selected].astype(float) / 255.0
    if len(cloud) > maximum_points:
        keep = np.linspace(0, len(cloud) - 1, maximum_points, dtype=int)
        cloud, colours = cloud[keep], colours[keep]
    return cloud, colours


def _world_cloud(points: Any, position: Any, orientation_xyzw: Any) -> Any:
    """Transform optical-camera points into the recorded world frame."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    points = np.asarray(points)
    position = np.asarray(position)
    orientation_xyzw = np.asarray(orientation_xyzw)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or position.shape != (3,)
        or orientation_xyzw.shape != (4,)
        or not np.isfinite(position).all()
        or not np.isfinite(orientation_xyzw).all()
        or np.linalg.norm(orientation_xyzw) < 1e-12
    ):
        raise DomainValidationError("simulation preview has an invalid recorded camera pose")
    return Rotation.from_quat(orientation_xyzw).apply(points) + position


def _blender_point_cloud_frames(clouds: list[Any], colours: list[Any]) -> list[Any]:
    """Render RGB point clouds from the same Blender view as the physics scene."""
    import numpy as np
    import trimesh
    from PIL import Image

    if not clouds or len(clouds) != len(colours):
        raise DomainValidationError("point-cloud frames and RGB colours must be aligned")
    blender = shutil.which("blender")
    if blender is None:
        raise MethodUnavailableError("simulation point-cloud preview requires Blender")
    renderer = Path(__file__).with_name("_blender_pointcloud_renderer.py")
    with tempfile.TemporaryDirectory(prefix="physrealx-blender-cloud-") as temporary:
        root = Path(temporary)
        output = root / "frames"
        output.mkdir()
        sources = []
        view = np.asarray(OVERVIEW_TARGET) - np.asarray(OVERVIEW_CAMERA)
        view /= np.linalg.norm(view)
        right = np.cross(view, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        upward = np.cross(right, view)
        corners = np.asarray((-right - upward, right - upward, right + upward, -right + upward))
        for index, (cloud, colour) in enumerate(zip(clouds, colours, strict=True)):
            cloud = np.asarray(cloud, dtype=np.float32)
            colour = np.asarray(colour, dtype=float)
            if (
                cloud.ndim != 2
                or cloud.shape[1:] != (3,)
                or colour.shape != cloud.shape
                or not len(cloud)
                or not np.isfinite(cloud).all()
                or not np.isfinite(colour).all()
            ):
                raise DomainValidationError(
                    "simulation preview contains an invalid RGB point cloud"
                )
            rgba = np.column_stack(
                (
                    np.rint(np.clip(colour, 0, 1) * 255).astype(np.uint8),
                    np.full(len(colour), 255, dtype=np.uint8),
                )
            )
            radius = float(np.clip(0.007 * np.sqrt(19_200 / len(cloud)), 0.004, 0.014))
            vertices = (cloud[:, None, :] + radius * corners[None, :, :]).reshape(-1, 3)
            base = np.arange(len(cloud), dtype=np.int64)[:, None] * 4
            faces = np.concatenate(
                (base + np.asarray([0, 1, 2]), base + np.asarray([0, 2, 3])), axis=0
            )
            source = root / f"{index:03d}.ply"
            payload = trimesh.Trimesh(
                vertices=vertices,
                faces=faces,
                vertex_colors=np.repeat(rgba, 4, axis=0),
                process=False,
            ).export(file_type="ply")
            source.write_bytes(payload if isinstance(payload, bytes) else payload.encode())
            sources.append(str(source))
        configuration = root / "render.json"
        configuration.write_text(
            json.dumps(
                {
                    "frames": sources,
                    "output_dir": str(output),
                    "width": 720,
                    "height": 480,
                    "camera": OVERVIEW_CAMERA,
                    "target": OVERVIEW_TARGET,
                }
            )
        )
        environment = os.environ.copy()
        environment["SDL_AUDIODRIVER"] = "dummy"
        result = subprocess.run(
            [blender, "-b", "--factory-startup", "-P", str(renderer), "--", str(configuration)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=max(90, len(clouds) * 12),
            env=environment,
            check=False,
        )
        paths = sorted(output.glob("*.png"))
        if len(paths) != len(clouds):
            tail = "\n".join(result.stdout.splitlines()[-20:])
            raise MethodExecutionError(f"Blender point-cloud rendering failed:\n{tail}")
        rendered = []
        for path in paths:
            with Image.open(path) as image:
                rendered.append(image.convert("RGB").copy())
        return rendered


def preview_simulation_episode(
    store: ArtifactStore,
    dataset_ref: VersionedRef,
    *,
    episode_index: int = 0,
    camera: str | None = None,
    frames: int = 12,
    fps: int = 6,
    sensor_source: str = "blender",
) -> ArtifactRef:
    """Render physics, synchronized modalities and state traces into one GIF."""
    import matplotlib
    import numpy as np
    from PIL import Image

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if sensor_source not in {"blender", "recorded"}:
        raise DomainValidationError("sensor_source must be 'blender' or 'recorded'")
    dataset = load_resource(store, dataset_ref, DatasetManifest)
    if type(episode_index) is not int or not 0 <= episode_index < len(dataset.episodes):
        raise DomainValidationError("simulation preview episode index is out of range")
    if type(frames) is not int or not 2 <= frames <= 30:
        raise DomainValidationError("simulation preview frames must be between 2 and 30")
    if type(fps) is not int or not 1 <= fps <= 20:
        raise DomainValidationError("simulation preview fps must be between 1 and 20")
    episode = load_episode(store, dataset.episodes[episode_index])
    task = load_resource(store, episode.task, TaskSpec)
    camera_name = _camera_name(episode, camera)
    required = (
        "body.position",
        "body.orientation",
        f"camera.{camera_name}.position",
        f"camera.{camera_name}.orientation",
    )
    missing = [name for name in required if name not in episode.steps[0].observations]
    if missing:
        raise DomainValidationError(f"simulation preview is missing recorded poses: {missing!r}")

    count = min(frames, len(episode.steps))
    selected = np.rint(np.linspace(0, len(episode.steps) - 1, count)).astype(int)
    states = _episode_states(episode, selected, store)
    sensor, camera_positions, camera_orientations, camera_spec = _sensor_camera(
        episode, selected, store, camera_name
    )
    rendered = render_blender_scene_frames(
        store, episode.scene, states, sensor=sensor if sensor_source == "blender" else None
    )
    scene_frames = list(rendered.overview)
    if sensor_source == "blender":
        display_rgbs = list(rendered.rgb)
        depths = list(rendered.depth)
        masks = list(rendered.mask)
        accumulator = EventAccumulator(camera_spec.event_threshold)
        events = [accumulator.update(np.asarray(rgb)) for rgb in display_rgbs]
        intrinsics_by_frame = [camera_spec.intrinsics()] * count
        source_label = "Blender PBR"
    else:

        def recorded(modality: str) -> list[Any]:
            return [
                _observation_array(episode.steps[int(i)], f"camera.{camera_name}.{modality}", store)
                for i in selected
            ]

        display_rgbs, depths, masks, events = (
            recorded(m) for m in ("rgb", "depth", "mask", "events")
        )
        intrinsics_by_frame = recorded("intrinsics")
        source_label = f"{episode.backend.name} native"
    point_clouds = []
    point_colours = []
    for rgb, depth, position, orientation, intrinsics in zip(
        display_rgbs,
        depths,
        camera_positions,
        camera_orientations,
        intrinsics_by_frame,
        strict=True,
    ):
        organized = project_depth(depth, intrinsics)
        cloud, colours = _visible_cloud(organized, np.asarray(rgb))
        cloud = _world_cloud(cloud, position, orientation)
        point_clouds.append(cloud)
        point_colours.append(colours)
    point_cloud_frames = _blender_point_cloud_frames(point_clouds, point_colours)

    times = np.asarray([step.timestamp_seconds for step in episode.steps])
    goal_signal = task.success_conditions[0].signal
    goal = np.asarray([_inline(step, goal_signal) for step in episode.steps])
    contact = np.asarray([_inline(step, "contact.force") for step in episode.steps])
    action_norm = np.asarray(
        [
            np.linalg.norm(step.action.inline)
            if step.action is not None and step.action.inline
            else np.nan
            for step in episode.steps
        ]
    )
    target = float(task.success_conditions[0].value)
    images = []
    for scene_frame, point_cloud_frame, display_rgb, depth, mask, event, sample, point_cloud in zip(
        scene_frames,
        point_cloud_frames,
        display_rgbs,
        depths,
        masks,
        events,
        selected,
        point_clouds,
        strict=True,
    ):
        step = episode.steps[int(sample)]
        figure = plt.figure(figsize=(16, 7.2), dpi=100)
        scene_axis = figure.add_axes((0.015, 0.315, 0.34, 0.55))
        scene_axis.imshow(scene_frame)
        scene_axis.set_title("Recorded physics execution")
        scene_axis.axis("off")

        axes = [
            figure.add_axes(bounds)
            for bounds in (
                (0.705, 0.60, 0.135, 0.25),
                (0.85, 0.60, 0.135, 0.25),
                (0.705, 0.315, 0.135, 0.25),
                (0.85, 0.315, 0.135, 0.25),
            )
        ]
        axes[0].imshow(display_rgb)
        axes[0].set_title(f"{camera_name} / {source_label} RGB")
        valid = depth[depth > 0]
        if valid.size:
            axes[1].imshow(
                depth,
                cmap="viridis",
                vmin=float(np.percentile(valid, 2)),
                vmax=float(np.percentile(valid, 98)),
            )
        else:
            axes[1].imshow(depth, cmap="viridis")
        axes[1].set_title("Depth (m)")
        axes[2].imshow(mask, cmap="tab20", interpolation="nearest")
        axes[2].set_title("Instance / link mask")
        signed_events = event[..., 0].astype(float) - event[..., 1].astype(float)
        limit = max(1.0, float(np.abs(signed_events).max()))
        axes[3].imshow(signed_events, cmap="coolwarm", vmin=-limit, vmax=limit)
        axes[3].set_title("Events (+/-)")
        point_axis = figure.add_axes((0.36, 0.315, 0.34, 0.55))
        point_axis.imshow(point_cloud_frame)
        point_axis.set_title(
            f"{source_label} RGB point cloud ({len(point_cloud):,} points)\n"
            f"t = {step.timestamp_seconds:.2f} s"
        )
        point_axis.axis("off")
        for axis in axes:
            axis.axis("off")

        traces = (
            (goal, "Object joint", task.success_conditions[0].signal.rsplit(".", 1)[-1]),
            (contact, "Contact force", "N"),
            (action_norm, "Robot action norm", task.action.signal.unit),
        )
        trace_bounds = (
            (0.04, 0.065, 0.29, 0.18),
            (0.355, 0.065, 0.29, 0.18),
            (0.67, 0.065, 0.29, 0.18),
        )
        for column, ((values, title, ylabel), bounds) in enumerate(
            zip(traces, trace_bounds, strict=True)
        ):
            axis = figure.add_axes(bounds)
            axis.plot(times, values, color=("#2563eb", "#dc2626", "#059669")[column])
            axis.axvline(step.timestamp_seconds, color="#111827", linestyle="--", linewidth=1)
            if column == 0:
                axis.axhline(target, color="#f59e0b", linestyle=":", label="target")
                axis.legend(loc="best", fontsize=8)
            axis.set_title(title)
            axis.set_xlabel("time (s)")
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.22)
        figure.suptitle("PhysReal-X synchronized simulation episode", fontsize=15, y=0.985)
        stream = io.BytesIO()
        figure.savefig(stream, format="png")
        plt.close(figure)
        stream.seek(0)
        with Image.open(stream) as composed:
            images.append(composed.convert("RGB").copy())
    output = io.BytesIO()
    images[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / fps),
        loop=0,
        disposal=2,
        comment=json.dumps(
            {
                "schema": "physrealx.simulation-preview/1",
                "dataset": dataset_ref.model_dump(mode="json"),
                "episode_id": episode.episode_id,
                "camera": camera_name,
                "renderer": "blender_eevee_pbr",
                "visual_source": "shared_scene_rgb_depth_mask",
                "pointcloud_source": "front_pbr_rgb+metric_depth",
                "mask_labels": rendered.labels,
                "camera_pose_convention": "OpenCV camera-to-world, xyzw",
                "camera_poses": [
                    {
                        "translation_m": [float(value) for value in position],
                        "rotation_xyzw": [float(value) for value in orientation],
                    }
                    for position, orientation in zip(
                        camera_positions, camera_orientations, strict=True
                    )
                ],
                "sample_indices": selected.tolist(),
                "fps": fps,
            }
        ).encode(),
    )
    return store.put_bytes(output.getvalue(), media_type="image/gif")
