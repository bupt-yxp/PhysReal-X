"""Shared camera contracts, metric projection, events, and MuJoCo sensor acquisition."""

from __future__ import annotations

import io
import math
from typing import Any

from pydantic import Field

from physrealx.contracts import ArtifactStore
from physrealx.domain import DataValue, DomainModel, PoseSpec, SignalDType, SignalSpec, TaskSpec
from physrealx.errors import DomainValidationError, MethodExecutionError

CAMERA_MODALITIES = (
    "rgb",
    "depth",
    "mask",
    "pointcloud",
    "events",
    "intrinsics",
    "position",
    "orientation",
)


class CameraSpec(DomainModel):
    """Camera pose uses OpenGL axes; depth and point clouds use OpenCV optical axes."""

    pose: PoseSpec
    width: int = Field(default=320, ge=16, le=1920)
    height: int = Field(default=240, ge=16, le=1080)
    fovy_degrees: float = Field(default=45.0, gt=0, lt=180)
    max_depth_m: float = Field(default=10.0, gt=0, allow_inf_nan=False)
    event_threshold: float = Field(default=0.2, ge=0.01, le=2.0)

    @classmethod
    def look_at(
        cls, position: Any, target: Any, *, up: Any = (0, 0, 1), **config: Any
    ) -> CameraSpec:
        """Construct an OpenGL camera; reject coincident or collinear view axes."""
        import numpy as np
        from scipy.spatial.transform import Rotation

        position, target, up = (np.asarray(v, dtype=float) for v in (position, target, up))
        if any(v.shape != (3,) or not np.isfinite(v).all() for v in (position, target, up)):
            raise DomainValidationError("look-at vectors must be finite three-vectors")
        backward = position - target
        right = np.cross(up, backward)
        if min(np.linalg.norm(backward), np.linalg.norm(right)) < 1e-10:
            raise DomainValidationError("look-at position/target/up do not define a camera frame")
        backward /= np.linalg.norm(backward)
        right /= np.linalg.norm(right)
        vertical = np.cross(backward, right)
        return cls(
            pose=PoseSpec(
                translation_m=position,
                rotation_xyzw=Rotation.from_matrix(
                    np.column_stack([right, vertical, backward])
                ).as_quat(),
            ),
            **config,
        )

    def signals(self, name: str, *, frequency_hz: float) -> tuple[SignalSpec, ...]:
        """Declare synchronized modalities from this camera's actual resolution."""
        if not name or "." in name:
            raise DomainValidationError("camera name must be a nonempty signal-path segment")
        h, w = self.height, self.width
        return tuple(
            SignalSpec(
                name=f"camera.{name}.{modality}",
                shape=shape,
                dtype=dtype,
                unit=unit,
                frame="world" if modality in {"position", "orientation"} else f"camera.{name}",
                frequency_hz=frequency_hz,
            )
            for modality, shape, dtype, unit in (
                ("rgb", (h, w, 3), SignalDType.UINT8, "1"),
                ("depth", (h, w), SignalDType.FLOAT32, "m"),
                ("mask", (h, w), SignalDType.INT32, "1"),
                ("events", (h, w, 2), SignalDType.INT32, "1"),
                ("pointcloud", (h, w, 3), SignalDType.FLOAT32, "m"),
                ("intrinsics", (3, 3), SignalDType.FLOAT64, "1"),
                ("position", (3,), SignalDType.FLOAT64, "m"),
                ("orientation", (4,), SignalDType.FLOAT64, "1"),
            )
        )

    def intrinsics(self) -> Any:
        import numpy as np

        focal = self.height / (2 * math.tan(math.radians(self.fovy_degrees) / 2))
        return np.array(
            [[focal, 0, (self.width - 1) / 2], [0, focal, (self.height - 1) / 2], [0, 0, 1]]
        )


def with_sensor_signals(
    task: TaskSpec,
    cameras: dict[str, CameraSpec],
    *,
    link_count: int,
    resource_id: str,
    frequency_hz: float,
) -> TaskSpec:
    """Derive camera/body signal contracts; incompatible duplicate declarations fail validation."""
    if isinstance(link_count, bool) or not isinstance(link_count, int) or link_count < 1:
        raise DomainValidationError("body signals require a positive link count")
    signals = list(task.observations)
    for name, camera in cameras.items():
        signals.extend(camera.signals(name, frequency_hz=frequency_hz))
    for name, size, unit in (
        ("body.position", 3, "m"),
        ("body.orientation", 4, "1"),
        ("body.external_force", 3, "N"),
        ("body.external_torque", 3, "N*m"),
        ("body.wrench_origin", 3, "m"),
    ):
        signals.append(
            SignalSpec(
                name=name,
                shape=(link_count, size),
                dtype=SignalDType.FLOAT32,
                unit=unit,
                frame="world",
                frequency_hz=frequency_hz,
            )
        )
    return TaskSpec.model_validate(
        {
            **task.model_dump(),
            "resource_id": resource_id,
            "observations": signals,
        }
    )


def project_depth(depth: Any, intrinsics: Any) -> Any:
    """Organized pinhole point cloud; missing depth remains an explicit zero vector."""
    import numpy as np

    depth, intrinsics = np.asarray(depth), np.asarray(intrinsics)
    if (
        depth.ndim != 2
        or intrinsics.shape != (3, 3)
        or not np.isfinite(depth).all()
        or not np.isfinite(intrinsics).all()
        or np.any(depth < 0)
        or intrinsics[0, 0] <= 0
        or intrinsics[1, 1] <= 0
        or not np.allclose(intrinsics[2], [0, 0, 1])
        or intrinsics[0, 1] != 0
        or intrinsics[1, 0] != 0
    ):
        raise DomainValidationError(
            "projection requires nonnegative metric depth and pinhole intrinsics"
        )
    y, x = np.indices(depth.shape)
    return np.stack(
        (
            (x - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (y - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ),
        axis=-1,
    ).astype("float32")


class EventAccumulator:
    """Threshold crossings between rendered frames, with residual contrast carried forward."""

    def __init__(self, threshold: float) -> None:
        if not math.isfinite(threshold) or not 0.01 <= threshold <= 2:
            raise DomainValidationError("event threshold must be finite and in [0.01, 2]")
        self.threshold = threshold
        self.reference: Any = None

    def reset(self) -> None:
        self.reference = None

    def update(self, rgb: Any) -> Any:
        import numpy as np

        rgb = np.asarray(rgb)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise DomainValidationError("event simulation requires an RGB uint8 frame")
        gray = rgb.astype(float) @ np.array([0.2126, 0.7152, 0.0722])
        logarithm = np.log(np.maximum(gray / 255, 1 / 255))
        if self.reference is None:
            self.reference = logarithm.copy()
        if self.reference.shape != logarithm.shape:
            raise DomainValidationError("event frame resolution changed without reset")
        change = logarithm - self.reference
        crossings = np.trunc(change / self.threshold).astype(np.int32)
        self.reference += crossings * self.threshold
        return np.stack((np.maximum(crossings, 0), np.maximum(-crossings, 0)), axis=-1)


class MujocoSensors:
    """Own native render resources and read every requested modality without advancing time."""

    def __init__(
        self,
        model: Any,
        cameras: dict[str, CameraSpec],
        signals: tuple[SignalSpec, ...],
        store: ArtifactStore,
    ) -> None:
        import mujoco

        self.model, self.cameras, self.store = model, cameras, store
        self.signals = tuple(signal for signal in signals if signal.name.startswith("camera."))
        self.renderers: dict[str, Any] = {}
        self.events = {
            name: EventAccumulator(camera.event_threshold) for name, camera in cameras.items()
        }
        try:
            for signal in self.signals:
                parts = signal.name.split(".")
                if len(parts) != 3 or parts[1] not in cameras or parts[2] not in CAMERA_MODALITIES:
                    raise DomainValidationError(f"unknown camera observation: {signal.name}")
                name = parts[1]
                if name not in self.renderers:
                    camera = cameras[name]
                    self.renderers[name] = mujoco.Renderer(
                        model, height=camera.height, width=camera.width
                    )
        except BaseException:
            self.close()
            raise

    def reset(self) -> None:
        for accumulator in self.events.values():
            accumulator.reset()

    def read(self, data: Any, timestamp: float) -> dict[str, DataValue]:
        import mujoco
        import numpy as np
        from scipy.spatial.transform import Rotation

        values = {}
        for name, renderer in self.renderers.items():
            camera = self.cameras[name]
            wanted = [s for s in self.signals if s.name.split(".")[1] == name]
            modalities = {s.name.split(".")[2] for s in wanted}
            # Never render invisible collision proxies over the supplied visual surface.
            option = mujoco.MjvOption()
            option.geomgroup[3] = 0
            renderer.update_scene(data, camera=name, scene_option=option)
            arrays: dict[str, Any] = {}
            if modalities & {"rgb", "events"}:
                arrays["rgb"] = renderer.render().copy()
            if "events" in modalities:
                arrays["events"] = self.events[name].update(arrays["rgb"])
            if modalities & {"depth", "pointcloud"}:
                renderer.enable_depth_rendering()
                try:
                    depth = renderer.render().copy()
                finally:
                    renderer.disable_depth_rendering()
                depth[(depth <= 0) | (depth > camera.max_depth_m)] = 0
                arrays["depth"] = depth
                if "pointcloud" in modalities:
                    arrays["pointcloud"] = project_depth(depth, camera.intrinsics())
            if "mask" in modalities:
                renderer.enable_segmentation_rendering()
                try:
                    segmentation = renderer.render().copy()
                finally:
                    renderer.disable_segmentation_rendering()
                mask = np.zeros(segmentation.shape[:2], dtype=np.int32)
                valid = (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & (
                    segmentation[..., 0] >= 0
                )
                mask[valid] = self.model.geom_bodyid[segmentation[..., 0][valid]]
                arrays["mask"] = mask
            arrays["intrinsics"] = camera.intrinsics()
            index = self.model.camera(name).id
            arrays["position"] = data.cam_xpos[index].copy()
            # Convert camera-to-world orientation from OpenGL to OpenCV optical axes.
            arrays["orientation"] = Rotation.from_matrix(
                data.cam_xmat[index].reshape(3, 3) @ np.diag([1, -1, -1])
            ).as_quat()
            for signal in wanted:
                modality = signal.name.split(".")[2]
                pixels = arrays[modality]
                if not np.isfinite(pixels).all():
                    raise MethodExecutionError("camera generated non-finite data")
                dtype = {
                    "rgb": SignalDType.UINT8,
                    "mask": SignalDType.INT32,
                    "events": SignalDType.INT32,
                    "depth": SignalDType.FLOAT32,
                    "pointcloud": SignalDType.FLOAT32,
                }.get(modality, SignalDType.FLOAT64)
                unit = "m" if modality in {"depth", "pointcloud", "position"} else "1"
                frame = "world" if modality in {"position", "orientation"} else f"camera.{name}"
                stream = io.BytesIO()
                np.save(stream, pixels, allow_pickle=False)
                value = DataValue(
                    dtype=dtype,
                    shape=pixels.shape,
                    unit=unit,
                    frame=frame,
                    timestamp_seconds=timestamp,
                    artifact=self.store.put_bytes(
                        stream.getvalue(), media_type="application/x-npy"
                    ),
                )
                value.assert_matches(signal)
                values[signal.name] = value
        return values

    def close(self) -> None:
        for renderer in self.renderers.values():
            renderer.close()
        self.renderers.clear()
