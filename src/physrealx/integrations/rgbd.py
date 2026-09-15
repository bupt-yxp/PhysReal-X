"""One immutable, camera-bound RGB-D input for visual reconstruction methods."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import Field

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, DomainModel
from physrealx.errors import DomainValidationError
from physrealx.evidence import strict_json_object
from physrealx.integrations._artifacts import read_array, read_json


class MetricRGBDInput(DomainModel):
    schema_id: Literal["physrealx.metric-multistate-rgbd-bundle/1"] = Field(
        default="physrealx.metric-multistate-rgbd-bundle/1", alias="schema"
    )
    camera_rig: ArtifactRef
    enhanced_depth: ArtifactRef
    reference_model: ArtifactRef


@dataclass(frozen=True)
class RGBDView:
    frame_id: str
    stream: str
    camera: dict[str, Any]
    pose: Any
    rgb: Any
    depth: Any
    rays: Any
    mask: Any


def load_rgbd(
    source: MetricRGBDInput, artifacts: ArtifactStore
) -> tuple[dict[str, Any], dict[str, Any], list[RGBDView]]:
    """Validate array bytes, view identities and world-to-camera geometry once."""
    import numpy as np

    camera = read_json(artifacts, source.camera_rig, "physrealx.camera-rig-calibration/1")
    if (
        camera.get("scale_status") != "resolved"
        or camera.get("coordinate_convention", {}).get("translation_unit") != "m"
    ):
        raise DomainValidationError("RGB-D reconstruction requires camera translations in metres")
    poses = {}
    for stream in camera["streams"]:
        for component in stream["components"]:
            intrinsics = {item["camera_id"]: item for item in component["cameras"]}
            for view in component["images"]:
                key = (stream["stream_name"], view["name"])
                if key in poses:
                    raise DomainValidationError("RGB-D camera view identities must be unique")
                poses[key] = (view, intrinsics[view["camera_id"]])
    if (
        source.enhanced_depth.media_type != "application/zip"
        or source.enhanced_depth.size_bytes > 2 * 1024**3
    ):
        raise DomainValidationError("RGB-D input requires a bounded enhanced-depth ZIP")
    result = []
    with zipfile.ZipFile(io.BytesIO(artifacts.get_bytes(source.enhanced_depth))) as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if len(set(names)) != len(names) or sum(item.file_size for item in members) > 2 * 1024**3:
            raise DomainValidationError(
                "RGB-D archive contains duplicate members or exceeds its bound"
            )
        if archive.getinfo("manifest.json").file_size > 32 * 1024**2:
            raise DomainValidationError("enhanced-depth manifest is too large")
        depth_manifest = strict_json_object(archive.read("manifest.json"), "enhanced depth")
        if (
            depth_manifest.get("schema") != "physrealx.enhanced-depth-bundle/1"
            or depth_manifest.get("source_camera_artifact_id") != source.camera_rig.artifact_id
            or depth_manifest.get("status") not in {"PASSED", "SELF_SUPERVISED"}
            or depth_manifest.get("depth_encoding", {}).get("unit") != "m"
        ):
            raise DomainValidationError("depth evidence is rejected or bound to another camera")
        expected, ids = {"manifest.json"}, set()
        for frame in depth_manifest["frames"]:
            frame_id = frame["frame_id"]
            if frame_id in ids:
                raise DomainValidationError("RGB-D frame identifiers must be unique")
            ids.add(frame_id)
            view = frame.get("view", {})
            key = (view.get("stream_name"), view.get("image_name"))
            resolved = poses.get(key)
            if resolved is None:
                raise DomainValidationError("RGB-D frame refers to an unknown camera view")
            image, intrinsics = resolved
            if image["source_artifact_id"] != view.get("source_rgb_artifact_id") or image[
                "cam_from_world"
            ] != view.get("cam_from_world"):
                raise DomainValidationError("RGB-D view differs from its immutable camera pose")
            record = frame["depth"]
            path = PurePosixPath(record["path"])
            if (
                path.is_absolute()
                or ".." in path.parts
                or "\\" in str(path)
                or str(path) in expected
            ):
                raise DomainValidationError("unsafe or reused RGB-D payload path")
            expected.add(str(path))
            payload = archive.read(str(path))
            if (
                record.get("unit") != "m"
                or record.get("dtype") != "float32"
                or record.get("media_type") != "application/x-npy"
                or len(payload) != record["size_bytes"]
            ):
                raise DomainValidationError("enhanced-depth record differs from its bytes or units")
            depth = np.load(io.BytesIO(payload), allow_pickle=False)
            rays = read_array(artifacts, ArtifactRef.model_validate(view["rays"]))
            rgb = read_array(artifacts, ArtifactRef.model_validate(frame["rgb"]))
            mask = read_array(artifacts, ArtifactRef.model_validate(view["object_mask"]))
            pose = np.asarray(image["cam_from_world"], dtype=float)
            if (
                not isinstance(depth, np.ndarray)
                or depth.dtype != np.float32
                or depth.ndim != 2
                or not np.isfinite(depth).all()
                or (depth < 0).any()
                or rays.shape != (*depth.shape, 3)
                or rgb.shape != rays.shape
                or rgb.dtype != np.uint8
                or mask.shape != depth.shape
                or not np.allclose(rays[..., 2], 1)
                or pose.shape != (3, 4)
                or not np.isfinite(pose).all()
                or not np.allclose(pose[:, :3] @ pose[:, :3].T, np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(pose[:, :3]), 1, atol=1e-6)
            ):
                raise DomainValidationError("RGB-D arrays, metric rays or rigid poses are invalid")
            result.append(RGBDView(frame_id, key[0], intrinsics, pose, rgb, depth, rays, mask > 0))
        if set(names) != expected or not result:
            raise DomainValidationError("RGB-D archive payload differs from its manifest")
    return camera, depth_manifest, result


def centered_pinhole(views: list[RGBDView]) -> dict[str, Any]:
    """Choose one centered pinhole covering calibrated views at their RGB-D resolution."""
    import pycolmap

    if not views or len({view.depth.shape for view in views}) != 1:
        raise DomainValidationError("rectification requires a common nonempty RGB-D resolution")
    height, width = views[0].depth.shape
    models = [
        pycolmap.Camera(
            model=v.camera["model"],
            width=v.camera["width"],
            height=v.camera["height"],
            params=v.camera["params"],
        )
        for v in views
    ]
    fx = min(m.focal_length_x * width / m.width for m in models)
    fy = min(m.focal_length_y * height / m.height for m in models)
    return {
        "model": "PINHOLE",
        "width": width,
        "height": height,
        "params": [fx, fy, width / 2, height / 2],
    }


def rectify_rgbd(view: RGBDView, target: dict[str, Any]) -> RGBDView:
    """Resample RGB, depth and masks together; poses and metric depth values stay fixed."""
    import cv2
    import numpy as np
    import pycolmap

    height, width = target["height"], target["width"]
    fx, fy, cx, cy = target["params"]
    y, x = np.mgrid[:height, :width]
    rays = np.column_stack(((x.ravel() - cx) / fx, (y.ravel() - cy) / fy, np.ones(height * width)))
    model = pycolmap.Camera(
        model=view.camera["model"],
        width=view.camera["width"],
        height=view.camera["height"],
        params=view.camera["params"],
    )
    source_height, source_width = view.depth.shape
    pixels = np.asarray(model.img_from_cam(rays)) * [
        source_width / model.width,
        source_height / model.height,
    ]
    mapping = pixels.reshape(height, width, 2).astype(np.float32)
    rgb = cv2.remap(view.rgb, mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR)
    depth = cv2.remap(view.depth, mapping[..., 0], mapping[..., 1], cv2.INTER_NEAREST)
    mask = (
        cv2.remap(view.mask.astype(np.uint8), mapping[..., 0], mapping[..., 1], cv2.INTER_NEAREST)
        > 0
    )
    return RGBDView(
        view.frame_id,
        view.stream,
        target,
        view.pose,
        rgb,
        np.where(mask, depth, 0),
        rays.reshape(height, width, 3),
        mask,
    )
