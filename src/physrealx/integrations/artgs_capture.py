"""Rectify canonical RGB-D observations into the pinned ArtGS camera convention."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, DomainModel, canonical_json_bytes
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import read_json
from physrealx.integrations.rgbd import MetricRGBDInput, RGBDView, load_rgbd


class ArtGSPreparedInput(DomainModel):
    scene_id: Literal["source"] = "source"
    coordinate_frame: str
    source_camera: ArtifactRef
    source_depth: ArtifactRef
    num_slots: int = Field(ge=2, le=32)
    reference_link_count: int = Field(ge=1)
    training_frames: int = Field(ge=4)
    validation_frames: int = Field(ge=4)
    training_frame_ids: tuple[str, ...]
    validation_frame_ids: tuple[str, ...]
    metric_evidence_scope: str
    depth_evaluation_scope: str


def prepare_artgs_capture(
    source: MetricRGBDInput,
    artifacts: ArtifactStore,
    destination: Path,
    *,
    num_slots: int,
    state_streams: tuple[str, str] | None = None,
) -> ArtGSPreparedInput:
    import cv2
    import numpy as np
    import pycolmap

    camera, depth, frames = load_rgbd(source, artifacts)
    reference = read_json(artifacts, source.reference_model, "physrealx.usd-reference/1")
    groups: dict[str, list[RGBDView]] = {}
    for frame in frames:
        if frame.mask.sum() >= 100:
            groups.setdefault(frame.stream, []).append(frame)
    streams = tuple(sorted(groups)) if state_streams is None else state_streams
    if len(streams) != 2 or len(set(streams)) != 2 or set(streams) != set(groups):
        raise DomainValidationError("ArtGS requires exactly two distinct observed capture states")
    count = min(len(groups[name]) for name in streams)
    if count < 4:
        raise DomainValidationError("ArtGS requires two train and two held-out views per state")
    if len({frame.depth.shape for frame in frames}) != 1:
        raise DomainValidationError("ArtGS requires a common RGB-D image size")
    height, width = frames[0].depth.shape
    models = {
        frame.frame_id: pycolmap.Camera(
            model=frame.camera["model"],
            width=frame.camera["width"],
            height=frame.camera["height"],
            params=frame.camera["params"],
        )
        for frame in frames
    }
    fx = min(model.focal_length_x * width / model.width for model in models.values())
    fy = min(model.focal_length_y * height / model.height for model in models.values())
    y, x = np.mgrid[:height, :width]
    normalized = np.column_stack(((x.ravel() - width / 2) / fx, (y.ravel() - height / 2) / fy))
    destination.mkdir(parents=True, exist_ok=False)
    training_ids, validation_ids, points, colors = [], [], [], []
    transforms: dict[str, Any] = {
        "camera_angle_x": 2 * math.atan(width / (2 * fx)),
        "camera_angle_y": 2 * math.atan(height / (2 * fy)),
    }
    for state_index, (state, stream) in enumerate(zip(("start", "end"), streams)):
        selected = [
            groups[stream][int(index)] for index in np.linspace(0, len(groups[stream]) - 1, count)
        ]
        heldout = set(range(0, count, 3))
        train, test = [], []
        for index, frame in enumerate(selected):
            # Renderers assume a centered pinhole camera. Apply the original
            # distortion model jointly to RGB, depth and object masks.
            model = models[frame.frame_id]
            pixels = np.asarray(
                model.img_from_cam(np.column_stack((normalized, np.ones(height * width))))
            )
            pixels *= np.array([width / model.width, height / model.height])
            mapping = pixels.reshape(height, width, 2).astype(np.float32)
            rgb = cv2.remap(frame.rgb, mapping[..., 0], mapping[..., 1], cv2.INTER_LINEAR)
            z = cv2.remap(frame.depth, mapping[..., 0], mapping[..., 1], cv2.INTER_NEAREST)
            mask = (
                cv2.remap(
                    frame.mask.astype(np.uint8), mapping[..., 0], mapping[..., 1], cv2.INTER_NEAREST
                )
                > 0
            )
            z = np.where(mask, z, 0)
            if z.max() > 65.535:
                raise DomainValidationError(
                    "ArtGS uint16 millimetre depth cannot represent this capture"
                )
            rgba = np.dstack((rgb, mask.astype(np.uint8) * 255))
            image_path = destination / state / "rgba" / f"{index:06d}.png"
            depth_path = destination / state / "depth" / f"{index:06d}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            depth_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(
                str(image_path), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            ) or not cv2.imwrite(str(depth_path), np.rint(z * 1000).astype(np.uint16)):
                raise DomainValidationError("ArtGS rectified image publication failed")
            w2c = np.eye(4)
            w2c[:3] = frame.pose
            c2w = np.linalg.inv(w2c)
            c2w[:3, 1:3] *= -1
            row = {
                "file_path": image_path.relative_to(destination).as_posix(),
                "time": state_index,
                "transform_matrix": c2w.tolist(),
            }
            if index in heldout:
                test.append(row)
                validation_ids.append(frame.frame_id)
            else:
                train.append(row)
                training_ids.append(frame.frame_id)
                valid = mask & (z > 0)
                xyz = np.column_stack((normalized, np.ones(height * width))) * z.ravel()[:, None]
                xyz = (xyz - frame.pose[:, 3]) @ frame.pose[:, :3]
                points.append(xyz[valid.ravel()])
                colors.append(rgb.reshape(-1, 3)[valid.ravel()])
        for split, rows in (("train", train), ("test", test)):
            document: dict[str, Any] = {**transforms, "frames": rows}
            (destination / f"transforms_{split}_{state}.json").write_bytes(
                canonical_json_bytes(document)
            )
    # Coarse fitting also uses --eval; held-out geometry never initializes Gaussians.
    xyz, rgb = np.concatenate(points), np.concatenate(colors)
    fields = [(name, "<f4") for name in ("x", "y", "z", "nx", "ny", "nz")]
    fields += [(name, "u1") for name in ("red", "green", "blue")]
    vertices = np.zeros(len(xyz), dtype=fields)
    for column, name in enumerate(("x", "y", "z")):
        vertices[name] = xyz[:, column]
    for column, name in enumerate(("red", "green", "blue")):
        vertices[name] = rgb[:, column]
    # ArtGS's initialization format includes normals; Gaussian initialization
    # uses positions/colors, so absent normal estimates are explicitly zero.
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(xyz)}"]
    header += [f"property float {name}" for name in ("x", "y", "z", "nx", "ny", "nz")]
    header += [f"property uchar {name}" for name in ("red", "green", "blue")]
    (destination / "points3d.ply").write_bytes(
        ("\n".join(header) + "\nend_header\n").encode() + vertices.tobytes()
    )
    (destination / "transforms_train.json").write_bytes(canonical_json_bytes(transforms))
    return ArtGSPreparedInput(
        coordinate_frame=camera.get("coordinate_frame", "capture_world"),
        source_camera=source.camera_rig,
        source_depth=source.enhanced_depth,
        num_slots=num_slots,
        reference_link_count=len(reference["links"]),
        training_frames=len(training_ids),
        validation_frames=len(validation_ids),
        training_frame_ids=tuple(training_ids),
        validation_frame_ids=tuple(validation_ids),
        metric_evidence_scope=camera.get("metric_evidence_scope", "independent_metric_alignment"),
        depth_evaluation_scope=depth.get("evaluation_scope", "independent_reference"),
    )
