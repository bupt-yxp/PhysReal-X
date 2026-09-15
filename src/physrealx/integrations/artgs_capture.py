"""Rectify canonical RGB-D observations into the pinned ArtGS camera convention."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, model_validator
from typing_extensions import Self

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, DomainModel, canonical_json_bytes
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import read_json
from physrealx.integrations.registration import (
    ReferenceDepthAlignment,
    RegistrationConfig,
    fit_loaded_reference_depth_alignment,
)
from physrealx.integrations.rgbd import (
    MetricRGBDInput,
    RGBDView,
    centered_pinhole,
    load_rgbd,
    rectify_rgbd,
)
from physrealx.reconstruction import joint_transform


class ArtGSJointInitialization(DomainModel):
    """Reference-guided two-state motion expressed in ArtGS capture coordinates."""

    schema_id: Literal["physrealx.artgs-joint-initialization/1"] = Field(
        default="physrealx.artgs-joint-initialization/1", alias="schema"
    )
    joint: str
    joint_type: Literal["hinge", "slide"]
    half_quaternion_wxyz: tuple[float, float, float, float]
    half_translation_m: tuple[float, float, float]
    observed_span: float
    observed_span_unit: Literal["rad", "m"]

    @model_validator(mode="after")
    def validate_transform(self) -> Self:
        import numpy as np

        quaternion = np.asarray(self.half_quaternion_wxyz, dtype=float)
        translation = np.asarray(self.half_translation_m, dtype=float)
        if (
            not np.isfinite(quaternion).all()
            or not np.isfinite(translation).all()
            or not np.isfinite(self.observed_span)
            or not np.isclose(np.linalg.norm(quaternion), 1, atol=1e-6)
            or self.observed_span_unit != ("rad" if self.joint_type == "hinge" else "m")
        ):
            raise ValueError("ArtGS joint initialization must be finite, rigid and unit-consistent")
        return self


class ArtGSPreparedInput(DomainModel):
    scene_id: Literal["source"] = "source"
    coordinate_frame: str
    state_pair: tuple[str, str]
    source_camera: ArtifactRef
    source_depth: ArtifactRef
    num_slots: int = Field(ge=2, le=32)
    reference_link_count: int = Field(ge=1)
    training_frames: int = Field(ge=4)
    validation_frames: int = Field(ge=4)
    training_frame_ids: tuple[str, ...]
    validation_frame_ids: tuple[str, ...]
    initial_point_count: int = Field(ge=1)
    metric_evidence_scope: str
    depth_evaluation_scope: str
    reference_depth_alignment: ReferenceDepthAlignment | None = None
    joint_initialization_file: str | None = None


def _artgs_joint_initialization(
    alignment: ReferenceDepthAlignment,
) -> ArtGSJointInitialization:
    """Convert two fitted USD joint positions to ArtGS's symmetric half transform."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    reference_rotation = np.asarray(alignment.rotation, dtype=float)
    reference_translation = np.asarray(alignment.translation_m, dtype=float)
    capture_to_reference = np.eye(4)
    capture_to_reference[:3, :3] = reference_rotation
    capture_to_reference[:3, 3] = reference_translation
    positions = [alignment.evaluations[state].joint_position for state in alignment.state_order]
    state_transforms = [
        joint_transform(
            alignment.joint_type,
            alignment.joint_axis,
            alignment.joint_origin_m,
            position,
        )
        for position in positions
    ]
    relative_reference = state_transforms[1] @ np.linalg.inv(state_transforms[0])
    relative_capture = (
        np.linalg.inv(capture_to_reference) @ relative_reference @ capture_to_reference
    )
    half_rotation = Rotation.from_rotvec(
        0.5 * Rotation.from_matrix(relative_capture[:3, :3]).as_rotvec()
    )
    half_matrix = half_rotation.as_matrix()
    half_translation, _, rank, _ = np.linalg.lstsq(
        np.eye(3) + half_matrix, relative_capture[:3, 3], rcond=None
    )
    reconstructed = np.eye(4)
    reconstructed[:3, :3] = half_matrix @ half_matrix
    reconstructed[:3, 3] = (np.eye(3) + half_matrix) @ half_translation
    if rank < 3 or not np.allclose(reconstructed, relative_capture, atol=1e-7):
        raise DomainValidationError("reference joint motion has no stable ArtGS half transform")
    xyzw = half_rotation.as_quat()
    return ArtGSJointInitialization(
        joint=alignment.observed_joint,
        joint_type=alignment.joint_type,
        half_quaternion_wxyz=(
            float(xyzw[3]),
            float(xyzw[0]),
            float(xyzw[1]),
            float(xyzw[2]),
        ),
        half_translation_m=tuple(float(value) for value in half_translation),
        observed_span=float(positions[1] - positions[0]),
        observed_span_unit="rad" if alignment.joint_type == "hinge" else "m",
    )


def prepare_artgs_capture(
    source: MetricRGBDInput,
    artifacts: ArtifactStore,
    destination: Path,
    *,
    num_slots: int,
    state_streams: tuple[str, str] | None = None,
    training_frames_per_state: int | None = None,
    max_initial_points: int = 1_000_000,
    reference_joint: str | None = None,
    seed: int = 2026,
) -> ArtGSPreparedInput:
    import cv2
    import numpy as np

    if destination.exists():
        raise FileExistsError(f"ArtGS preparation destination already exists: {destination}")
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
    if training_frames_per_state is not None:
        if (
            isinstance(training_frames_per_state, bool)
            or not isinstance(training_frames_per_state, int)
            or not 2 <= training_frames_per_state <= 2048
        ):
            raise DomainValidationError("training_frames_per_state must be an integer in 2..2048")
        # Every third selected view is held out. Invert that split for an exact train count.
        required = max(4, (3 * training_frames_per_state + 1) // 2)
        available = {name: len(groups[name]) for name in streams}
        if count < required:
            raise DomainValidationError(
                f"ArtGS needs {required} valid views per state for "
                f"{training_frames_per_state} training views and "
                f"{required - training_frames_per_state} held-out views; available: {available}"
            )
        count = required
    if (
        isinstance(max_initial_points, bool)
        or not isinstance(max_initial_points, int)
        or not 4096 <= max_initial_points <= 2_000_000
    ):
        raise DomainValidationError("max_initial_points must be an integer in 4096..2000000")
    train_count = count - len(range(0, count, 3))
    points_per_view = max_initial_points // (2 * train_count)
    if points_per_view < 1:
        raise DomainValidationError("initial point budget must cover every training view")
    target = centered_pinhole(frames)
    height, width = target["height"], target["width"]
    fx, fy = target["params"][:2]
    y, x = np.mgrid[:height, :width]
    normalized = np.column_stack(((x.ravel() - width / 2) / fx, (y.ravel() - height / 2) / fy))
    heldout = set(range(0, count, 3))
    selected_by_state = {
        state: [
            groups[stream][int(index)] for index in np.linspace(0, len(groups[stream]) - 1, count)
        ]
        for state, stream in zip(("start", "end"), streams, strict=True)
    }
    training_ids = [
        frame.frame_id
        for selected in selected_by_state.values()
        for index, frame in enumerate(selected)
        if index not in heldout
    ]
    validation_ids = [
        frame.frame_id
        for selected in selected_by_state.values()
        for index, frame in enumerate(selected)
        if index in heldout
    ]
    alignment = None
    initialization = None
    if reference_joint is not None:
        if num_slots != 2:
            raise DomainValidationError(
                "reference-guided ArtGS currently requires one root and one moving slot"
            )
        alignment = fit_loaded_reference_depth_alignment(
            source,
            artifacts,
            camera,
            frames,
            reference,
            training_ids,
            validation_ids,
            RegistrationConfig(joint=reference_joint, seed=seed),
        )
        initialization = _artgs_joint_initialization(alignment)
    destination.mkdir(parents=True, exist_ok=False)
    state_points: list[list[Any]] = [[], []]
    state_colors: list[list[Any]] = [[], []]
    transforms: dict[str, Any] = {
        "camera_angle_x": 2 * math.atan(width / (2 * fx)),
        "camera_angle_y": 2 * math.atan(height / (2 * fy)),
    }
    for state_index, state in enumerate(("start", "end")):
        selected = selected_by_state[state]
        train, test = [], []
        for index, frame in enumerate(selected):
            rectified = rectify_rgbd(frame, target)
            rgb, z, mask = rectified.rgb, rectified.depth, rectified.mask
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
            else:
                train.append(row)
                valid = np.flatnonzero(mask & (z > 0))
                if not len(valid):
                    raise DomainValidationError(
                        f"training view {frame.frame_id} has no valid depth"
                    )
                # Bound seeding memory independently of the number of training images.
                # Each train view contributes; held-out depth never seeds Gaussians.
                if len(valid) > points_per_view:
                    valid = valid[np.linspace(0, len(valid) - 1, points_per_view, dtype=int)]
                xyz = (
                    np.column_stack((normalized[valid], np.ones(len(valid))))
                    * z.ravel()[valid, None]
                )
                xyz = (xyz - frame.pose[:, 3]) @ frame.pose[:, :3]
                state_points[state_index].append(xyz)
                state_colors[state_index].append(rgb.reshape(-1, 3)[valid])
        for split, rows in (("train", train), ("test", test)):
            document: dict[str, Any] = {**transforms, "frames": rows}
            (destination / f"transforms_{split}_{state}.json").write_bytes(
                canonical_json_bytes(document)
            )
    # ArtGS first fits each observed state independently. It therefore needs
    # one metric seed cloud per state; a merged cloud makes both coarse models
    # start from the same two-state geometry and corrupts motion separation.
    xyz_by_state = tuple(np.concatenate(group) for group in state_points)
    rgb_by_state = tuple(np.concatenate(group) for group in state_colors)

    def write_point_cloud(path: Path, xyz: Any, rgb: Any) -> None:
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
        path.write_bytes(("\n".join(header) + "\nend_header\n").encode() + vertices.tobytes())

    # The merged file remains part of the upstream reader contract. Coarse
    # fitting explicitly consumes the two state-specific files below.
    xyz, rgb = np.concatenate(xyz_by_state), np.concatenate(rgb_by_state)
    write_point_cloud(destination / "points3d.ply", xyz, rgb)
    for state, state_xyz, state_rgb in zip(
        ("start", "end"), xyz_by_state, rgb_by_state, strict=True
    ):
        write_point_cloud(destination / f"point_cloud_{state}.ply", state_xyz, state_rgb)
    (destination / "transforms_train.json").write_bytes(canonical_json_bytes(transforms))
    initialization_file = None
    if initialization is not None:
        initialization_file = "reference_joint_initialization.json"
        (destination / initialization_file).write_bytes(
            canonical_json_bytes(initialization.model_dump(mode="json", by_alias=True))
        )
    return ArtGSPreparedInput(
        state_pair=(streams[0], streams[1]),
        coordinate_frame=camera.get("coordinate_frame", "capture_world"),
        source_camera=source.camera_rig,
        source_depth=source.enhanced_depth,
        num_slots=num_slots,
        reference_link_count=len(reference["links"]),
        training_frames=len(training_ids),
        validation_frames=len(validation_ids),
        training_frame_ids=tuple(training_ids),
        validation_frame_ids=tuple(validation_ids),
        initial_point_count=len(xyz),
        metric_evidence_scope=camera.get("metric_evidence_scope", "independent_metric_alignment"),
        depth_evaluation_scope=depth.get("evaluation_scope", "independent_reference"),
        reference_depth_alignment=alignment,
        joint_initialization_file=initialization_file,
    )
