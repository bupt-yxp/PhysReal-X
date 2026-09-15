"""Convert calibrated metric observations to DTArt's two-state RGB-D format."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.errors import DomainValidationError
from physrealx.integrations.rgbd import MetricRGBDInput, centered_pinhole, load_rgbd, rectify_rgbd


def prepare_dtart_capture(
    source: MetricRGBDInput,
    artifacts: ArtifactStore,
    destination: Path,
    *,
    state_streams: tuple[str, str],
    training_frames_per_state: int,
) -> dict[str, Any]:
    import cv2
    import numpy as np
    import yaml

    camera, depth, views = load_rgbd(source, artifacts)
    if len(set(state_streams)) != 2 or set(state_streams) != {v.stream for v in views}:
        raise DomainValidationError("DTArt requires exactly two distinct capture states")
    if (
        isinstance(training_frames_per_state, bool)
        or not isinstance(training_frames_per_state, int)
        or not 2 <= training_frames_per_state <= 2048
    ):
        raise DomainValidationError("DTArt training_frames_per_state must be in 2..2048")
    target = centered_pinhole(views)
    height, width = target["height"], target["width"]
    fx, fy = target["params"][:2]
    destination.mkdir(parents=True, exist_ok=False)
    np.savetxt(destination / "cam_K.txt", [[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])
    count = max(4, (3 * training_frames_per_state + 1) // 2)
    keyframes: dict[str, Any] = {}
    validation: dict[str, Any] = {}
    training_ids: list[str] = []
    validation_ids: list[str] = []
    for state, stream in enumerate(state_streams):
        available = [v for v in views if v.stream == stream and v.mask.sum() >= 100]
        if len(available) < count:
            raise DomainValidationError(
                f"DTArt requires {count} valid views in {stream}; found {len(available)}"
            )
        for index, selected in enumerate(np.linspace(0, len(available) - 1, count, dtype=int)):
            view = available[selected]
            rectified = rectify_rgbd(view, target)
            rgb, z, mask = rectified.rgb, rectified.depth, rectified.mask
            if not np.isfinite(z).all() or (z < 0).any() or z.max() > 65.535 or not (z > 0).any():
                raise DomainValidationError(
                    f"DTArt view {view.frame_id} lacks representable metric depth"
                )
            held_out = index % 3 == 0
            folder = destination / "validation" if held_out else destination
            name = f"{state}_{index:06d}"
            for subfolder, image in (
                (
                    "color_segmented",
                    cv2.cvtColor(np.where(mask[..., None], rgb, 0), cv2.COLOR_RGB2BGR),
                ),
                ("depth_filtered", np.rint(z * 1000).astype(np.uint16)),
                ("mask", mask.astype(np.uint8) * 255),
            ):
                path = folder / subfolder / f"{name}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(path), image):
                    raise DomainValidationError("DTArt RGB-D publication failed")
            w2c = np.eye(4)
            w2c[:3] = view.pose
            c2w = np.linalg.inv(w2c)
            c2w[:3, 1:3] *= -1
            row = {"cam_in_ob": c2w.tolist(), "time": state}
            (validation if held_out else keyframes)[f"frame_{name}"] = row
            (validation_ids if held_out else training_ids).append(view.frame_id)
    (destination / "init_keyframes.yml").write_text(yaml.safe_dump(keyframes), encoding="utf-8")
    (destination / "validation" / "init_keyframes.yml").write_text(
        yaml.safe_dump(validation), encoding="utf-8"
    )
    return {
        "state_pair": list(state_streams),
        "coordinate_frame": camera["coordinate_frame"],
        "training_frame_ids": training_ids,
        "validation_frame_ids": validation_ids,
        "metric_evidence_scope": camera.get("metric_evidence_scope", "model_conditioned"),
        "depth_evaluation_scope": depth.get("evaluation_scope", "masked_observed_pixels"),
    }
