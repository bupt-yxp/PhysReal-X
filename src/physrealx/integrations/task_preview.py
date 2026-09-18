"""Blender visualization for an executed scene task."""

from __future__ import annotations

import io
import json
from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.domain import (
    ArtifactRef,
    AssetSpec,
    SceneSpec,
    TaskSpec,
    VersionedRef,
)
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import load_resource, read_json
from physrealx.integrations._asset_preview import link_transforms, pose_matrix
from physrealx.integrations.blender_scene import render_blender_scene_frames


def _same_resource(left: VersionedRef, right: VersionedRef) -> bool:
    return (left.resource_type, left.resource_id, left.version, left.artifact_id) == (
        right.resource_type,
        right.resource_id,
        right.version,
        right.artifact_id,
    )


def _trajectory_arrays(document: dict[str, Any], task: VersionedRef) -> tuple[Any, ...]:
    import numpy as np

    try:
        trajectory_task = VersionedRef.model_validate(document["task"])
        config = document["backend_config"]
        controlled = tuple(config["controlled_joints"])
        commands = np.asarray(document["commands"], dtype=float)
        observed = np.asarray(document["observed_positions"], dtype=float)
        passive = {
            str(name): np.asarray(values, dtype=float)
            for name, values in document["observed_passive_positions"].items()
        }
        dt = float(document["dt_seconds"])
    except (KeyError, TypeError, ValueError) as error:
        raise DomainValidationError(
            "task preview requires a complete executed trajectory"
        ) from error
    if not _same_resource(trajectory_task, task):
        raise DomainValidationError("trajectory and task references do not match")
    if (
        commands.ndim != 2
        or commands.shape != observed.shape
        or commands.shape[0] < 2
        or commands.shape[1] != len(controlled)
        or not np.isfinite(commands).all()
        or not np.isfinite(observed).all()
        or not np.isfinite(dt)
        or dt <= 0
        or any(
            values.shape != (len(commands),) or not np.isfinite(values).all()
            for values in passive.values()
        )
    ):
        raise DomainValidationError("task preview trajectory arrays are inconsistent")
    return controlled, commands, observed, passive, dt


def preview_task_execution(
    store: ArtifactStore,
    scene_ref: VersionedRef,
    task_ref: VersionedRef,
    trajectory: ArtifactRef,
    *,
    frames: int = 16,
    fps: int = 8,
) -> ArtifactRef:
    """Render the executed scene and robot trajectory from the shared Blender PBR scene."""
    import numpy as np
    from PIL import Image

    if type(frames) is not int or not 2 <= frames <= 60:
        raise DomainValidationError("preview frames must be an integer between 2 and 60")
    if type(fps) is not int or not 1 <= fps <= 30:
        raise DomainValidationError("preview fps must be an integer between 1 and 30")
    scene = load_resource(store, scene_ref, SceneSpec)
    task = load_resource(store, task_ref, TaskSpec)
    if not _same_resource(task.scene, scene_ref):
        raise DomainValidationError("task does not reference the supplied scene")
    document = read_json(store, trajectory, "physrealx.joint-trajectory/1")
    controlled, commands, observed, passive, _ = _trajectory_arrays(document, task_ref)
    assets = {
        entity.name: load_resource(store, entity.asset, AssetSpec) for entity in scene.entities
    }
    known = {
        f"{entity.name}/{joint.name}"
        for entity in scene.entities
        for joint in assets[entity.name].joints
    }
    if set(controlled) - known or set(passive) - known:
        raise DomainValidationError("trajectory references joints absent from the scene")
    selected = np.rint(np.linspace(0, len(commands) - 1, min(frames, len(commands)))).astype(int)
    controlled_columns = {name: index for index, name in enumerate(controlled)}
    states = []
    for sample in selected:
        state = {}
        for entity in scene.entities:
            positions = {}
            for joint in assets[entity.name].joints:
                key = f"{entity.name}/{joint.name}"
                if key in controlled_columns:
                    value = observed[sample, controlled_columns[key]]
                    positions[joint.name] = min(
                        max(float(value), joint.lower_limit), joint.upper_limit
                    )
                elif key in passive:
                    value = passive[key][sample]
                    positions[joint.name] = min(
                        max(float(value), joint.lower_limit), joint.upper_limit
                    )
            root = pose_matrix(entity.pose)
            state[entity.name] = {
                link: root @ transform
                for link, transform in link_transforms(assets[entity.name], positions).items()
            }
        states.append(state)

    sequence = render_blender_scene_frames(store, scene_ref, states)
    images = list(sequence.overview)
    try:
        payload = io.BytesIO()
        images[0].save(
            payload,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=round(1000 / fps),
            loop=0,
            disposal=2,
            comment=json.dumps(
                {
                    "renderer": "blender_eevee_pbr",
                    "scene": scene_ref.model_dump(mode="json"),
                    "task": task_ref.model_dump(mode="json"),
                    "trajectory": trajectory.model_dump(mode="json"),
                    "sample_indices": selected.tolist(),
                    "fps": fps,
                }
            ).encode(),
        )
        return store.put_bytes(payload.getvalue(), media_type="image/gif")
    finally:
        for image in images:
            if isinstance(image, Image.Image):
                image.close()
