from __future__ import annotations

import io
import json

import numpy as np
import pytest
from PIL import Image

from physrealx import PhysRealX
from physrealx.errors import DomainValidationError
from physrealx.reference import reference_asset, reference_scene, reference_task


def _execution(px: PhysRealX):
    asset_ref = px.assets.register(reference_asset())
    scene = reference_scene().model_copy(
        update={
            "entities": (reference_scene().entities[0].model_copy(update={"asset": asset_ref}),)
        }
    )
    scene_ref = px.scenes.register(scene)
    task_ref = px.tasks.register(reference_task(scene).model_copy(update={"scene": scene_ref}))
    trajectory = px.artifacts.put_json(
        {
            "schema": "physrealx.joint-trajectory/1",
            "task": task_ref.model_dump(mode="json"),
            "backend_config": {"controlled_joints": ["drawer/drawer.slide"]},
            "dt_seconds": 0.1,
            "commands": [[0.0], [0.2], [0.4]],
            "observed_positions": [[0.0], [0.18], [0.39]],
            "observed_passive_positions": {},
        }
    )
    return scene_ref, task_ref, trajectory


def test_task_preview_animates_scene_and_tracking(tmp_path) -> None:
    px = PhysRealX.open(tmp_path / "workspace")
    scene, task, trajectory = _execution(px)

    animation = px.scenes.preview_task(scene, task, trajectory, frames=3, fps=10)

    with Image.open(io.BytesIO(px.artifacts.get_bytes(animation))) as gif:
        assert gif.size == (720, 480)
        assert gif.n_frames == 3
        assert gif.info["duration"] == 100
        metadata = json.loads(gif.info["comment"])
        assert metadata["renderer"] == "blender_eevee_pbr"
        assert metadata["trajectory"]["artifact_id"] == trajectory.artifact_id
        assert metadata["sample_indices"] == [0, 1, 2]
        first = np.asarray(gif.convert("RGB"))
        gif.seek(2)
        last = np.asarray(gif.convert("RGB"))
        assert np.any(first != last)


def test_task_preview_rejects_incomplete_execution(tmp_path) -> None:
    px = PhysRealX.open(tmp_path / "workspace")
    scene, task, _ = _execution(px)
    incomplete = px.artifacts.put_json(
        {
            "schema": "physrealx.joint-trajectory/1",
            "task": task.model_dump(mode="json"),
            "backend_config": {"controlled_joints": ["drawer/drawer.slide"]},
            "dt_seconds": 0.1,
            "commands": [[0.0], [0.2]],
            "observed_positions": [[0.0], [0.2]],
        }
    )

    with pytest.raises(DomainValidationError, match="complete executed trajectory"):
        px.scenes.preview_task(scene, task, incomplete)
    with pytest.raises(DomainValidationError, match="frames"):
        px.scenes.preview_task(scene, task, incomplete, frames=True)
