"""Render registered captured appearance from recorded simulator joint states."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Literal

from pydantic import Field

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    ArtifactRef,
    AssetSpec,
    DomainModel,
    EpisodeSummary,
    MethodAccelerator,
    MethodCapability,
    OperationSpec,
    SceneSpec,
)
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import input_artifact, load_resource, publish
from physrealx.integrations.artgs import _SubprocessArtGSRunner
from physrealx.integrations.registration import load_registration
from physrealx.integrations.sensors import CameraSpec
from physrealx.methods import builtin_method
from physrealx.runtime import load_episode


class RegisteredRenderInput(DomainModel):
    schema_id: Literal["physrealx.registered-render-input/1"] = Field(
        default="physrealx.registered-render-input/1", alias="schema"
    )
    registration: ArtifactRef
    episode: EpisodeSummary
    entity: str = Field(min_length=1)
    camera: CameraSpec
    frame_indices: tuple[int, ...] = Field(min_length=2, max_length=128)


class GaussianRenderConfig(DomainModel):
    seed: int = Field(default=2026, ge=0)
    max_seconds: int = Field(default=180, ge=30, le=600)
    max_surface_distance_m: float = Field(default=0.04, gt=0, le=0.1, allow_inf_nan=False)


class RegisteredGaussianRenderer:
    descriptor = builtin_method(
        MethodCapability.RENDER_GENERATE,
        "registered_artgs",
        input_contract="StateTrajectory/1",
        output_contract="RenderStreams/1+CameraMetadata/1",
        accelerator=MethodAccelerator.CUDA,
        python=">=3.10,<3.13",
    ).model_copy(
        update={
            "source_uri": "https://github.com/YuLiu-LY/ArtGS",
            "source_revision": "7c1f41be2cb8b96a.physrealx3",
            "license_spdx": "LicenseRef-ArtGS-Research-Only",
        }
    )

    def probe(self) -> MethodProbe:
        return _SubprocessArtGSRunner().probe()

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import numpy as np
        from scipy.spatial.transform import Rotation

        config = GaussianRenderConfig.model_validate(spec.method.config)
        source = input_artifact(spec, "state_trajectory", artifacts)
        request = RegisteredRenderInput.model_validate_json(artifacts.get_bytes(source))
        registration = load_registration(artifacts, request.registration)
        episode = load_episode(artifacts, request.episode)
        scene = load_resource(artifacts, episode.scene, SceneSpec)
        entity = next((row for row in scene.entities if row.name == request.entity), None)
        if entity is None:
            raise DomainValidationError(
                "registered render entity is absent from the recorded scene"
            )
        asset = load_resource(artifacts, entity.asset, AssetSpec)
        reference = ArtifactRef.model_validate(registration["source_reference"])
        if reference not in asset.evidence:
            raise DomainValidationError(
                "recorded physical asset and appearance use different USD references"
            )
        indices = request.frame_indices
        if (
            tuple(sorted(set(indices))) != indices
            or indices[0] < 0
            or indices[-1] >= len(episode.steps)
        ):
            raise DomainValidationError("render frame indices must be unique, ordered and recorded")
        joint = next(j for j in asset.joints if j.name == registration["observed_joint"])
        sensors = episode.metadata.get("sensor_metadata", {})
        body_order = sensors.get("body_pose_order") if isinstance(sensors, dict) else None
        if not isinstance(body_order, list) or any(
            not isinstance(name, str) for name in body_order
        ):
            raise DomainValidationError(
                "registered rendering requires recorded body-pose identities"
            )
        order = [str(name) for name in body_order]
        static_indices = [
            order.index(f"{entity.name}/{link}") for link in asset.links if link != joint.child
        ]
        static_poses = []
        for index in indices:
            values = []
            for name, width in (("body.position", 3), ("body.orientation", 4)):
                value = episode.steps[index].observations.get(name)
                if value is None or value.inline is None or value.shape != (len(order), width):
                    raise DomainValidationError(
                        "registered rendering requires complete recorded body poses"
                    )
                values.append(np.asarray(value.inline).reshape(-1, width)[static_indices])
            static_poses.append(np.concatenate(values, axis=1))
        if not np.allclose(static_poses, static_poses[0], atol=1e-5):
            raise DomainValidationError(
                "merged captured appearance cannot animate unobserved link motion"
            )
        from physrealx.integrations.mujoco import MujocoConfig

        backend = MujocoConfig.model_validate(episode.backend.config)
        for name in registration["unobserved_joint_motion"]:
            key = f"{entity.name}/{name}"
            if (
                key in backend.controlled_joints
                or abs(backend.passive_positions.get(key, 0)) > 1e-6
            ):
                raise DomainValidationError(
                    "unobserved captured joints must remain in their reference pose"
                )
        signal = f"joint.{entity.name}.{joint.name}.position"
        positions = []
        for index in indices:
            value = episode.steps[index].observations.get(signal)
            if value is None or value.inline is None or len(value.inline) != 1:
                raise DomainValidationError("appearance requires recorded passive-joint positions")
            positions.append(value.inline[0])
            for name in registration["unobserved_joint_motion"]:
                observed = episode.steps[index].observations.get(
                    f"joint.{entity.name}.{name}.position"
                )
                if (
                    observed is not None
                    and observed.inline is not None
                    and abs(observed.inline[0]) > 1e-6
                ):
                    raise DomainValidationError(
                        "captured merged appearance cannot animate an unobserved joint"
                    )
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(entity.pose.rotation_xyzw).as_matrix()
        pose[:3, 3] = entity.pose.translation_m
        root_pose = np.eye(4)
        root_pose[:3, :3] = Rotation.from_quat(asset.root_pose.rotation_xyzw).as_matrix()
        root_pose[:3, 3] = asset.root_pose.translation_m
        pose = pose @ root_pose
        transform = np.eye(4)
        transform[:3, :3], transform[:3, 3] = (
            registration["rotation"],
            registration["translation_m"],
        )
        world_from_capture = pose @ transform
        camera = request.camera
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = Rotation.from_quat(camera.pose.rotation_xyzw).as_matrix() @ np.diag(
            [1, -1, -1]
        )
        camera_pose[:3, 3] = camera.pose.translation_m
        view = np.linalg.inv(camera_pose) @ world_from_capture
        order = registration["state_order"]
        start = registration["evaluations"][order[0]]["joint_position"]
        end = registration["evaluations"][order[1]]["joint_position"]
        visual = ArtifactRef.model_validate(registration["source_visual"])
        runner = _SubprocessArtGSRunner()
        with tempfile.TemporaryDirectory(prefix="physrealx-gaussian-render-") as directory:
            root = Path(directory)
            from importlib import import_module

            from physrealx.integrations._artifacts import read_json
            from physrealx.integrations.collision import load_meshes

            trimesh = import_module("trimesh")
            reference_model = read_json(artifacts, reference, "physrealx.usd-reference/1")
            samples = []
            for link in reference_model["links"]:
                for mesh in load_meshes(artifacts, ArtifactRef.model_validate(link["geometry"])):
                    points = trimesh.sample.sample_surface(mesh, 10000, seed=config.seed)[0]
                    if link["name"] == joint.child:
                        points += np.asarray(joint.axis) * start
                    samples.append(points)
            np.save(root / "reference.npy", np.concatenate(samples), allow_pickle=False)
            with zipfile.ZipFile(io.BytesIO(artifacts.get_bytes(visual))) as archive:
                for name in ("config.json", "point_cloud.ply", "deform.pth"):
                    member = archive.getinfo("model/" + name)
                    if member.file_size > 512 * 1024**2:
                        raise DomainValidationError("Gaussian inference payload exceeds its bound")
                    (root / name).write_bytes(archive.read(member))
            payload = {
                "view_matrix": view.tolist(),
                "camera": camera.model_dump(mode="json"),
                "positions": positions,
                "registration_rotation": registration["rotation"],
                "registration_translation": registration["translation_m"],
                "max_surface_distance_m": config.max_surface_distance_m,
                "start_position": start,
                "capture_axis": (
                    np.asarray(registration["rotation"]).T @ np.asarray(joint.axis)
                ).tolist(),
                "moving_group": int(
                    next(
                        k
                        for k, links in registration["motion_groups"].items()
                        if links == [joint.child]
                    )
                ),
            }
            (root / "request.json").write_text(json.dumps(payload), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).with_name("_gaussian_worker.py")), str(root)],
                cwd=runner.source_root,
                env=runner._environment(config.seed),
                capture_output=True,
                text=True,
                timeout=config.max_seconds,
                check=False,
            )
            if completed.returncode:
                raise MethodExecutionError(
                    "registered Gaussian rendering failed: " + completed.stderr[-4096:]
                )
            streams = {}
            for name in ("rgb", "alpha", "depth"):
                path = root / (name + ".npy")
                values = np.load(path, allow_pickle=False)
                if len(values) != len(indices) or not np.isfinite(values).all():
                    raise MethodExecutionError("Gaussian renderer returned invalid frames")
                streams[name] = artifacts.put_file(path, media_type="application/x-npy").model_dump(
                    mode="json"
                )
            diagnostics = json.loads((root / "diagnostics.json").read_text())
        timestamps = [episode.steps[index].timestamp_seconds for index in indices]
        render = {
            "schema": "physrealx.render-streams/1",
            "status": "COMPLETED",
            "streams": streams,
            "episode": request.episode.model_dump(mode="json"),
            "entity": entity.name,
            "registration": request.registration.model_dump(mode="json"),
            "frame_indices": indices,
            "timestamps_seconds": timestamps,
            "recorded_joint_positions": positions,
            "outside_observed_span": [
                not min(start, end) <= q <= max(start, end) for q in positions
            ],
            "evidence_scope": "captured object appearance driven by recorded simulator states",
            "motion_source": "recorded_joint_state_with_usd_axis",
            "independent_physical_validation": False,
        }
        metadata = {
            "schema": "physrealx.camera-metadata/1",
            "camera": camera.model_dump(mode="json"),
            "world_from_capture": world_from_capture.tolist(),
            "capture_to_optical": view.tolist(),
            "depth_unit": "m",
            "depth_semantics": "alpha_normalized_expected_optical_depth",
            "appearance_scope": "selected entity; other scene entities are not composited",
        }
        return publish(
            spec,
            artifacts,
            {"render_streams": render, "camera_metadata": metadata},
            metrics={key: float(value) for key, value in diagnostics.items()},
        )
