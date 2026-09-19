"""Isaac Sim 4.5 backend with a private Kit process and the standard Episode contract."""

from __future__ import annotations

import json
import math
import os
import signal
import socket
import struct
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import Field

from physrealx.contracts import ArtifactStore, BackendCapabilities, Frame, MethodProbe, StepResult
from physrealx.domain import (
    AssetSpec,
    DataValue,
    JointType,
    JSONValue,
    MethodAccelerator,
    MethodCapability,
    SceneSpec,
    SignalDType,
    TaskSpec,
)
from physrealx.errors import (
    DomainValidationError,
    InvalidStateTransitionError,
    MethodExecutionError,
    MethodUnavailableError,
)
from physrealx.integrations.sensors import CAMERA_MODALITIES, EventAccumulator, project_depth
from physrealx.integrations.simulation_config import PhysicsConfig
from physrealx.methods import builtin_method


class IsaacSimConfig(PhysicsConfig):
    timeout_seconds: float = Field(default=180.0, gt=0, le=1800, allow_inf_nan=False)


def isaac_installation() -> Path:
    """The runtime location is deployment configuration, not an asset or method parameter."""
    root = Path(os.environ.get("PHYSREALX_ISAAC_SIM_ROOT", "")).expanduser()
    if not os.environ.get("PHYSREALX_ISAAC_SIM_ROOT") or not (root / "python.sh").is_file():
        raise MethodUnavailableError(
            "set PHYSREALX_ISAAC_SIM_ROOT to an Isaac Sim 4.5 installation"
        )
    version = (root / "VERSION").read_text().strip()
    if not (version == "4.5.0" or version.startswith("4.5.0-")):
        raise MethodUnavailableError(f"Isaac Sim adapter requires 4.5.0, found {version}")
    return root.resolve()


def prepare_scene(
    directory: Path,
    scene: SceneSpec,
    assets: Mapping[str, AssetSpec],
    store: ArtifactStore,
) -> dict[str, Any]:
    """Use the packaging author, preserving meshes, materials, inertias and joint frames."""
    from physrealx.integrations.usd import author_asset_stage

    entities = []
    for index, entity in enumerate(scene.entities):
        asset = assets[entity.name]
        if asset.up_axis != "z":
            raise DomainValidationError("Isaac Sim scene assets must be in the Z-up metric frame")
        folder = directory / f"entity_{index}"
        folder.mkdir()
        stage, textures, _ = author_asset_stage(asset, store)
        for name, payload in textures.items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        path = folder / "asset.usdc"
        if not stage.GetRootLayer().Export(str(path)):
            raise MethodExecutionError("failed to author Isaac Sim USD asset")
        entities.append(
            {
                "name": entity.name,
                "path": str(path),
                "pose": entity.pose.model_dump(mode="json"),
                "asset": asset.model_dump(mode="json"),
            }
        )
    return {"entities": entities}


class IsaacSimSession:
    def __init__(
        self,
        scene: SceneSpec,
        assets: Mapping[str, AssetSpec],
        task: TaskSpec,
        config: IsaacSimConfig,
        store: ArtifactStore,
    ) -> None:
        self._task, self._config, self._store = task, config, store
        self._events = {
            name: EventAccumulator(c.event_threshold) for name, c in config.cameras.items()
        }
        body_names = {f"{e.name}/{link}" for e in scene.entities for link in assets[e.name].links}
        if set(config.external_wrenches) - body_names:
            raise DomainValidationError("external wrench names must identify scene links")
        self._closed, self._initialized, self._steps = False, False, 0
        self._process: subprocess.Popen[bytes] | None = None
        self._socket: socket.socket | None = None
        self._log: Any = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._joints = {
            f"{entity.name}/{joint.name}": joint
            for entity in scene.entities
            for joint in assets[entity.name].joints
            if joint.joint_type is not JointType.FIXED
        }
        self._body_count = sum(len(assets[entity.name].links) for entity in scene.entities)
        self._validate(task, config)
        root = isaac_installation()
        try:
            self._directory = tempfile.TemporaryDirectory(prefix="physrealx-isaac-")
            directory = Path(self._directory.name)
            request = prepare_scene(directory, scene, assets, store)
            request.update(
                config=config.model_dump(mode="json"),
                task=task.model_dump(mode="json"),
                output_directory=str(directory),
            )
            (directory / "request.json").write_text(json.dumps(request, allow_nan=False))
            self._log = (directory / "worker.log").open("wb")
            parent, child = socket.socketpair()
            self._socket = parent
            parent.settimeout(config.timeout_seconds)
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTHONNOUSERSITE"] = "1"
            # Kit owns its Python and native libraries; never inherit a Conda interpreter override.
            for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONEXE", "CONDA_PREFIX", "LD_LIBRARY_PATH"):
                env.pop(key, None)
            try:
                self._process = subprocess.Popen(
                    [
                        str(root / "python.sh"),
                        str(Path(__file__).with_name("_isaac_worker.py")),
                        str(directory / "request.json"),
                        str(child.fileno()),
                    ],
                    pass_fds=(child.fileno(),),
                    stdin=subprocess.DEVNULL,
                    stdout=self._log,
                    stderr=subprocess.STDOUT,
                    cwd=root,
                    env=env,
                    start_new_session=True,
                )
            finally:
                child.close()
            self._receive()
        except BaseException:
            self.close()
            raise

    def _validate(self, task: TaskSpec, config: IsaacSimConfig) -> None:
        if task.action.kind not in IsaacSimBackend.action_kinds:
            raise DomainValidationError("unsupported Isaac Sim action kind")
        for name, value in dict(
            zip(config.controlled_joints, config.initial_positions), **config.passive_positions
        ).items():
            if name not in self._joints:
                raise DomainValidationError(f"unknown movable joint {name!r}")
            joint = self._joints[name]
            if not joint.lower_limit <= value <= joint.upper_limit:
                raise DomainValidationError(f"initial position exceeds limits for {name}")
        kinds = {self._joints[name].joint_type for name in config.controlled_joints}
        if len(kinds) != 1:
            raise DomainValidationError("one action vector requires homogeneous joint units")
        self._unit = "m" if JointType.PRISMATIC in kinds else "rad"
        unit = (
            ("N" if self._unit == "m" else "N*m")
            if task.action.kind == "joint_effort"
            else self._unit
        )
        if (
            task.action.signal.shape != (len(config.controlled_joints),)
            or task.action.signal.unit != unit
            or task.action.signal.frame != "joint"
            or task.action.signal.dtype not in {SignalDType.FLOAT32, SignalDType.FLOAT64}
        ):
            raise DomainValidationError("action shape or units disagree with controlled joints")
        for name, joint in self._joints.items():
            if name not in (*config.controlled_joints, *config.passive_positions) and not (
                joint.lower_limit <= 0 <= joint.upper_limit
            ):
                raise DomainValidationError(f"passive joint {name} needs an initial position")
        expected = {
            "body.position": ((self._body_count, 3), "m", "world"),
            "body.orientation": ((self._body_count, 4), "1", "world"),
            "contact.force": ((1,), "N", "joint"),
            **{
                f"body.{name}": ((self._body_count, 3), unit, "world")
                for name, unit in (
                    ("external_force", "N"),
                    ("external_torque", "N*m"),
                    ("wrench_origin", "m"),
                )
            },
        }
        for prefix, count, position_unit in (
            ("joint", len(config.controlled_joints), self._unit),
            *(
                (
                    "joint." + name.replace("/", "."),
                    1,
                    "m" if joint.joint_type is JointType.PRISMATIC else "rad",
                )
                for name, joint in self._joints.items()
            ),
        ):
            for suffix, signal_unit in (
                ("position", position_unit),
                ("velocity", position_unit + "/s"),
                ("effort", "N" if position_unit == "m" else "N*m"),
            ):
                expected[prefix + "." + suffix] = ((count,), signal_unit, "joint")
        camera_signals = {
            s.name: s
            for name, camera in config.cameras.items()
            for s in camera.signals(name, frequency_hz=1 / self.dt_seconds)
        }
        for observation in task.observations:
            if observation.name in camera_signals:
                sensor = camera_signals[observation.name]
                if (observation.dtype, observation.shape, observation.unit, observation.frame) != (
                    sensor.dtype,
                    sensor.shape,
                    sensor.unit,
                    sensor.frame,
                ):
                    raise DomainValidationError(f"invalid camera observation: {observation.name}")
                continue
            if observation.dtype not in {SignalDType.FLOAT32, SignalDType.FLOAT64} or expected.get(
                observation.name
            ) != (observation.shape, observation.unit, observation.frame):
                raise DomainValidationError(
                    f"unsupported Isaac Sim observation contract: {observation.name}"
                )

    @property
    def dt_seconds(self) -> float:
        return self._config.timestep_seconds * self._config.substeps

    def _ensure_open(self) -> None:
        if self._closed:
            raise InvalidStateTransitionError("Isaac Sim session is closed")

    def _receive(self) -> dict[str, Any]:
        assert self._socket is not None
        connection = self._socket

        def read(count: int) -> bytes:
            chunks = bytearray()
            while len(chunks) < count:
                chunk = connection.recv(count - len(chunks))
                if not chunk:
                    raise EOFError("Isaac Sim worker exited")
                chunks.extend(chunk)
            return bytes(chunks)

        try:
            size = struct.unpack("!I", read(4))[0]
            if size > 32 * 1024 * 1024:
                raise ValueError("Isaac Sim response exceeds 32 MiB")
            result = json.loads(read(size))
            if not isinstance(result, dict) or "error" in result:
                raise ValueError(str(result))
            return result
        except (OSError, ValueError, EOFError) as error:
            log = ""
            if self._directory is not None:
                with (Path(self._directory.name) / "worker.log").open("rb") as stream:
                    stream.seek(max(0, os.fstat(stream.fileno()).st_size - 8000))
                    log = stream.read().decode(errors="replace")
            self.close()
            raise MethodExecutionError(
                f"Isaac Sim worker failed: {error}", details={"worker_log_tail": log}
            ) from error

    def _request(self, command: dict[str, Any]) -> dict[str, Any]:
        self._ensure_open()
        assert self._socket is not None
        payload = json.dumps(command, allow_nan=False).encode()
        try:
            self._socket.sendall(struct.pack("!I", len(payload)) + payload)
        except OSError as error:
            self.close()
            raise MethodExecutionError("Isaac Sim worker connection failed") from error
        return self._receive()

    def _frame(self, response: dict[str, Any]) -> Frame:
        import io

        import numpy as np

        values = {}
        cameras = {}
        for name in response.get("cameras", ()):
            if name not in self._config.cameras or self._directory is None:
                raise MethodExecutionError("worker returned an unknown camera")
            with np.load(
                Path(self._directory.name) / f"camera_{name}.npz", allow_pickle=False
            ) as archive:
                arrays = {key: archive[key] for key in archive.files}
            requested = {
                s.name.rsplit(".", 1)[-1]
                for s in self._task.observations
                if s.name.startswith(f"camera.{name}.")
            }
            if "pointcloud" in requested:
                arrays["pointcloud"] = project_depth(arrays["depth"], arrays["intrinsics"])
            if "events" in requested:
                arrays["events"] = self._events[name].update(arrays["rgb"])
            cameras[name] = arrays
        for spec in self._task.observations:
            if spec.name.startswith("camera."):
                _, name, modality = spec.name.split(".")
                array = cameras[name][modality]
                if (
                    array.shape != spec.shape
                    or str(array.dtype) != spec.dtype.value
                    or not np.isfinite(array).all()
                ):
                    raise MethodExecutionError(f"invalid native camera data: {spec.name}")
                stream = io.BytesIO()
                np.save(stream, array, allow_pickle=False)
                values[spec.name] = DataValue(
                    dtype=spec.dtype,
                    shape=spec.shape,
                    unit=spec.unit,
                    frame=spec.frame,
                    timestamp_seconds=self._steps * self.dt_seconds,
                    artifact=self._store.put_bytes(
                        stream.getvalue(), media_type="application/x-npy"
                    ),
                )
                continue
            item = response["signals"][spec.name]
            value = DataValue(
                dtype=spec.dtype,
                shape=tuple(item["shape"]),
                unit=item["unit"],
                frame=item["frame"],
                inline=tuple(item["values"]),
                timestamp_seconds=self._steps * self.dt_seconds,
            )
            value.assert_matches(spec)
            values[spec.name] = value
        self._positions = response["signals"]["joint.position"]["values"]
        return Frame(observations=values, state=dict(values), metadata=response["metadata"])

    def reset(self, seed: int) -> Frame:
        self._ensure_open()
        if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**63:
            raise DomainValidationError("seed must be an integer in [0, 2**63)")
        for accumulator in self._events.values():
            accumulator.reset()
        response = self._request({"command": "reset", "seed": seed})
        self._steps = 0
        result = self._frame(response)
        self._initialized = True
        return result

    def step(self, action: DataValue) -> StepResult:
        self._ensure_open()
        if not self._initialized:
            raise InvalidStateTransitionError("Isaac Sim session must be reset before stepping")
        action.assert_matches(self._task.action.signal)
        if action.inline is None or any(
            not math.isfinite(float(v)) or not lo <= float(v) <= hi
            for v, lo, hi in zip(
                action.inline or (), self._task.action.lower, self._task.action.upper
            )
        ):
            raise DomainValidationError("joint command exceeds the task action bounds")
        if self._task.action.kind != "joint_effort":
            for index, (name, value) in enumerate(
                zip(self._config.controlled_joints, action.inline)
            ):
                target = float(value)
                if self._task.action.kind == "joint_delta":
                    target += float(self._positions[index])
                joint = self._joints[name]
                if not joint.lower_limit <= target <= joint.upper_limit:
                    raise DomainValidationError(f"commanded joint target exceeds limits: {name}")
        response = self._request({"command": "step", "action": list(action.inline)})
        self._steps += 1
        frame = self._frame(response)
        return StepResult(
            frame=frame,
            reward=0.0,
            terminated=False,
            truncated=self._steps >= self._task.max_steps,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._socket is not None:
            self._socket.close()
        if self._process is not None:
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(self._process.pid, signal.SIGTERM)
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self._process.pid, signal.SIGKILL)
                    self._process.wait(timeout=5)
        if self._log is not None:
            self._log.close()
        if self._directory is not None:
            self._directory.cleanup()


class IsaacSimBackend:
    descriptor = builtin_method(
        MethodCapability.SIMULATION_RUN,
        "isaac_sim",
        input_contract="SceneSpec/1+TaskSpec/1+AssetSpec/1",
        output_contract="SimulationSession/1",
        python=">=3.10,<3.13",
        accelerator=MethodAccelerator.CUDA,
    )
    action_kinds = ("joint_delta", "joint_position", "joint_effort")

    def __init__(self, artifacts: ArtifactStore) -> None:
        self.artifacts = artifacts

    def probe(self) -> MethodProbe:
        try:
            root = isaac_installation()
            from physrealx.integrations._artifacts import dependency_probe

            probe = dependency_probe(
                {
                    "pxr.Usd": ("usd-core", "24.11"),
                    "scipy": ("scipy", "1.13.1"),
                    "trimesh": ("trimesh", "4.5.1"),
                }
            )
            if not probe.ready:
                return probe
        except (MethodUnavailableError, OSError) as error:
            return MethodProbe(False, str(error))
        return MethodProbe(
            True, f"Isaac Sim 4.5 at {root}; device readiness checked at session startup"
        )

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            contact_force=True,
            deterministic_reset=True,
            observation_keys=(
                "joint.position",
                "joint.velocity",
                "joint.effort",
                "contact.force",
                "body.position",
                "body.orientation",
                "body.external_force",
                "body.external_torque",
                "body.wrench_origin",
            ),
            observation_patterns=(
                "joint.*.position",
                "joint.*.velocity",
                "joint.*.effort",
                *(f"camera.*.{m}" for m in CAMERA_MODALITIES),
            ),
            action_kinds=self.action_kinds,
        )

    def create_session(
        self,
        scene: SceneSpec,
        assets: Mapping[str, AssetSpec],
        task: TaskSpec,
        config: Mapping[str, JSONValue],
    ) -> IsaacSimSession:
        return IsaacSimSession(
            scene, assets, task, IsaacSimConfig.model_validate(config), self.artifacts
        )
