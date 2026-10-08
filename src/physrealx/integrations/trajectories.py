"""Smooth joint-space trajectories validated by executing the canonical physical scene."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    AssetSpec,
    DataValue,
    DomainModel,
    MethodCapability,
    OperationSpec,
    SceneSpec,
    TaskSpec,
    VersionedRef,
)
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import input_model, load_resource, publish
from physrealx.integrations.mujoco import MujocoBackend, MujocoConfig, MujocoSession
from physrealx.methods import builtin_method


class TaskInstance(DomainModel):
    schema_id: Literal["physrealx.task-instance/1"] = Field(
        default="physrealx.task-instance/1", alias="schema"
    )
    task: VersionedRef
    backend_config: MujocoConfig
    waypoints: tuple[tuple[float, ...], ...] = ()
    target_positions: tuple[float, ...] = Field(min_length=1)


class TrajectoryConfig(DomainModel):
    duration_seconds: float = Field(default=1.5, gt=0, allow_inf_nan=False)
    settle_seconds: float = Field(default=0.5, ge=0, allow_inf_nan=False)
    max_tracking_error: float = Field(default=0.05, gt=0, allow_inf_nan=False)
    max_contact_force_n: float = Field(default=100.0, gt=0, allow_inf_nan=False)
    seed: int = Field(default=2026, ge=0)
    require_success: bool = True


class JointTrajectoryAdapter:
    descriptor = builtin_method(
        MethodCapability.TRAJECTORY_GENERATE,
        "mujoco_joint_trajectory",
        input_contract="TaskInstance/1",
        output_contract="Trajectory/1+PlanningReport/1",
        python=">=3.10,<3.13",
    )

    def probe(self) -> MethodProbe:
        from physrealx.integrations._artifacts import dependency_probe

        return dependency_probe(
            {
                "mujoco": ("mujoco", "3.3.7"),
                "numpy": ("numpy", "1.26.4"),
                "trimesh": ("trimesh", "4.5.1"),
            }
        )

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import numpy as np

        config = TrajectoryConfig.model_validate(spec.method.config)
        instance = input_model(spec, "instance", artifacts, TaskInstance)
        task = load_resource(artifacts, instance.task, TaskSpec)
        scene = load_resource(artifacts, task.scene, SceneSpec)
        assets = {
            item.name: load_resource(artifacts, item.asset, AssetSpec) for item in scene.entities
        }
        target = np.asarray(instance.target_positions)
        if not {"joint.position", "contact.force"} <= {s.name for s in task.observations}:
            raise DomainValidationError(
                "trajectory verification requires position and contact force"
            )
        if (
            task.action.kind != "joint_position"
            or target.shape != task.action.signal.shape
            or not np.isfinite(target).all()
            or np.any(target < task.action.lower)
            or np.any(target > task.action.upper)
        ):
            raise DomainValidationError("trajectory target must respect position action bounds")
        dt = instance.backend_config.timestep_seconds * instance.backend_config.substeps
        count = int(np.ceil((config.duration_seconds + config.settle_seconds) / dt))
        if count < 2 or count > task.max_steps:
            raise DomainValidationError("trajectory duration exceeds the task execution horizon")
        times = (np.arange(count) + 1) * dt
        initial = np.asarray(instance.backend_config.initial_positions)
        waypoints = np.asarray(instance.waypoints, dtype=float)
        if waypoints.size == 0:
            waypoints = np.empty((0, len(target)))
        if (
            waypoints.ndim != 2
            or waypoints.shape[1:] != target.shape
            or not np.isfinite(waypoints).all()
            or np.any(waypoints < np.asarray(task.action.lower))
            or np.any(waypoints > np.asarray(task.action.upper))
        ):
            raise DomainValidationError("trajectory waypoints must respect action bounds")
        path = np.vstack((initial, waypoints, target))
        segment_count = len(path) - 1
        phase = np.clip(times / config.duration_seconds, 0, 1)
        time_scaling = 10 * phase**3 - 15 * phase**4 + 6 * phase**5
        progress = time_scaling * segment_count
        segments = np.minimum(progress.astype(int), segment_count - 1)
        local_phase = progress - segments
        commands = path[segments] + local_phase[:, None] * (path[segments + 1] - path[segments])
        positions, contacts = [], []
        passive_positions: dict[str, list[float]] = {
            name: [] for name in instance.backend_config.passive_positions
        }
        session = MujocoSession(scene, assets, task, instance.backend_config, artifacts)
        try:
            initial_frame = session.reset(config.seed)
            initial_scalars = {
                name: float(value.inline[0])
                for name, value in initial_frame.observations.items()
                if value.inline is not None and len(value.inline) == 1
            }
            if not all(p.evaluate(initial_scalars) for p in task.initial_conditions):
                raise DomainValidationError("trajectory reset violates task initial conditions")
            for command in commands:
                frame = session.step(
                    DataValue(
                        dtype=task.action.signal.dtype,
                        shape=task.action.signal.shape,
                        unit=task.action.signal.unit,
                        frame=task.action.signal.frame,
                        inline=tuple(command),
                    )
                ).frame
                positions.append(frame.observations["joint.position"].inline)
                for name, values in passive_positions.items():
                    entity, joint = name.split("/", 1)
                    observation = frame.observations[f"joint.{entity}.{joint}.position"]
                    assert observation.inline is not None
                    values.append(float(observation.inline[0]))
                contact = frame.observations["contact.force"]
                assert contact.inline is not None
                contacts.append(float(contact.inline[0]))
        finally:
            session.close()
        error = float(np.max(np.abs(np.asarray(positions) - commands)))
        peak = max(contacts)
        scalars = {
            name: float(value.inline[0])
            for name, value in frame.observations.items()
            if value.inline is not None and len(value.inline) == 1
        }
        success = all(predicate.evaluate(scalars) for predicate in task.success_conditions)
        accepted = (
            error <= config.max_tracking_error and peak <= config.max_contact_force_n and success
        )
        if config.require_success and not accepted:
            raise MethodExecutionError(
                f"trajectory execution failed: tracking_error={error:.6g}, "
                f"contact_N={peak:.6g}, task_success={success}"
            )
        metrics = {"max_tracking_error": error, "peak_contact_force_n": peak, "steps": float(count)}
        return publish(
            spec,
            artifacts,
            {
                "trajectory": {
                    "schema": "physrealx.joint-trajectory/1",
                    "task": instance.task.model_dump(mode="json"),
                    "backend": MujocoBackend.descriptor.model_dump(mode="json"),
                    "backend_config": instance.backend_config.model_dump(mode="json"),
                    "dt_seconds": dt,
                    "commands": commands.tolist(),
                    "observed_positions": positions,
                    "observed_passive_positions": passive_positions,
                    "waypoints": waypoints.tolist(),
                    "target_positions": target.tolist(),
                    "seed": config.seed,
                },
                "planning_report": {
                    "schema": "physrealx.planning-report/1",
                    "status": "PASSED" if accepted else "FAILED",
                    "validation_scope": "mujoco_execution",
                    "independent_physical_validation": False,
                    "task_success": success,
                    "gates": {
                        "tracking_error": error <= config.max_tracking_error,
                        "contact_force": peak <= config.max_contact_force_n,
                        "task_goal": success,
                    },
                    "metrics": metrics,
                },
            },
            metrics=metrics,
        )
