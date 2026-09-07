"""Bounded scene placement and grounded articulated-task templates."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import Field, model_validator

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    ActionSpec,
    AssetSpec,
    DomainModel,
    JointType,
    MethodCapability,
    OperationSpec,
    PoseSpec,
    Predicate,
    PredicateOperator,
    ResourceType,
    SceneEntityRole,
    SceneEntitySpec,
    SceneSpec,
    SignalDType,
    SignalSpec,
    TaskSpec,
    VersionedRef,
)
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations._artifacts import (
    input_artifact,
    input_model,
    load_resource,
    publish,
    read_json,
)
from physrealx.integrations.collision import load_meshes, mesh_reference
from physrealx.integrations.usd import _link_poses
from physrealx.methods import builtin_method


class PlacementObject(DomainModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    asset: VersionedRef
    role: SceneEntityRole = SceneEntityRole.OBJECT


class SceneGenerationRequest(DomainModel):
    resource_id: str
    version: str
    display_name: str
    domain: Literal["home", "tabletop", "shelf", "industrial"]
    objects: tuple[PlacementObject, ...] = Field(min_length=1, max_length=32)
    xy_bounds_m: tuple[float, float, float, float]
    support_height_m: float = Field(default=0.0, allow_inf_nan=False)
    clearance_m: float = Field(default=0.05, ge=0, allow_inf_nan=False)
    yaw_range_rad: tuple[float, float] = (0.0, 0.0)
    seed: int = Field(ge=0)
    max_attempts: int = Field(default=200, ge=1, le=10000)

    @model_validator(mode="after")
    def validate_request(self) -> SceneGenerationRequest:
        x0, y0, x1, y1 = self.xy_bounds_m
        if (
            not all(math.isfinite(v) for v in (*self.xy_bounds_m, *self.yaw_range_rad))
            or x0 >= x1
            or y0 >= y1
            or self.yaw_range_rad[0] > self.yaw_range_rad[1]
        ):
            raise ValueError("placement bounds must be finite ordered intervals")
        if len({item.name for item in self.objects}) != len(self.objects):
            raise ValueError("placement names must be unique")
        if any(item.asset.resource_type is not ResourceType.ASSET for item in self.objects):
            raise ValueError("scene placement requires asset references")
        return self


def asset_vertices(asset: AssetSpec, store: ArtifactStore, *, link: str | None = None) -> Any:
    import numpy as np
    from scipy.spatial.transform import Rotation

    poses = _link_poses(asset)
    vertices = []
    for geometry in asset.geometries:
        if geometry.role.value != "collision" or (link is not None and geometry.link != link):
            continue
        if geometry.primitive_dimensions_m is not None:
            import itertools

            points = (
                np.array(list(itertools.product([-0.5, 0.5], repeat=3)))
                * geometry.primitive_dimensions_m
            )
        else:
            points = np.concatenate(
                [mesh.vertices for mesh in load_meshes(store, mesh_reference(geometry.uri, store))]
            )
        points = points * geometry.scale
        points = (
            points @ Rotation.from_quat(geometry.local_pose.rotation_xyzw).as_matrix().T
            + geometry.local_pose.translation_m
        )
        vertices.append(points @ poses[geometry.link][:3, :3].T + poses[geometry.link][:3, 3])
    if not vertices:
        raise DomainValidationError("scene placement requires collision geometry")
    return np.concatenate(vertices)


def scene_bounds(scene: SceneSpec, store: ArtifactStore) -> dict[str, Any]:
    import numpy as np
    from scipy.spatial.transform import Rotation

    result = {}
    for entity in scene.entities:
        asset = load_resource(store, entity.asset, AssetSpec)
        points = asset_vertices(asset, store)
        points = (
            points @ Rotation.from_quat(entity.pose.rotation_xyzw).as_matrix().T
            + entity.pose.translation_m
        )
        result[entity.name] = np.array([points.min(axis=0), points.max(axis=0)])
    return result


class SceneGenerationAdapter:
    descriptor = builtin_method(
        MethodCapability.SCENE_GENERATE,
        "bounded_layout",
        input_contract="SceneGenerationRequest/1",
        output_contract="SceneDraft/1",
        python=">=3.10,<3.13",
    )

    def probe(self) -> MethodProbe:
        from physrealx.integrations._artifacts import dependency_probe

        return dependency_probe(
            {
                "numpy": ("numpy", "1.26.4"),
                "scipy": ("scipy", "1.13.1"),
                "trimesh": ("trimesh", "4.5.1"),
            }
        )

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import numpy as np
        from scipy.spatial.transform import Rotation

        if spec.method.config:
            raise DomainValidationError("layout settings belong in the scene generation request")
        request = SceneGenerationRequest.model_validate(spec.arguments)
        rng = np.random.default_rng(request.seed)
        entities: list[SceneEntitySpec] = []
        occupied: list[Any] = []
        attempts = 0
        for item in request.objects:
            asset = load_resource(artifacts, item.asset, AssetSpec)
            points = asset_vertices(asset, artifacts)
            for _ in range(request.max_attempts):
                attempts += 1
                rotation = Rotation.from_euler("z", rng.uniform(*request.yaw_range_rad))
                rotated = points @ rotation.as_matrix().T
                lo, hi = rotated.min(axis=0), rotated.max(axis=0)
                minimum = np.array(request.xy_bounds_m[:2]) - lo[:2]
                maximum = np.array(request.xy_bounds_m[2:]) - hi[:2]
                if np.any(minimum > maximum):
                    continue
                position = np.r_[rng.uniform(minimum, maximum), request.support_height_m - lo[2]]
                bounds = np.array([lo + position, hi + position])
                if any(
                    np.all(bounds[0][:2] < other[1][:2] + request.clearance_m)
                    and np.all(other[0][:2] < bounds[1][:2] + request.clearance_m)
                    for other in occupied
                ):
                    continue
                entities.append(
                    SceneEntitySpec(
                        name=item.name,
                        asset=item.asset,
                        role=item.role,
                        pose=PoseSpec(translation_m=position, rotation_xyzw=rotation.as_quat()),
                    )
                )
                occupied.append(bounds)
                break
            else:
                raise MethodExecutionError(
                    f"no feasible collision layout for {item.name} "
                    f"within {request.max_attempts} attempts"
                )
        scene = SceneSpec(
            resource_id=request.resource_id,
            version=request.version,
            display_name=request.display_name,
            entities=tuple(entities),
            metadata={
                "domain": request.domain,
                "seed": request.seed,
                "placement_scope": "conservative_rest_state_xy_bounds",
                "generation_request": request.model_dump(mode="json"),
            },
        )
        return publish(
            spec,
            artifacts,
            {
                "scene_draft": {
                    "schema": "physrealx.scene-draft/1",
                    "scene_spec": scene.model_dump(mode="json"),
                }
            },
            metrics={"entities": float(len(entities)), "placement_attempts": float(attempts)},
        )


class SceneCompositionAdapter:
    descriptor = builtin_method(
        MethodCapability.SCENE_COMPOSE,
        "collision_constraints",
        input_contract="SceneDraft/1",
        output_contract="SceneSpec/1+ConstraintReport/1",
        python=">=3.10,<3.13",
    )

    def probe(self) -> MethodProbe:
        return SceneGenerationAdapter().probe()

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import numpy as np

        if spec.method.config:
            raise DomainValidationError("composition does not accept implicit layout changes")
        document = read_json(
            artifacts, input_artifact(spec, "scene", artifacts), "physrealx.scene-draft/1"
        )
        scene = SceneSpec.model_validate(document["scene_spec"])
        bounds = scene_bounds(scene, artifacts)
        names = list(bounds)
        collisions = [
            (a, b)
            for i, a in enumerate(names)
            for b in names[i + 1 :]
            if np.all(bounds[a][0] < bounds[b][1] - 1e-6)
            and np.all(bounds[b][0] < bounds[a][1] - 1e-6)
        ]
        if collisions:
            raise DomainValidationError(
                f"scene contains overlapping collision bounds: {collisions}"
            )
        return publish(
            spec,
            artifacts,
            {
                "scene_spec": scene,
                "constraint_report": {
                    "schema": "physrealx.constraint-report/1",
                    "status": "PASSED",
                    "scope": "rest_state_collision_bounds",
                    "bounds_m": {name: value.tolist() for name, value in bounds.items()},
                    "motion_requires_simulation_validation": True,
                },
            },
        )


class JointActuation(DomainModel):
    """Robot control is distinct from the object joint defining task success."""

    entity: str
    joint: str
    initial_position: float = Field(allow_inf_nan=False)
    target_position: float = Field(allow_inf_nan=False)


class ArticulatedTaskConfig(DomainModel):
    resource_id: str
    version: str = "1.0.0"
    entity: str
    joint: str
    target_fraction: float = Field(default=0.85, gt=0, lt=1)
    tolerance: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    max_steps: int = Field(default=80, ge=2, le=10000)
    dt_seconds: float = Field(default=0.05, gt=0, allow_inf_nan=False)
    initial_fraction: float | None = Field(default=None, ge=0, le=1)
    actuation: JointActuation | None = None


class TaskTemplateAdapter:
    descriptor = builtin_method(
        MethodCapability.TASK_COMPILE,
        "articulated_templates",
        input_contract="SceneSpec/1+Instruction/1",
        output_contract="TaskSpec/1",
        version="2.0.0",
    )

    def probe(self) -> MethodProbe:
        return MethodProbe(True, "Grounded open/close joint templates available")

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        config = ArticulatedTaskConfig.model_validate(spec.method.config)
        scene = input_model(spec, "scene", artifacts, SceneSpec)
        instruction = str(spec.arguments.get("instruction", "")).strip()
        lower = instruction.lower()
        opened = any(token in lower for token in ("打开", "拉开", "推开", "open "))
        closed = any(token in lower for token in ("关闭", "关上", "合上", "close "))
        if opened == closed or any(token in lower for token in ("不要", "禁止", "don't", "do not")):
            raise DomainValidationError(
                "instruction must unambiguously open or close the selected joint"
            )
        entity = next((item for item in scene.entities if item.name == config.entity), None)
        if entity is None:
            raise DomainValidationError("task entity is absent from the scene")
        asset = load_resource(artifacts, entity.asset, AssetSpec)
        joint = next((item for item in asset.joints if item.name == config.joint), None)
        if (
            joint is None
            or joint.joint_type is JointType.FIXED
            or joint.lower_limit == joint.upper_limit
        ):
            raise DomainValidationError("task requires a movable joint with a nonzero range")
        target = joint.lower_limit + (
            config.target_fraction if opened else 1 - config.target_fraction
        ) * (joint.upper_limit - joint.lower_limit)
        initial = joint.lower_limit if opened else joint.upper_limit
        if config.initial_fraction is not None:
            initial = joint.lower_limit + config.initial_fraction * (
                joint.upper_limit - joint.lower_limit
            )
        controlled, control_entity = joint, entity
        control_initial, control_target = initial, target
        goal_signal = "joint.position"
        passive_positions = {}
        if config.actuation is not None:
            actuation = config.actuation
            robot_entity = next(
                (item for item in scene.entities if item.name == actuation.entity), None
            )
            if robot_entity is None or robot_entity.role is not SceneEntityRole.ROBOT:
                raise DomainValidationError("external actuation requires a declared robot entity")
            control_entity = robot_entity
            robot = load_resource(artifacts, control_entity.asset, AssetSpec)
            robot_joint = next(
                (item for item in robot.joints if item.name == actuation.joint), None
            )
            if robot_joint is None or robot_joint.joint_type is JointType.FIXED:
                raise DomainValidationError("actuation requires a movable robot joint")
            controlled = robot_joint
            control_initial, control_target = actuation.initial_position, actuation.target_position
            goal_signal = f"joint.{config.entity}.{config.joint}.position"
            passive_positions[f"{config.entity}/{config.joint}"] = initial
        if controlled.effort_limit is None or any(
            not controlled.lower_limit <= value <= controlled.upper_limit
            for value in (control_initial, control_target)
        ):
            raise DomainValidationError("control requires an effort limit and bounded positions")
        unit = "m" if controlled.joint_type is JointType.PRISMATIC else "rad"
        frequency = 1 / config.dt_seconds
        observations = tuple(
            SignalSpec(
                name=name,
                shape=(1,),
                dtype=SignalDType.FLOAT32,
                unit=u,
                frame="joint",
                frequency_hz=frequency,
            )
            for name, u in (
                ("joint.position", unit),
                ("joint.velocity", unit + "/s"),
                ("joint.effort", "N" if unit == "m" else "N*m"),
                ("contact.force", "N"),
            )
        )
        if config.actuation is not None:
            goal_unit = "m" if joint.joint_type is JointType.PRISMATIC else "rad"
            observations += tuple(
                SignalSpec(
                    name=name,
                    shape=(1,),
                    dtype=SignalDType.FLOAT32,
                    unit=u,
                    frame="joint",
                    frequency_hz=frequency,
                )
                for name, u in (
                    (goal_signal, goal_unit),
                    (goal_signal.removesuffix("position") + "velocity", goal_unit + "/s"),
                )
            )
        scene_input = next(item for item in spec.inputs if item.name == "scene")
        assert scene_input.resource is not None
        task = TaskSpec(
            resource_id=config.resource_id,
            version=config.version,
            scene=scene_input.resource,
            subject_entity=config.entity,
            instruction=instruction,
            initial_conditions=(
                Predicate(
                    signal=goal_signal,
                    operator=PredicateOperator.EQ,
                    value=initial,
                    tolerance=config.tolerance,
                ),
            ),
            success_conditions=(
                Predicate(
                    signal=goal_signal,
                    operator=PredicateOperator.EQ,
                    value=target,
                    tolerance=config.tolerance,
                ),
            ),
            action=ActionSpec(
                kind="joint_position",
                signal=SignalSpec(
                    name="joint.command",
                    shape=(1,),
                    dtype=SignalDType.FLOAT32,
                    unit=unit,
                    frame="joint",
                    frequency_hz=frequency,
                ),
                lower=(controlled.lower_limit,),
                upper=(controlled.upper_limit,),
            ),
            observations=observations,
            max_steps=config.max_steps,
            metadata={
                "task_template": "open_joint" if opened else "close_joint",
                "controlled_joints": [f"{control_entity.name}/{controlled.name}"],
                "initial_positions": [control_initial],
                "target_positions": [control_target],
                "passive_positions": passive_positions,
                "dt_seconds": config.dt_seconds,
                "effort_limits": [controlled.effort_limit],
                "evidence_scope": "robot_contact"
                if config.actuation
                else "articulated_object_control",
            },
        )
        return publish(spec, artifacts, {"task_spec": task})


class TaskValidationAdapter:
    descriptor = builtin_method(
        MethodCapability.TASK_VALIDATE,
        "joint_constraints",
        input_contract="TaskSpec/1",
        output_contract="TaskValidationReport/1",
    )

    def probe(self) -> MethodProbe:
        return MethodProbe(True, "Task schema, scene identity and joint constraints available")

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        if spec.method.config:
            raise DomainValidationError("task constraints are defined by the task and asset")
        task = input_model(spec, "task", artifacts, TaskSpec)
        scene = load_resource(artifacts, task.scene, SceneSpec)
        entity = next((item for item in scene.entities if item.name == task.subject_entity), None)
        if entity is None:
            raise DomainValidationError("task subject is not in its scene")
        names = task.metadata.get("controlled_joints", [])
        available = {
            f"{item.name}/{joint.name}": joint
            for item in scene.entities
            for joint in load_resource(artifacts, item.asset, AssetSpec).joints
        }
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(name, str) or name not in available for name in names)
            or len(names) != len(task.action.lower)
        ):
            raise DomainValidationError("task joints are not defined by its scene assets")
        for i, name in enumerate(names):
            assert isinstance(name, str)
            joint = available[name]
            if task.action.lower[i] < joint.lower_limit or task.action.upper[i] > joint.upper_limit:
                raise DomainValidationError("task commands exceed asset joint limits")
        return publish(
            spec,
            artifacts,
            {
                "validation_report": {
                    "schema": "physrealx.task-validation-report/1",
                    "status": "PASSED",
                    "scope": "schema_scene_and_joint_limits",
                    "physics_execution_required": True,
                }
            },
        )
