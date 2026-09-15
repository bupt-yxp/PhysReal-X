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
from physrealx.reconstruction import joint_transform


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


def _asset_collision_meshes(asset: AssetSpec, store: ArtifactStore, link: str | None) -> list[Any]:
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation

    poses = _link_poses(asset)
    result = []
    for geometry in asset.geometries:
        if geometry.role.value != "collision" or (link is not None and geometry.link != link):
            continue
        meshes = (
            [trimesh.creation.box(extents=geometry.primitive_dimensions_m)]
            if geometry.primitive_dimensions_m is not None
            else load_meshes(store, mesh_reference(geometry.uri, store))
        )
        local = np.eye(4)
        local[:3, :3] = Rotation.from_quat(geometry.local_pose.rotation_xyzw).as_matrix()
        local[:3, 3] = geometry.local_pose.translation_m
        for mesh in meshes:
            mesh.apply_scale(geometry.scale)
            mesh.apply_transform(poses[geometry.link] @ local)
            result.append(mesh)
    if not result:
        raise DomainValidationError("scene placement requires collision geometry")
    return result


def asset_vertices(asset: AssetSpec, store: ArtifactStore, *, link: str | None = None) -> Any:
    import numpy as np

    return np.concatenate([mesh.vertices for mesh in _asset_collision_meshes(asset, store, link)])


def _contact_support(meshes: list[Any], transform: Any, lower: Any, upper: Any) -> float:
    """Find the outward surface within the actuator's actual Y/Z contact window."""
    import numpy as np

    support = -math.inf
    for mesh in meshes:
        points = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
        triangles = points[mesh.faces]
        relevant = np.all(triangles[:, :, 1:].min(axis=1) <= upper, axis=1) & np.all(
            triangles[:, :, 1:].max(axis=1) >= lower, axis=1
        )
        for triangle in triangles[relevant]:
            polygon = list(triangle)
            for axis, bound, sign in (
                (1, lower[0], 1),
                (1, upper[0], -1),
                (2, lower[1], 1),
                (2, upper[1], -1),
            ):
                clipped = []
                for previous, current in zip(polygon[-1:] + polygon[:-1], polygon):
                    before = sign * (previous[axis] - bound) >= 0
                    after = sign * (current[axis] - bound) >= 0
                    if before != after:
                        clipped.append(
                            previous
                            + (current - previous)
                            * ((bound - previous[axis]) / (current[axis] - previous[axis]))
                        )
                    if after:
                        clipped.append(current)
                polygon = clipped
                if not polygon:
                    break
            if polygon:
                support = max(support, max(float(point[0]) for point in polygon))
    if not math.isfinite(support):
        raise DomainValidationError("actuator contact window misses target collision geometry")
    return support


def place_linear_actuator(
    scene: SceneSpec,
    asset: AssetSpec,
    store: ArtifactStore,
    *,
    entity: str,
    joint: str,
    actuator: VersionedRef,
    name: str,
    initial_position: float,
    clearance_m: float,
    resource_id: str,
) -> SceneSpec:
    """Place a linear contact actuator outside a posed slide or hinged link."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    if any(e.name == name for e in scene.entities):
        raise DomainValidationError(f"scene already contains entity {name!r}")
    target = next((e for e in scene.entities if e.name == entity), None)
    target_joint = next((j for j in asset.joints if j.name == joint), None)
    if (
        target is None
        or (target.asset.resource_id, target.asset.version) != (asset.resource_id, asset.version)
        or target_joint is None
    ):
        raise DomainValidationError("actuator placement requires a matching scene entity and joint")
    if target_joint.joint_type not in (JointType.PRISMATIC, JointType.REVOLUTE):
        raise DomainValidationError("linear actuator placement requires a movable target joint")
    if (
        not math.isfinite(initial_position)
        or not target_joint.lower_limit <= initial_position <= target_joint.upper_limit
        or not math.isfinite(clearance_m)
        or clearance_m <= 0
    ):
        raise DomainValidationError(
            "actuator placement requires valid joint position and positive clearance"
        )
    parent = _link_poses(asset)[target_joint.parent]
    joint_rotation = Rotation.from_matrix(parent[:3, :3]) * Rotation.from_quat(
        target_joint.parent_to_joint.rotation_xyzw
    )
    opening = joint_rotation.apply(target_joint.axis)
    closed = asset_vertices(asset, store, link=target_joint.child)
    origin = parent[:3, :3] @ np.asarray(target_joint.parent_to_joint.translation_m) + parent[:3, 3]
    motion = joint_transform(
        "slide" if target_joint.joint_type is JointType.PRISMATIC else "hinge",
        opening,
        origin,
        initial_position,
    )
    posed = closed @ motion[:3, :3].T + motion[:3, 3]
    if target_joint.joint_type is JointType.REVOLUTE:
        opening = np.mean(posed - closed, axis=0)
        opening[2] = 0
        if np.linalg.norm(opening) < 1e-6:
            raise DomainValidationError(
                "hinged actuator placement needs a nonzero horizontal opening"
            )
        opening /= np.linalg.norm(opening)
    rotation, _ = Rotation.align_vectors([opening], [[1, 0, 0]])
    points = rotation.inv().apply(posed)
    lo, hi = points.min(0), points.max(0)
    actuator_model = load_resource(store, actuator, AssetSpec)
    actuator_joints = [j for j in actuator_model.joints if j.joint_type is not JointType.FIXED]
    if (
        len(actuator_joints) != 1
        or actuator_joints[0].joint_type is not JointType.PRISMATIC
        or not np.allclose(actuator_joints[0].axis, [-1, 0, 0])
    ):
        raise DomainValidationError("contact actuator must have one local -X sliding joint")
    actuator_vertices = asset_vertices(actuator_model, store, link=actuator_joints[0].child)
    center = (lo[1:] + hi[1:]) / 2
    window_lower = center + actuator_vertices[:, 1:].min(axis=0)
    window_upper = center + actuator_vertices[:, 1:].max(axis=0)
    meshes = _asset_collision_meshes(asset, store, target_joint.child)
    projection = np.eye(4)
    projection[:3, :3] = rotation.inv().as_matrix()
    initial_support = _contact_support(meshes, projection @ motion, window_lower, window_upper)
    closed_motion = joint_transform(
        "slide" if target_joint.joint_type is JointType.PRISMATIC else "hinge",
        joint_rotation.apply(target_joint.axis),
        origin,
        target_joint.lower_limit,
    )
    closed_support = _contact_support(
        meshes, projection @ closed_motion, window_lower, window_upper
    )
    front_offset = float(actuator_vertices[:, 0].min())
    relative = rotation.apply([initial_support + clearance_m - front_offset, *center])
    scene_rotation = Rotation.from_quat(target.pose.rotation_xyzw)
    placed = SceneEntitySpec(
        name=name,
        asset=actuator,
        role=SceneEntityRole.ROBOT,
        pose=PoseSpec(
            translation_m=np.asarray(target.pose.translation_m) + scene_rotation.apply(relative),
            rotation_xyzw=(scene_rotation * rotation).as_quat(),
        ),
    )
    travel = float(initial_support + clearance_m - closed_support)
    if not actuator_joints[0].lower_limit <= travel <= actuator_joints[0].upper_limit:
        raise DomainValidationError("contact closing travel exceeds actuator limits")
    return SceneSpec.model_validate(
        {
            **scene.model_dump(),
            "resource_id": resource_id,
            "entities": (*scene.entities, placed),
            "metadata": {
                **scene.metadata,
                "actuator_placement": {
                    "entity": entity,
                    "joint": joint,
                    "closing_travel_m": travel,
                    "initial_joint_position": initial_position,
                    "joint_unit": "m" if target_joint.joint_type is JointType.PRISMATIC else "rad",
                },
            },
        }
    )


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
