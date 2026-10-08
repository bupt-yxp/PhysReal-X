"""Bounded scene placement and grounded articulated-task templates."""

from __future__ import annotations

import math
from typing import Any, Literal, cast

from pydantic import Field, model_validator

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    ActionSpec,
    AssetSpec,
    DomainModel,
    GeometryRole,
    JointSpec,
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
    yaw_rad: float = Field(default=0.0, allow_inf_nan=False)


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


class SemanticRelation(DomainModel):
    subject: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    relation: Literal["on", "left_of", "right_of", "front_of", "behind"]
    object: str = Field(pattern=r"^[a-z][a-z0-9_.-]*$")
    clearance_m: float = Field(default=0.05, ge=0, allow_inf_nan=False)


class SemanticSceneGenerationRequest(DomainModel):
    resource_id: str
    version: str
    display_name: str
    domain: Literal["home", "tabletop", "shelf", "industrial"]
    environment: VersionedRef | None = None
    objects: tuple[PlacementObject, ...] = Field(min_length=2, max_length=32)
    relations: tuple[SemanticRelation, ...] = Field(min_length=1, max_length=128)
    xy_bounds_m: tuple[float, float, float, float]
    floor_height_m: float = Field(default=0.0, allow_inf_nan=False)
    interaction_target: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_.-]*$")
    agent_radius_m: float = Field(default=0.30, gt=0, allow_inf_nan=False)
    seed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_request(self) -> SemanticSceneGenerationRequest:
        names = [item.name for item in self.objects]
        x0, y0, x1, y1 = self.xy_bounds_m
        if len(names) != len(set(names)):
            raise ValueError("semantic layout object names must be unique")
        if (
            not all(math.isfinite(value) for value in (*self.xy_bounds_m, self.floor_height_m))
            or x0 >= x1
            or y0 >= y1
        ):
            raise ValueError("semantic layout bounds must be finite ordered intervals")
        known = set(names)
        if self.interaction_target is not None and self.interaction_target not in known:
            raise ValueError("interaction target must reference a scene object")
        if self.environment is not None and "room" in known:
            raise ValueError("semantic layout reserves the room entity name for the environment")
        if (
            self.environment is not None
            and self.environment.resource_type is not ResourceType.ASSET
        ):
            raise ValueError("semantic layout environment must reference an asset")
        supports: dict[str, str] = {}
        dependencies: dict[str, set[str]] = {name: set() for name in names}
        for relation in self.relations:
            if (
                relation.subject not in known
                or relation.object not in known
                or relation.subject == relation.object
            ):
                raise ValueError("semantic relations must reference two distinct scene objects")
            if relation.relation == "on":
                if relation.subject in supports:
                    raise ValueError("an object can have only one support relation")
                supports[relation.subject] = relation.object
            dependencies[relation.subject].add(relation.object)
        for name in names:
            visited = set()
            current = name
            while current in supports:
                if current in visited:
                    raise ValueError("support relations must form an acyclic graph")
                visited.add(current)
                current = supports[current]
        remaining = {name: set(values) for name, values in dependencies.items()}
        while remaining:
            ready = {name for name, values in remaining.items() if not values}
            if not ready:
                raise ValueError("semantic relation dependencies must be acyclic")
            remaining = {
                name: values - ready for name, values in remaining.items() if name not in ready
            }
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


def place_robot_for_joint_task(
    scene: SceneSpec,
    target_asset: AssetSpec,
    robot_asset: AssetSpec,
    store: ArtifactStore,
    *,
    entity: str,
    joint: str,
    robot: VersionedRef,
    name: str,
    end_effector_link: str,
    initial_position: float,
    target_position: float,
    clearance_m: float,
    resource_id: str,
) -> SceneSpec:
    """Place an articulated robot and solve endpoint joint configurations for a target link."""
    import numpy as np
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation

    from physrealx.integrations._asset_preview import (
        asset_geometry,
        link_transforms,
        pose_matrix,
    )

    if any(item.name == name for item in scene.entities):
        raise DomainValidationError(f"scene already contains entity {name!r}")
    target = next((item for item in scene.entities if item.name == entity), None)
    target_joint = next((item for item in target_asset.joints if item.name == joint), None)
    if (
        target is None
        or (target.asset.resource_id, target.asset.version)
        != (target_asset.resource_id, target_asset.version)
        or target_joint is None
        or target_joint.joint_type is JointType.FIXED
    ):
        raise DomainValidationError("robot placement requires a matching movable target joint")
    if (
        not all(math.isfinite(value) for value in (initial_position, target_position, clearance_m))
        or initial_position == target_position
        or clearance_m <= 0
        or not target_joint.lower_limit <= initial_position <= target_joint.upper_limit
        or not target_joint.lower_limit <= target_position <= target_joint.upper_limit
    ):
        raise DomainValidationError(
            "robot placement requires bounded states and positive clearance"
        )
    if end_effector_link not in robot_asset.links:
        raise DomainValidationError("robot end-effector link is absent from the asset")
    controlled = tuple(
        item
        for item in robot_asset.joints
        if item.joint_type is JointType.REVOLUTE and item.effort_limit is not None
    )
    if not controlled:
        raise DomainValidationError("robot placement requires bounded revolute arm joints")
    home = robot_asset.metadata.get("home_positions")
    if not isinstance(home, dict) or any(
        not isinstance(home.get(item.name), (int, float)) for item in controlled
    ):
        raise DomainValidationError("robot asset requires a named home configuration")
    home_positions = np.asarray([float(home[item.name]) for item in controlled])  # type: ignore[arg-type]

    geometry = asset_geometry(store, target_asset, roles=(GeometryRole.COLLISION,))[
        GeometryRole.COLLISION
    ]
    moving_parts = [mesh.vertices for link, mesh in geometry if link == target_joint.child]
    if not moving_parts:
        raise DomainValidationError("target joint child requires collision geometry")
    moving = np.concatenate(moving_parts)
    initial_transform = link_transforms(target_asset, {target_joint.name: initial_position})[
        target_joint.child
    ]
    target_transform = link_transforms(target_asset, {target_joint.name: target_position})[
        target_joint.child
    ]
    initial_points = moving @ initial_transform[:3, :3].T + initial_transform[:3, 3]
    target_points = moving @ target_transform[:3, :3].T + target_transform[:3, 3]
    displacement = np.linalg.norm(target_points - initial_points, axis=1)
    candidates = np.flatnonzero(displacement >= displacement.max() - 1e-6)
    if not len(candidates):
        raise DomainValidationError("target joint does not move its child collision geometry")
    contact_index = int(
        candidates[np.argmax(-initial_points[candidates, 1] + 0.1 * initial_points[candidates, 2])]
    )
    target_pose = pose_matrix(target.pose)
    contacts = []
    for point in (initial_points[contact_index], target_points[contact_index]):
        contacts.append(target_pose[:3, :3] @ point + target_pose[:3, 3])

    home_map = {
        str(name): float(value) for name, value in home.items() if isinstance(value, (int, float))
    }
    home_transforms = link_transforms(robot_asset, home_map)
    home_effector_transform = home_transforms[end_effector_link]
    home_effector = home_effector_transform[:3, 3]
    contact_motion = contacts[1] - contacts[0]
    horizontal_motion = contact_motion.copy()
    horizontal_motion[2] = 0.0
    horizontal_motion_norm = float(np.linalg.norm(horizontal_motion))
    # Revolute contacts follow an arc.  Place the base on that arc's horizontal
    # tangent and center its workspace over both endpoints; using only the first
    # endpoint can make an otherwise feasible closing motion unreachable.
    # Purely vertical motion retains the stable room-facing approach.
    tangent_approach = (
        target_joint.joint_type is JointType.REVOLUTE and horizontal_motion_norm > 1e-4
    )
    if tangent_approach:
        push_direction = horizontal_motion / horizontal_motion_norm
        yaw = math.atan2(float(push_direction[1]), float(push_direction[0]))
    else:
        yaw = math.pi / 2
    base_rotation = Rotation.from_euler("z", yaw)
    forward_world = base_rotation.apply(np.asarray([1.0, 0.0, 0.0]))
    desired_world_orientation = base_rotation.as_matrix() @ home_effector_transform[:3, :3]
    forward_in_effector = desired_world_orientation.T @ forward_world
    descendants = {end_effector_link}
    while True:
        added = {
            item.child
            for item in robot_asset.joints
            if item.parent in descendants and item.child not in descendants
        }
        if not added:
            break
        descendants.update(added)
    tool_vertices = []
    robot_geometry = asset_geometry(store, robot_asset, roles=(GeometryRole.COLLISION,))[
        GeometryRole.COLLISION
    ]
    for link, mesh in robot_geometry:
        if link not in descendants:
            continue
        transform = home_transforms[link]
        root_points = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
        tool_vertices.append(
            (root_points - home_effector_transform[:3, 3]) @ home_effector_transform[:3, :3]
        )
    if not tool_vertices:
        raise DomainValidationError("robot end effector requires collision geometry")
    tool_projection = np.concatenate(tool_vertices) @ forward_in_effector
    leading_tool = tool_projection[tool_projection > 0]
    if not len(leading_tool):
        raise DomainValidationError("robot end effector has no forward contact surface")
    tool_reach = float(np.median(leading_tool))
    hand_contacts = [point - forward_world * tool_reach for point in contacts]
    placement_contact = (
        (hand_contacts[0] + hand_contacts[1]) / 2 if tangent_approach else hand_contacts[0]
    )
    standoff = clearance_m if tangent_approach else clearance_m + 0.15
    base_translation = placement_contact - base_rotation.apply(
        home_effector + np.asarray([standoff, 0.0, 0.0])
    )
    base_translation[2] = 0.0
    base_pose = PoseSpec(
        translation_m=tuple(float(value) for value in base_translation),
        rotation_xyzw=tuple(float(value) for value in base_rotation.as_quat()),
    )

    moving_robot_links = {item.child for item in controlled}
    while True:
        added = {
            item.child
            for item in robot_asset.joints
            if item.parent in moving_robot_links and item.child not in moving_robot_links
        }
        if not added:
            break
        moving_robot_links.update(added)
    static_robot_points = []
    for link, mesh in robot_geometry:
        if link in moving_robot_links:
            continue
        transform = home_transforms[link]
        points = mesh.vertices @ transform[:3, :3].T + transform[:3, 3]
        static_robot_points.append(base_rotation.apply(points) + base_translation)
    if not static_robot_points:
        raise DomainValidationError("robot requires static base collision geometry")
    base_points = np.concatenate(static_robot_points)
    base_bounds = np.asarray((base_points.min(axis=0), base_points.max(axis=0)))
    occupied = scene_bounds(scene, store)
    colliding = []
    for item in scene.entities:
        if item.role is SceneEntityRole.ENVIRONMENT:
            continue
        other = occupied[item.name]
        separated = bool(
            np.any(base_bounds[1] + clearance_m <= other[0])
            or np.any(other[1] + clearance_m <= base_bounds[0])
        )
        if not separated:
            colliding.append(item.name)
    if colliding:
        raise DomainValidationError(
            f"task-aware robot base placement collides with {sorted(colliding)}"
        )

    lower = np.asarray([item.lower_limit for item in controlled]) + 1e-5
    upper = np.asarray([item.upper_limit for item in controlled]) - 1e-5

    def solve(destination: Any, seed: Any) -> Any:
        local = base_rotation.inv().apply(np.asarray(destination) - base_translation)
        seed = np.clip(np.asarray(seed), lower, upper)

        def residual(values: Any, regularization: Any) -> Any:
            positions = {item.name: float(value) for item, value in zip(controlled, values)}
            transform = link_transforms(robot_asset, positions)[end_effector_link]
            return np.r_[
                transform[:3, 3] - local,
                0.00001 * (np.asarray(values) - regularization),
            ]

        span = upper - lower
        joint_offsets = np.arange(1, len(controlled) + 1, dtype=float)
        dispersed = tuple(
            lower + np.mod((sample + 1) * joint_offsets * 0.6180339887498949, 1.0) * span
            for sample in range(12)
        )
        starts = (
            seed,
            np.clip(home_positions, lower, upper),
            lower + 0.50 * span,
            lower + 0.25 * span,
            lower + 0.75 * span,
            *dispersed,
        )
        best = None
        best_error = math.inf
        for start in starts:
            result = least_squares(
                lambda values: residual(values, seed),
                start,
                bounds=(lower, upper),
                max_nfev=500,
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
            )
            position_error = float(np.linalg.norm(residual(result.x, seed)[:3]))
            if result.success and position_error < best_error:
                best, best_error = result.x, position_error
            if best_error <= 0.01:
                break
        if best is None or best_error > 0.01:
            raise DomainValidationError(
                "robot cannot reach the target articulation; "
                f"best endpoint error is {best_error:.4f} m"
            )
        return best

    approach_direction = base_translation - hand_contacts[0]
    approach_direction[2] = 0.0
    approach_norm = float(np.linalg.norm(approach_direction))
    if approach_norm <= 1e-6:
        raise DomainValidationError("robot approach direction is degenerate")
    approach_point = hand_contacts[0] + approach_direction / approach_norm * max(
        0.18, 6 * clearance_m
    )
    world_path = [
        approach_point + fraction * (hand_contacts[0] - approach_point)
        for fraction in np.linspace(0.0, 1.0, 5)
    ]
    for fraction in np.linspace(0.0, 1.0, 9)[1:]:
        position = initial_position + fraction * (target_position - initial_position)
        transform = link_transforms(target_asset, {target_joint.name: position})[target_joint.child]
        point = moving[contact_index] @ transform[:3, :3].T + transform[:3, 3]
        object_contact = target_pose[:3, :3] @ point + target_pose[:3, 3]
        world_path.append(object_contact - forward_world * tool_reach)
    robot_path = []
    seed = home_positions
    for point in world_path:
        seed = solve(point, seed)
        robot_path.append(seed)
    approach_robot = robot_path[0]
    target_robot = robot_path[-1]
    passive = {
        item.name: float(home[item.name])  # type: ignore[arg-type]
        for item in robot_asset.joints
        if item.joint_type is not JointType.FIXED and item not in controlled and item.name in home
    }
    placed = SceneEntitySpec(
        name=name,
        asset=robot,
        role=SceneEntityRole.ROBOT,
        pose=base_pose,
        tags=("open_source_robot",),
    )
    return SceneSpec.model_validate(
        {
            **scene.model_dump(),
            "resource_id": resource_id,
            "entities": (*scene.entities, placed),
            "metadata": {
                **scene.metadata,
                "robot_interaction": {
                    "robot": name,
                    "entity": entity,
                    "joint": joint,
                    "end_effector_link": end_effector_link,
                    "controlled_joints": [item.name for item in controlled],
                    "initial_positions": approach_robot.tolist(),
                    "waypoints": [values.tolist() for values in robot_path[1:-1]],
                    "target_positions": target_robot.tolist(),
                    "passive_robot_positions": passive,
                    "initial_joint_position": initial_position,
                    "target_joint_position": target_position,
                    "contact_points_m": [point.tolist() for point in contacts],
                    "tool_reach_m": tool_reach,
                    "approach_point_m": approach_point.tolist(),
                    "approach_yaw_rad": yaw,
                    "approach_policy": "closing_tangent_or_vertical_front",
                    "base_clearance_m": clearance_m,
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


class SceneSmithAdapter:
    """Place a retrieved candidate with SceneSmith physical and interaction guidance."""

    descriptor = builtin_method(
        MethodCapability.SCENE_GENERATE,
        "scenesmith",
        input_contract="SceneGenerationRequest/1",
        output_contract="SceneDraft/1",
        python=">=3.10,<3.13",
        version="1.1.0",
    ).model_copy(
        update={
            "source_uri": "https://github.com/nepfaff/scenesmith",
            "source_revision": "67cc408fd38334b4a926efef45e284302ed5055b",
            "license_spdx": "MIT",
        }
    )

    def probe(self) -> MethodProbe:
        return SceneGenerationAdapter().probe()

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import numpy as np

        if spec.method.config:
            raise DomainValidationError("layout settings belong in the scene generation request")
        request = SemanticSceneGenerationRequest.model_validate(spec.arguments)
        objects = {item.name: item for item in request.objects}
        assets = {
            name: load_resource(artifacts, item.asset, AssetSpec) for name, item in objects.items()
        }
        local_bounds = {}
        rotations = {}
        for name, asset in assets.items():
            vertices = asset_vertices(asset, artifacts)
            yaw = objects[name].yaw_rad
            cosine, sine = math.cos(yaw), math.sin(yaw)
            rotation = np.asarray(((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)))
            rotated = vertices @ rotation.T
            local_bounds[name] = np.array([rotated.min(0), rotated.max(0)])
            rotations[name] = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        incoming: dict[str, list[SemanticRelation]] = {name: [] for name in objects}
        dependencies: dict[str, set[str]] = {name: set() for name in objects}
        supports: dict[str, SemanticRelation] = {}
        for relation in request.relations:
            incoming[relation.subject].append(relation)
            dependencies[relation.subject].add(relation.object)
            if relation.relation == "on":
                supports[relation.subject] = relation
        order: list[str] = []
        pending = {name: set(values) for name, values in dependencies.items()}
        while pending:
            ready = [
                item.name
                for item in request.objects
                if item.name in pending and not pending[item.name]
            ]
            if not ready:
                raise DomainValidationError("semantic relation graph cannot be ordered")
            order.extend(ready)
            pending = {
                name: values - set(ready) for name, values in pending.items() if name not in ready
            }

        rng = np.random.default_rng(request.seed)
        placed: dict[str, Any] = {}
        entities: list[SceneEntitySpec] = []
        room_lower = np.asarray(request.xy_bounds_m[:2], dtype=float)
        room_upper = np.asarray(request.xy_bounds_m[2:], dtype=float)

        def intersects(candidate: Any, other: Any) -> bool:
            return bool(
                np.all(candidate[0] < other[1] - 1e-6) and np.all(other[0] < candidate[1] - 1e-6)
            )

        for name in order:
            bounds = local_bounds[name]
            support = supports.get(name)
            directional = [item for item in incoming[name] if item.relation != "on"]
            if support is None:
                z = request.floor_height_m - bounds[0, 2]
                area_lower, area_upper = room_lower, room_upper
            else:
                parent = placed[support.object]
                z = parent[1, 2] + support.clearance_m - bounds[0, 2]
                area_lower, area_upper = parent[0, :2], parent[1, :2]
            center_lower = area_lower - bounds[0, :2]
            center_upper = area_upper - bounds[1, :2]
            if np.any(center_lower > center_upper):
                raise MethodExecutionError(f"{name} does not fit on its declared support area")
            center = (center_lower + center_upper) / 2
            axes: set[int] = set()
            for relation in directional:
                reference = placed[relation.object]
                axis = 0 if relation.relation in ("left_of", "right_of") else 1
                if axis in axes:
                    raise DomainValidationError(f"{name} has multiple constraints on one axis")
                axes.add(axis)
                if relation.relation in ("left_of", "front_of"):
                    center[axis] = reference[0, axis] - relation.clearance_m - bounds[1, axis]
                else:
                    center[axis] = reference[1, axis] + relation.clearance_m - bounds[0, axis]
            candidate = bounds + np.r_[center, z]
            if np.any(candidate[0, :2] < area_lower - 1e-8) or np.any(
                candidate[1, :2] > area_upper + 1e-8
            ):
                raise MethodExecutionError(f"semantic constraints place {name} outside its support")
            allowed_overlap = {support.object} if support is not None else set()
            conflicts = [
                other
                for other, other_bounds in placed.items()
                if other not in allowed_overlap and intersects(candidate, other_bounds)
            ]
            if conflicts and not directional:
                for _ in range(200):
                    sampled = rng.uniform(center_lower, center_upper)
                    trial = bounds + np.r_[sampled, z]
                    if not any(
                        other not in allowed_overlap and intersects(trial, other_bounds)
                        for other, other_bounds in placed.items()
                    ):
                        center, candidate, conflicts = sampled, trial, []
                        break
            if conflicts:
                raise MethodExecutionError(
                    f"semantic layout places {name} in collision with {sorted(conflicts)}"
                )
            placed[name] = candidate
            entities.append(
                SceneEntitySpec(
                    name=name,
                    asset=objects[name].asset,
                    role=objects[name].role,
                    pose=PoseSpec(
                        translation_m=tuple(float(value) for value in (*center, z)),
                        rotation_xyzw=rotations[name],
                    ),
                    tags=(str(assets[name].metadata.get("category", "asset")).lower(),),
                )
            )
        reachability_clearance = None
        if request.interaction_target is not None:
            destination = request.interaction_target
            while destination in supports:
                destination = supports[destination].object
            target_center_x = float(placed[request.interaction_target][:, 0].mean())
            corridor = np.asarray(
                (
                    (target_center_x - request.agent_radius_m, room_lower[1]),
                    (target_center_x + request.agent_radius_m, placed[destination][0, 1]),
                )
            )
            blockers = []
            clearances = []
            for other, other_bounds in placed.items():
                if other == destination or other in supports:
                    continue
                if intersects(corridor, other_bounds[:, :2]):
                    blockers.append(other)
                clearances.append(
                    max(
                        float(corridor[0, 0] - other_bounds[1, 0]),
                        float(other_bounds[0, 0] - corridor[1, 0]),
                        0.0,
                    )
                )
            if blockers:
                raise MethodExecutionError(
                    f"interaction corridor to {request.interaction_target} is blocked by "
                    f"{sorted(blockers)}"
                )
            reachability_clearance = min(clearances, default=request.agent_radius_m)
        if request.environment is not None:
            environment = load_resource(artifacts, request.environment, AssetSpec)
            entities.insert(
                0,
                SceneEntitySpec(
                    name="room",
                    asset=request.environment,
                    role=SceneEntityRole.ENVIRONMENT,
                    tags=(str(environment.metadata.get("asset_kind", "environment")),),
                ),
            )
        scene = SceneSpec(
            resource_id=request.resource_id,
            version=request.version,
            display_name=request.display_name,
            entities=tuple(entities),
            metadata={
                "domain": request.domain,
                "seed": request.seed,
                "layout_method": "scenesmith",
                "method_scope": "task_conditioned_candidate_placement_and_physical_screening",
                "method_basis": {
                    "project": "SceneSmith",
                    "source": "https://github.com/nepfaff/scenesmith",
                    "revision": "67cc408fd38334b4a926efef45e284302ed5055b",
                    "license": "MIT",
                    "adaptation": "PhysReal-X canonical SceneSpec and PM URDF bridge",
                    "objectives": [
                        "semantic_relations",
                        "collision",
                        "support",
                        "room_layout",
                        "interaction_reachability",
                    ],
                },
                "relations": [item.model_dump(mode="json") for item in request.relations],
                "interaction_corridor": None
                if request.interaction_target is None
                else {
                    "target": request.interaction_target,
                    "agent_radius_m": request.agent_radius_m,
                    "minimum_clearance_m": reachability_clearance,
                },
                "coordinate_convention": "z_up; front_of=-y",
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
            metrics={
                "entities": float(len(entities)),
                "relations": float(len(request.relations)),
                "reachability_clearance_m": float(reachability_clearance or 0.0),
            },
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


class ArticulatedTaskConfig(DomainModel):
    resource_id: str
    version: str = "1.0.0"
    entity: str
    joint: str
    target_fraction: float = Field(default=0.85, gt=0, lt=1)
    target_position: float | None = Field(default=None, allow_inf_nan=False)
    tolerance: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    max_steps: int = Field(default=80, ge=2, le=10000)
    dt_seconds: float = Field(default=0.05, gt=0, allow_inf_nan=False)
    initial_fraction: float | None = Field(default=None, ge=0, le=1)
    robot: str | None = Field(default=None, pattern=r"^[a-z][a-z0-9_.-]*$")


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
        if config.target_position is not None:
            target = config.target_position
        if not joint.lower_limit <= target <= joint.upper_limit:
            raise DomainValidationError("task target position exceeds joint limits")
        initial = joint.lower_limit if opened else joint.upper_limit
        if config.initial_fraction is not None:
            initial = joint.lower_limit + config.initial_fraction * (
                joint.upper_limit - joint.lower_limit
            )
        controlled: tuple[JointSpec, ...] = (joint,)
        control_entity = entity
        control_initial: tuple[float, ...] = (initial,)
        control_target: tuple[float, ...] = (target,)
        goal_signal = "joint.position"
        passive_positions: dict[str, float] = {}
        control_waypoints: tuple[tuple[float, ...], ...] = ()
        if config.robot is not None:
            interaction = scene.metadata.get("robot_interaction")
            if not isinstance(interaction, dict) or any(
                interaction.get(key) != value
                for key, value in (
                    ("robot", config.robot),
                    ("entity", config.entity),
                    ("joint", config.joint),
                )
            ):
                raise DomainValidationError(
                    "robot task requires matching scene robot-interaction metadata"
                )
            robot_entity = next(
                (item for item in scene.entities if item.name == config.robot), None
            )
            if robot_entity is None or robot_entity.role is not SceneEntityRole.ROBOT:
                raise DomainValidationError("robot task requires a declared robot entity")
            control_entity = robot_entity
            robot = load_resource(artifacts, control_entity.asset, AssetSpec)
            names = interaction.get("controlled_joints")
            initial_values = interaction.get("initial_positions")
            waypoint_values = interaction.get("waypoints")
            target_values = interaction.get("target_positions")
            if (
                not isinstance(names, list)
                or not isinstance(initial_values, list)
                or not isinstance(waypoint_values, list)
                or not isinstance(target_values, list)
                or not all(isinstance(name, str) for name in names)
                or not all(isinstance(value, (int, float)) for value in initial_values)
                or not all(
                    isinstance(row, list) and all(isinstance(value, (int, float)) for value in row)
                    for row in waypoint_values
                )
                or not all(isinstance(value, (int, float)) for value in target_values)
            ):
                raise DomainValidationError("robot interaction joint plan is incomplete")
            by_name = {item.name: item for item in robot.joints}
            try:
                controlled = tuple(by_name[name] for name in names)  # type: ignore[index]
                control_initial = tuple(float(value) for value in initial_values)  # type: ignore[arg-type]
                waypoint_rows = cast(list[list[int | float]], waypoint_values)
                control_waypoints = tuple(
                    tuple(float(value) for value in row) for row in waypoint_rows
                )
                control_target = tuple(float(value) for value in target_values)  # type: ignore[arg-type]
            except (KeyError, TypeError, ValueError) as error:
                raise DomainValidationError("robot interaction joint plan is invalid") from error
            if (
                not controlled
                or not (len(controlled) == len(control_initial) == len(control_target))
                or any(len(row) != len(controlled) for row in control_waypoints)
            ):
                raise DomainValidationError("robot interaction vectors must align")
            goal_signal = f"joint.{config.entity}.{config.joint}.position"
            passive_positions[f"{config.entity}/{config.joint}"] = initial
        if any(item.effort_limit is None for item in controlled) or any(
            not item.lower_limit <= value <= item.upper_limit
            for item, values in zip(
                controlled,
                zip(control_initial, *control_waypoints, control_target),
            )
            for value in values
        ):
            raise DomainValidationError("control requires an effort limit and bounded positions")
        units = {"m" if item.joint_type is JointType.PRISMATIC else "rad" for item in controlled}
        if len(units) != 1:
            raise DomainValidationError("one robot action requires homogeneous joint units")
        unit = units.pop()
        frequency = 1 / config.dt_seconds
        observations = tuple(
            SignalSpec(
                name=name,
                shape=shape,
                dtype=SignalDType.FLOAT32,
                unit=u,
                frame="joint",
                frequency_hz=frequency,
            )
            for name, u, shape in (
                ("joint.position", unit, (len(controlled),)),
                ("joint.velocity", unit + "/s", (len(controlled),)),
                ("joint.effort", "N" if unit == "m" else "N*m", (len(controlled),)),
                ("contact.force", "N", (1,)),
            )
        )
        if config.robot is not None:
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
                    shape=(len(controlled),),
                    dtype=SignalDType.FLOAT32,
                    unit=unit,
                    frame="joint",
                    frequency_hz=frequency,
                ),
                lower=tuple(item.lower_limit for item in controlled),
                upper=tuple(item.upper_limit for item in controlled),
            ),
            observations=observations,
            max_steps=config.max_steps,
            metadata={
                "task_template": "open_joint" if opened else "close_joint",
                "controlled_joints": [f"{control_entity.name}/{item.name}" for item in controlled],
                "initial_positions": list(control_initial),
                "waypoints": [list(row) for row in control_waypoints],
                "target_positions": list(control_target),
                "passive_positions": passive_positions,
                "dt_seconds": config.dt_seconds,
                "effort_limits": [item.effort_limit for item in controlled],
                "evidence_scope": "robot_contact" if config.robot else "articulated_object_control",
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
