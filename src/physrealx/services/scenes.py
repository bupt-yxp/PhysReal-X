"""Scene composition, task compilation and trajectory generation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from physrealx.domain import (
    ArtifactRef,
    JSONValue,
    MethodCapability,
    MethodSpec,
    OperationManifest,
    ResourceType,
    SceneEntitySpec,
    SceneSpec,
    TaskSpec,
    VersionedRef,
)
from physrealx.errors import (
    DomainValidationError,
    ResourceConflictError,
)

if TYPE_CHECKING:
    from physrealx.integrations.sensors import CameraSpec
    from physrealx.platform import PhysRealX


class ScenesService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def register(self, spec: SceneSpec) -> VersionedRef:
        resolved_entities: list[SceneEntitySpec] = []
        for entity in spec.entities:
            asset_ref = self._platform.repository.get_resource(
                ResourceType.ASSET.value,
                entity.asset.resource_id,
                entity.asset.version,
            )
            if (
                entity.asset.artifact_id is not None
                and entity.asset.artifact_id != asset_ref.artifact_id
            ):
                raise ResourceConflictError(
                    f"scene entity {entity.name!r} asset does not match the registry"
                )
            resolved_entities.append(entity.model_copy(update={"asset": asset_ref}))
        resolved = spec.model_copy(update={"entities": tuple(resolved_entities)})
        return self._platform._register(
            resolved,
            resolved.reference(),
            "application/vnd.physrealx.scene+json",
        )

    def get(self, reference: VersionedRef) -> SceneSpec:
        return self._platform._load_registered(reference, SceneSpec)

    def list(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.SCENE)

    def place_robot_for_joint_task(
        self,
        scene: SceneSpec,
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
        """Place a canonical articulated robot and solve target-link endpoint poses."""
        from physrealx.integrations.scenes import place_robot_for_joint_task

        target = next((item for item in scene.entities if item.name == entity), None)
        if target is None:
            raise DomainValidationError(f"scene entity not found: {entity}")
        return place_robot_for_joint_task(
            scene,
            self._platform.assets.get(target.asset),
            self._platform.assets.get(robot),
            self._platform.artifacts,
            entity=entity,
            joint=joint,
            robot=robot,
            name=name,
            end_effector_link=end_effector_link,
            initial_position=initial_position,
            target_position=target_position,
            clearance_m=clearance_m,
            resource_id=resource_id,
        )

    def generate(
        self,
        request: Mapping[str, JSONValue],
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.SCENE_GENERATE,
            method,
            config=config,
            arguments=request,
        )

    def compose(
        self,
        scene: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.SCENE_COMPOSE,
            method,
            inputs={"scene": scene},
            config=config,
        )

    def preview_task(
        self,
        scene: VersionedRef,
        task: VersionedRef,
        trajectory: ArtifactRef,
        *,
        frames: int = 16,
        fps: int = 8,
    ) -> ArtifactRef:
        """Render one Blender scene-execution preview from recorded states."""
        from physrealx.integrations.task_preview import preview_task_execution

        return preview_task_execution(
            self._platform.artifacts,
            scene,
            task,
            trajectory,
            frames=frames,
            fps=fps,
        )


class TasksService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def register(self, spec: TaskSpec) -> VersionedRef:
        scene_ref = self._platform.repository.get_resource(
            ResourceType.SCENE.value,
            spec.scene.resource_id,
            spec.scene.version,
        )
        if spec.scene.artifact_id is not None and spec.scene.artifact_id != scene_ref.artifact_id:
            raise ResourceConflictError("task scene does not match the registered scene")
        scene = self._platform._load_registered(scene_ref, SceneSpec)
        if spec.subject_entity not in {entity.name for entity in scene.entities}:
            raise DomainValidationError("task subject_entity does not exist in the scene")
        resolved = spec.model_copy(update={"scene": scene_ref})
        return self._platform._register(
            resolved,
            resolved.reference(),
            "application/vnd.physrealx.task+json",
        )

    def get(self, reference: VersionedRef) -> TaskSpec:
        return self._platform._load_registered(reference, TaskSpec)

    def list(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.TASK)

    def with_sensors(
        self,
        task: TaskSpec,
        scene: VersionedRef,
        cameras: Mapping[str, CameraSpec],
        *,
        resource_id: str,
        frequency_hz: float,
    ) -> TaskSpec:
        from physrealx.integrations.sensors import with_sensor_signals

        if task.scene != scene:
            raise DomainValidationError("sensor task and scene references must match")
        layout = self._platform.scenes.get(scene)
        link_count = sum(len(self._platform.assets.get(e.asset).links) for e in layout.entities)
        return with_sensor_signals(
            task,
            dict(cameras),
            link_count=link_count,
            resource_id=resource_id,
            frequency_hz=frequency_hz,
        )

    def compile(
        self,
        instruction: str,
        scene: VersionedRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.TASK_COMPILE,
            method,
            inputs={"scene": scene},
            config=config,
            arguments={"instruction": instruction},
        )

    def validate(
        self,
        task: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.TASK_VALIDATE,
            method,
            inputs={"task": task},
            config=config,
        )


class TrajectoriesService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def generate(
        self,
        instance: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.TRAJECTORY_GENERATE,
            method,
            inputs={"instance": instance},
            config=config,
        )
