"""Policy training, model registration and world-model learning."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from physrealx.domain import (
    ArtifactRef,
    Episode,
    JSONValue,
    MethodCapability,
    MethodSpec,
    ModelManifest,
    OperationManifest,
    ResourceType,
    TrainingSpec,
    VersionedRef,
)
from physrealx.errors import (
    DomainValidationError,
    ResourceConflictError,
    ResourceNotFoundError,
)

if TYPE_CHECKING:
    from physrealx.platform import PhysRealX


class TrainingService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def submit(self, spec: TrainingSpec) -> VersionedRef:
        registered = self._platform.repository.get_dataset(
            spec.dataset.resource_id, spec.dataset.version
        )
        if registered.artifact_id != spec.dataset.artifact_id:
            raise ResourceConflictError("training dataset does not match the registry")
        dataset = self._platform.datasets.get(registered)
        self._platform.plugins.resolve_method(spec.trainer)
        if spec.trainer.checkpoint is not None and not self._platform.artifacts.verify(
            spec.trainer.checkpoint
        ):
            raise DomainValidationError("trainer checkpoint is not present in artifact storage")
        try:
            existing = self._platform.repository.get_model(spec.resource_id, spec.version)
        except ResourceNotFoundError:
            existing = None
        if existing is not None:
            manifest = self.get_model(existing)
            if manifest.dataset != registered or manifest.trainer != spec.trainer:
                raise ResourceConflictError(
                    f"model/{spec.resource_id}@{spec.version} "
                    "already uses a different training spec"
                )
        else:
            episodes = (
                self._platform._executor.load_episode(summary) for summary in dataset.episodes
            )
            trainer = self._platform.plugins.trainer(spec.trainer.name)
            manifest = trainer.train(spec, dataset, episodes, self._platform.artifacts)
        if (
            manifest.resource_id != spec.resource_id
            or manifest.version != spec.version
            or manifest.dataset != registered
        ):
            raise DomainValidationError("trainer returned a model outside the requested lineage")
        if manifest.trainer != spec.trainer:
            raise DomainValidationError("trainer returned mismatched method identity")
        self._platform.plugins.policy_descriptor(manifest.policy.name)
        self._platform.plugins.resolve_method(manifest.policy)
        if manifest.action_signal != dataset.action_schema:
            raise DomainValidationError("trainer changed the dataset action schema")
        if not self._platform.artifacts.verify(manifest.weights):
            raise DomainValidationError("trainer returned unpublished or corrupt model weights")
        if existing is not None:
            return existing
        return self._platform._register(
            manifest,
            manifest.reference(),
            "application/vnd.physrealx.model+json",
            specialized="model",
        )

    def get_model(self, reference: VersionedRef) -> ModelManifest:
        return self._platform._load_registered(reference, ModelManifest)

    def list_models(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.MODEL)

    def preview_policy(
        self,
        model: VersionedRef,
        evaluation: VersionedRef,
    ) -> ArtifactRef:
        """Render policy optimization and closed-loop evaluation evidence."""
        from physrealx.integrations.learning_preview import preview_policy_training

        return preview_policy_training(
            self._platform.artifacts,
            model,
            self.get_model(model),
            self._platform.evaluations.get_report(evaluation),
        )

    def prepare_interface(
        self,
        snapshot: ArtifactRef,
        *,
        method: Literal["openvla_oft", "td_mpc2"],
        config: Mapping[str, JSONValue] | None = None,
    ) -> ArtifactRef:
        """Validate a snapshot hand-off to one pinned official training runtime."""
        from physrealx.integrations.training_interfaces import prepare_training_interface

        return prepare_training_interface(
            self._platform.artifacts,
            snapshot,
            method=method,
            config=config,
        )

    def preview_interfaces(self, interfaces: tuple[ArtifactRef, ...]) -> ArtifactRef:
        """Show validated external interfaces and their execution status."""
        from physrealx.integrations.training_interfaces import preview_training_interfaces

        return preview_training_interfaces(self._platform.artifacts, interfaces)


class ModelsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def register(
        self,
        model: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.MODEL_REGISTER,
            method,
            inputs={"model": model},
            config=config,
        )


class WorldModelsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def prepare_encoder(
        self, *, cache: Path | None = None, weights_path: Path | None = None
    ) -> ArtifactRef:
        """Prepare DINOv2 from verified cache, local weights or the official source."""
        from physrealx.integrations.world_model import prepare_dinov2_encoder

        return prepare_dinov2_encoder(
            self._platform.artifacts,
            cache
            if cache is not None
            else self._platform.workspace / "pretrained/dinov2-encoder.json",
            weights_path=weights_path,
        )

    def train(
        self,
        dataset: ArtifactRef,
        encoder: ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.WORLD_MODEL_TRAIN,
            method,
            inputs={"dataset": dataset, "encoder": encoder},
            config=config,
        )

    def rollout(
        self,
        model: VersionedRef | ArtifactRef,
        context: VersionedRef | ArtifactRef,
        actions: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.WORLD_MODEL_ROLLOUT,
            method,
            inputs={"model": model, "context": context, "actions": actions},
            config=config,
        )

    def evaluate_rollout(
        self,
        rollout: ArtifactRef,
        reference: Episode,
        *,
        start_index: int = 0,
    ) -> ArtifactRef:
        """Compare predicted states with a time-aligned episode and persistence."""
        from physrealx.integrations.world_model import evaluate_world_model_rollout

        return evaluate_world_model_rollout(
            self._platform.artifacts,
            rollout,
            reference,
            start_index=start_index,
        )

    def preview_rollout(
        self,
        rollout: ArtifactRef,
        evaluation: ArtifactRef,
        reference: Episode,
        *,
        camera: str = "front",
        frames: int = 8,
        fps: int = 4,
    ) -> ArtifactRef:
        """Animate world-model RGB and state predictions against aligned truth."""
        from physrealx.integrations.learning_preview import preview_world_model_rollout

        return preview_world_model_rollout(
            self._platform.artifacts,
            rollout,
            evaluation,
            reference,
            camera=camera,
            frames=frames,
            fps=fps,
        )
