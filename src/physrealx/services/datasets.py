"""Dataset publication, selection and inspection."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from physrealx.domain import (
    ArtifactRef,
    DatasetManifest,
    JSONValue,
    MethodCapability,
    MethodSpec,
    OperationManifest,
    ResourceType,
    VersionedRef,
)
from physrealx.errors import (
    ResourceConflictError,
    ResourceNotFoundError,
)

if TYPE_CHECKING:
    from physrealx.platform import PhysRealX


class DatasetsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def get(self, reference: VersionedRef) -> DatasetManifest:
        return self._platform._load_registered(reference, DatasetManifest)

    def list(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.DATASET)

    def register(self, manifest: DatasetManifest) -> VersionedRef:
        """Publish an assembled or selected dataset after checking its actual episodes."""
        from physrealx.integrations.datasets import validate_dataset

        validate_dataset(self._platform.artifacts, manifest)
        return self._platform._register(
            manifest, manifest.reference(), "application/vnd.physrealx.dataset+json"
        )

    def preview_episode(
        self,
        dataset: VersionedRef,
        *,
        episode_index: int = 0,
        camera: str | None = None,
        frames: int = 12,
        fps: int = 6,
        sensor_source: str = "blender",
    ) -> ArtifactRef:
        """Render one recorded episode with synchronized physics and sensor views."""
        from physrealx.integrations.simulation_preview import preview_simulation_episode

        return preview_simulation_episode(
            self._platform.artifacts,
            dataset,
            episode_index=episode_index,
            camera=camera,
            frames=frames,
            fps=fps,
            sensor_source=sensor_source,
        )

    def preview_snapshot(
        self,
        snapshot: ArtifactRef,
        *,
        camera: str = "front",
    ) -> ArtifactRef:
        """Render modalities, sequences, labels and split membership from a snapshot."""
        from physrealx.integrations.learning_preview import preview_dataset_snapshot

        return preview_dataset_snapshot(self._platform.artifacts, snapshot, camera=camera)

    def ingest(
        self,
        source: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.DATASET_INGEST,
            method,
            inputs={"source": source},
            config=config,
        )

    def snapshot(
        self,
        dataset: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.DATASET_SNAPSHOT,
            method,
            inputs={"dataset": dataset},
            config=config,
        )

    def training_subset(
        self,
        snapshot: ArtifactRef,
        *,
        resource_id: str,
        version: str = "1.0.0",
        augmentations: tuple[VersionedRef, ...] = (),
        excluded_datasets: tuple[VersionedRef, ...] = (),
        required_tag: str | None = None,
        selection: Literal["successful_demonstrations", "all_episodes"] = (
            "successful_demonstrations"
        ),
    ) -> VersionedRef:
        from physrealx.integrations.datasets import training_subset

        selected = training_subset(
            self._platform.artifacts,
            snapshot,
            resource_id=resource_id,
            version=version,
            augmentations=augmentations,
            excluded_datasets=excluded_datasets,
            required_tag=required_tag,
            selection=selection,
        )
        try:
            existing_ref = self._platform.repository.get_dataset(resource_id, version)
        except ResourceNotFoundError:
            existing_ref = None
        if existing_ref is not None:
            existing = self.get(existing_ref)
            if existing.model_dump(exclude={"created_at"}) == selected.model_dump(
                exclude={"created_at"}
            ):
                return existing_ref
            raise ResourceConflictError(
                f"dataset/{resource_id}@{version} already uses a different training selection"
            )
        return self.register(selected)
