"""Canonical episode quality, immutable dataset assembly, and leakage-resistant splitting."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Literal

from pydantic import Field

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    ArtifactRef,
    DatasetManifest,
    DomainModel,
    Episode,
    MethodCapability,
    OperationSpec,
    VersionedRef,
)
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import (
    input_artifact,
    load_resource,
    publish,
    read_array,
    read_json,
)
from physrealx.integrations.provenance import capture_runtime
from physrealx.methods import builtin_method
from physrealx.runtime import load_episode
from physrealx.training_lifecycle import DatasetSnapshot, DatasetSplit, EpisodeArtifact


def validate_dataset(store: ArtifactStore, dataset: DatasetManifest) -> dict[str, Any]:
    """Validate actual arrays, schemas, labels and synchronization before dataset publication."""
    episodes, transitions, arrays = 0, 0, 0
    checked: set[str] = set()
    source_counts: dict[str, int] = defaultdict(int)
    for summary in dataset.episodes:
        episode = load_episode(store, summary)
        source_counts[episode.source.value] += 1
        episodes += 1
        transitions += len(episode.steps) - 1
        for step in episode.steps:
            if set(step.observations) != {signal.name for signal in dataset.observation_schema}:
                raise DomainValidationError("episode observations differ from the dataset schema")
            for signal in dataset.observation_schema:
                value = step.observations[signal.name]
                value.assert_matches(signal)
                if value.timestamp_seconds is None or not math.isclose(
                    value.timestamp_seconds, step.timestamp_seconds, abs_tol=1e-9
                ):
                    raise DomainValidationError(
                        "dataset observations are not synchronized to the episode step"
                    )
                if value.artifact is not None and value.artifact.artifact_id not in checked:
                    array = read_array(store, value.artifact)
                    if array.shape != value.shape or str(array.dtype) != value.dtype.value:
                        raise DomainValidationError(
                            "observation array differs from its declared shape or dtype"
                        )
                    checked.add(value.artifact.artifact_id)
                    arrays += 1
            if step.action is not None:
                step.action.assert_matches(dataset.action_schema)
    return {
        "schema": "physrealx.data-quality-report/1",
        "status": "PASSED",
        "episodes": episodes,
        "transitions": transitions,
        "checked_arrays": arrays,
        "source_counts": dict(source_counts),
        "successful_episodes": sum(item.success for item in dataset.episodes),
        "synchronized": True,
    }


class EpisodeCollection(DomainModel):
    schema_id: Literal["physrealx.episode-collection/1"] = Field(
        default="physrealx.episode-collection/1", alias="schema"
    )
    datasets: tuple[VersionedRef, ...] = Field(min_length=1)


class DatasetConfig(DomainModel):
    resource_id: str
    version: str = "1.0.0"


class EpisodeDatasetAdapter:
    descriptor = builtin_method(
        MethodCapability.DATASET_INGEST,
        "episode_dataset",
        input_contract="EpisodeBundle/1",
        output_contract="DatasetBundle/1+DataQualityReport/1",
    )

    def probe(self) -> MethodProbe:
        return MethodProbe(True, "Canonical episode and dataset validators available")

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        config = DatasetConfig.model_validate(spec.method.config)
        source = input_artifact(spec, "source", artifacts)
        collection = EpisodeCollection.model_validate(
            read_json(artifacts, source, "physrealx.episode-collection/1")
        )
        datasets = [load_resource(artifacts, ref, DatasetManifest) for ref in collection.datasets]
        first = datasets[0]
        if any(
            (dataset.observation_schema, dataset.action_schema)
            != (first.observation_schema, first.action_schema)
            for dataset in datasets
        ):
            raise DomainValidationError(
                "datasets must share the same observation and action schemas"
            )
        summaries = tuple(summary for dataset in datasets for summary in dataset.episodes)
        dataset = DatasetManifest(
            resource_id=config.resource_id,
            version=config.version,
            episodes=summaries,
            source_runs=tuple(dict.fromkeys(summary.run_id for summary in summaries)),
            observation_schema=first.observation_schema,
            action_schema=first.action_schema,
            total_steps=sum(summary.step_count for summary in summaries),
            metadata={"source_collection": source.model_dump(mode="json")},
        )
        quality = validate_dataset(artifacts, dataset)
        return publish(
            spec,
            artifacts,
            {
                "dataset": {
                    "schema": "physrealx.dataset-bundle/1",
                    "manifest": dataset.model_dump(mode="json"),
                },
                "quality_report": quality,
            },
            metrics={
                "episodes": float(len(summaries)),
                "transitions": float(quality["transitions"]),
            },
        )


def dataset_bundle(store: ArtifactStore, reference: ArtifactRef) -> DatasetManifest:
    document = read_json(store, reference, "physrealx.dataset-bundle/1")
    return DatasetManifest.model_validate(document["manifest"])


def episode_group(episode: Episode) -> tuple[str, ...]:
    if episode.source.value == "real":
        capture = episode.metadata.get("capture_id")
        if not isinstance(capture, str) or not capture:
            raise DomainValidationError("real-data splitting requires an explicit capture_id group")
        return ("real", capture)
    return (
        episode.source.value,
        episode.scene.resource_id,
        episode.scene.version,
        episode.task.resource_id,
        episode.task.version,
        str(episode.seed),
    )


def load_snapshot(
    store: ArtifactStore, reference: ArtifactRef
) -> tuple[DatasetSnapshot, DatasetManifest, dict[str, list[Episode]]]:
    snapshot = DatasetSnapshot.model_validate(
        read_json(store, reference, "physrealx.dataset-snapshot/1")
    )
    if any(
        not store.verify(item)
        for item in (snapshot.manifest, snapshot.code.source_tree, snapshot.environment.lockfile)
    ):
        raise DomainValidationError("snapshot source, environment or manifest evidence is missing")
    dataset = load_resource(store, snapshot.dataset, DatasetManifest)
    if snapshot.manifest.artifact_id != snapshot.dataset.artifact_id:
        raise DomainValidationError("snapshot dataset and manifest identities disagree")
    if {item.episode_id: item.artifact for item in snapshot.episodes} != {
        item.episode_id: item.artifact for item in dataset.episodes
    }:
        raise DomainValidationError("snapshot episode payloads differ from its dataset")
    membership_by_split = {
        name: getattr(snapshot.split, name) for name in ("train", "validation", "test")
    }
    episodes = {summary.episode_id: load_episode(store, summary) for summary in dataset.episodes}
    membership = {identifier for split in membership_by_split.values() for identifier in split}
    if membership != set(episodes):
        raise DomainValidationError("snapshot membership differs from its source dataset")
    owners: dict[tuple[str, ...], str] = {}
    result: dict[str, list[Episode]] = {}
    for name, identifiers in membership_by_split.items():
        result[name] = [episodes[identifier] for identifier in identifiers]
        for episode in result[name]:
            key = episode_group(episode)
            if key in owners and owners[key] != name:
                raise DomainValidationError("snapshot leaks a capture or scene/task/seed group")
            owners[key] = name
    return snapshot, dataset, result


def training_subset(
    store: ArtifactStore,
    snapshot_ref: ArtifactRef,
    *,
    resource_id: str,
    version: str,
    augmentations: tuple[VersionedRef, ...] = (),
    required_tag: str | None = None,
    selection: Literal["successful_demonstrations", "all_episodes"] = ("successful_demonstrations"),
) -> DatasetManifest:
    """Select a purpose-specific train split without leaking held-out groups."""
    _, dataset, partitions = load_snapshot(store, snapshot_ref)
    if selection not in {"successful_demonstrations", "all_episodes"}:
        raise DomainValidationError("unsupported training episode selection")
    if selection == "all_episodes" and required_tag is not None:
        raise DomainValidationError("all-episode training cannot require a demonstration tag")
    protected = {
        episode_group(episode) for name in ("validation", "test") for episode in partitions[name]
    }

    def selected_episode(episode: Episode) -> bool:
        if selection == "all_episodes":
            return True
        tags = episode.metadata.get("tags", [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise DomainValidationError("demonstration tags must be a list of strings")
        return episode.success and (required_tag is None or required_tag in tags)

    selected = {episode.episode_id for episode in partitions["train"] if selected_episode(episode)}
    summaries = [row for row in dataset.episodes if row.episode_id in selected]
    for reference in augmentations:
        extra = load_resource(store, reference, DatasetManifest)
        validate_dataset(store, extra)
        if (extra.observation_schema, extra.action_schema) != (
            dataset.observation_schema,
            dataset.action_schema,
        ):
            raise DomainValidationError(
                "training augmentation has different observation/action schemas"
            )
        for summary in extra.episodes:
            episode = load_episode(store, summary)
            if episode_group(episode) in protected:
                raise DomainValidationError(
                    "training augmentation overlaps a held-out capture or scenario"
                )
            if not selected_episode(episode):
                raise DomainValidationError(
                    "imitation augmentation requires successful demonstrations "
                    "with the required tag"
                )
            if episode.episode_id in selected:
                raise DomainValidationError("training augmentation duplicates an episode")
            summaries.append(summary)
            selected.add(episode.episode_id)
    if not summaries:
        raise DomainValidationError("training selection contains no episodes")
    return DatasetManifest(
        resource_id=resource_id,
        version=version,
        episodes=tuple(summaries),
        source_runs=tuple(dict.fromkeys(row.run_id for row in summaries)),
        observation_schema=dataset.observation_schema,
        action_schema=dataset.action_schema,
        total_steps=sum(row.step_count for row in summaries),
        metadata={
            "snapshot": snapshot_ref.model_dump(mode="json"),
            "split": "train",
            "selection": selection,
            "required_tag": required_tag,
            "augmentations": [ref.model_dump(mode="json") for ref in augmentations],
            "protected_groups": len(protected),
        },
    )


class SplitConfig(DomainModel):
    seed: int = Field(default=2026, ge=0)
    validation_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    test_fraction: float = Field(default=0.2, gt=0, lt=0.5)
    stratify_success: bool = False


def _interleave_strata(
    strata: dict[tuple[bool, ...], list[tuple[str, ...]]],
    rng: Any,
) -> list[tuple[str, ...]]:
    """Shuffle within outcome strata and interleave them deterministically."""
    queues = {label: sorted(values) for label, values in strata.items()}
    labels = sorted(queues)
    rng.shuffle(labels)
    for values in queues.values():
        rng.shuffle(values)
    result: list[tuple[str, ...]] = []
    while any(queues.values()):
        for label in labels:
            if queues[label]:
                result.append(queues[label].pop())
    return result


def _stratified_partitions(
    groups: dict[tuple[str, ...], list[str]],
    episodes: dict[str, Episode],
    *,
    seed: int,
    test_count: int,
    validation_count: int,
) -> dict[str, list[tuple[str, ...]]]:
    """Keep every observed outcome stratum in training when cardinality permits."""
    import random

    train_count = len(groups) - test_count - validation_count
    strata: dict[tuple[bool, ...], list[tuple[str, ...]]] = defaultdict(list)
    for key, identifiers in groups.items():
        label = tuple(sorted({episodes[identifier].success for identifier in identifiers}))
        strata[label].append(key)
    if len(strata) > train_count:
        raise DomainValidationError(
            "success-stratified split has too few training groups for all outcome strata"
        )
    rng = random.Random(seed)
    training = []
    remainder: dict[tuple[bool, ...], list[tuple[str, ...]]] = {}
    for label, values in strata.items():
        shuffled = sorted(values)
        rng.shuffle(shuffled)
        training.append(shuffled.pop())
        remainder[label] = shuffled
    ordered = _interleave_strata(remainder, rng)
    testing = ordered[:test_count]
    validation = ordered[test_count : test_count + validation_count]
    training.extend(ordered[test_count + validation_count :])
    if not all((testing, validation, training)):
        raise DomainValidationError("stratified split produced an empty partition")
    return {"test": testing, "validation": validation, "train": training}


class DatasetSnapshotAdapter:
    descriptor = builtin_method(
        MethodCapability.DATASET_SNAPSHOT,
        "episode_split",
        version="2.0.0",
        input_contract="DatasetBundle/1",
        output_contract="DatasetSnapshot/1+SnapshotManifest/1",
    )

    def probe(self) -> MethodProbe:
        return MethodProbe(True, "Episode-group split and immutable snapshots available")

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        import random

        config = SplitConfig.model_validate(spec.method.config)
        source = input_artifact(spec, "dataset", artifacts)
        dataset = dataset_bundle(artifacts, source)
        validate_dataset(artifacts, dataset)
        groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
        episodes: dict[str, Episode] = {}
        for summary in dataset.episodes:
            episode = load_episode(artifacts, summary)
            episodes[episode.episode_id] = episode
            key = episode_group(episode)
            groups[key].append(episode.episode_id)
        keys = sorted(groups)
        if len(keys) < 3:
            raise DomainValidationError(
                "train/validation/test require at least three independent episode groups"
            )
        test_count = max(1, int(len(keys) * config.test_fraction))
        validation_count = max(1, int(len(keys) * config.validation_fraction))
        if test_count + validation_count >= len(keys):
            raise DomainValidationError("split leaves no training groups")
        if config.stratify_success:
            partitions = _stratified_partitions(
                groups,
                episodes,
                seed=config.seed,
                test_count=test_count,
                validation_count=validation_count,
            )
        else:
            random.Random(config.seed).shuffle(keys)
            partitions = {
                "test": keys[:test_count],
                "validation": keys[test_count : test_count + validation_count],
                "train": keys[test_count + validation_count :],
            }
        membership = {
            name: [identifier for key in values for identifier in groups[key]]
            for name, values in partitions.items()
        }
        manifest = artifacts.put_json(dataset)
        code, environment = capture_runtime(artifacts)
        snapshot = DatasetSnapshot(
            dataset=dataset.reference(manifest.artifact_id),
            manifest=manifest,
            episodes=tuple(
                EpisodeArtifact(episode_id=row.episode_id, artifact=row.artifact)
                for row in dataset.episodes
                if row.artifact is not None
            ),
            code=code,
            environment=environment,
            seed=config.seed,
            split=DatasetSplit(
                train=tuple(membership["train"]),
                validation=tuple(membership["validation"]),
                test=tuple(membership["test"]),
            ),
        )
        report = {
            "schema": "physrealx.snapshot-manifest/1",
            "status": "PASSED",
            "split_sizes": {name: len(ids) for name, ids in membership.items()},
            "independent_groups": len(keys),
            "grouping": "capture_or_scene_task_seed",
            "stratified_by": "success" if config.stratify_success else None,
            "success_counts": {
                name: sum(episodes[identifier].success for identifier in identifiers)
                for name, identifiers in membership.items()
            },
            "episode_leakage": False,
            "group_leakage": False,
        }
        return publish(spec, artifacts, {"snapshot": snapshot, "snapshot_manifest": report})
