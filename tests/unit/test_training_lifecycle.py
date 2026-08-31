from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from physrealx.domain import (
    ArtifactRef,
    FailureCategory,
    MethodAccelerator,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodRuntime,
    ResourceType,
    VersionedRef,
    canonical_record_id,
)
from physrealx.training_lifecycle import (
    BenchmarkSuite,
    CodePin,
    DatasetSnapshot,
    DatasetSplit,
    EnvironmentPin,
    EpisodeArtifact,
    EpisodeExecutionResult,
    EvaluationLifecycle,
    EvaluationOutcome,
    FailureLifecycle,
    FeedbackDatasetSpec,
    FeedbackLifecycle,
    ModelKind,
    PolicyTrainingPlan,
    SnapshotLineage,
    TrainingExecutionResult,
    TrainingLifecycle,
    WorldModelTrainingPlan,
)


def _record_id(label: str) -> str:
    return canonical_record_id(label)


def _artifact_id(label: str) -> str:
    return f"art_{uuid.uuid5(uuid.NAMESPACE_URL, label).hex}"


def _artifact(label: str, media_type: str = "application/octet-stream") -> ArtifactRef:
    return ArtifactRef(artifact_id=_artifact_id(label), size_bytes=128, media_type=media_type)


def _episode_id(index: int) -> str:
    return str(uuid.UUID(int=index))


def _method(capability: MethodCapability, name: str):  # type: ignore[no-untyped-def]
    descriptor = MethodDescriptor(
        capability=capability,
        name=name,
        version="1.0.0",
        api_version="2",
        source_uri=f"https://example.test/{name}",
        source_revision=f"revision-{name}",
        license_spdx="MIT",
        runtime=MethodRuntime(
            mode=MethodExecutionMode.IN_PROCESS,
            accelerator=MethodAccelerator.CPU,
            python="3.11",
        ),
        input_contract="input/1",
        output_contract="output/1",
    )
    return descriptor.bind()


def _code(label: str = "code") -> CodePin:
    return CodePin(
        source_uri="https://github.com/example/implementation",
        revision=f"revision-{label}",
        source_tree=_artifact(f"{label}-tree", "application/x-tar"),
    )


def _environment(label: str = "environment") -> EnvironmentPin:
    return EnvironmentPin(
        runtime="python-3.11-cpu",
        lockfile=_artifact(f"{label}-lock", "text/plain"),
        environment_record_id=_record_id(label),
    )


def _snapshot(
    *,
    version: str = "1.0.0",
    manifest_label: str = "manifest-v1",
    lineage: SnapshotLineage | None = None,
    code: CodePin | None = None,
    environment: EnvironmentPin | None = None,
    seed: int = 41,
) -> DatasetSnapshot:
    ids = tuple(_episode_id(index) for index in range(1, 5))
    return DatasetSnapshot(
        dataset=VersionedRef(
            resource_type=ResourceType.DATASET,
            resource_id="cabinet-episodes",
            version=version,
            artifact_id=_artifact_id(f"dataset-{version}-{manifest_label}"),
        ),
        manifest=_artifact(manifest_label, "application/json"),
        episodes=tuple(
            EpisodeArtifact(episode_id=episode_id, artifact=_artifact(f"episode-{index}"))
            for index, episode_id in enumerate(ids, start=1)
        ),
        code=code or _code(),
        environment=environment or _environment(),
        seed=seed,
        split=DatasetSplit(train=(ids[0],), validation=(ids[1],), test=ids[2:]),
        lineage=lineage,
    )


def _policy_plan(snapshot: DatasetSnapshot) -> PolicyTrainingPlan:
    return PolicyTrainingPlan(
        model_id="cabinet-policy",
        version="1.0.0",
        dataset=snapshot,
        trainer=_method(MethodCapability.POLICY_TRAIN, "lerobot-act-adapter"),
        inference=_method(MethodCapability.POLICY_INFERENCE, "act-runtime"),
        code=_code("policy-code"),
        environment=_environment("policy-env"),
        seed=7,
    )


def _world_plan(snapshot: DatasetSnapshot) -> WorldModelTrainingPlan:
    return WorldModelTrainingPlan(
        model_id="cabinet-world-model",
        version="1.0.0",
        dataset=snapshot,
        trainer=_method(MethodCapability.WORLD_MODEL_TRAIN, "dino-wm-adapter"),
        rollout=_method(MethodCapability.WORLD_MODEL_ROLLOUT, "dino-wm-rollout"),
        code=_code("world-code"),
        environment=_environment("world-env"),
        seed=11,
    )


def _training_result(plan: PolicyTrainingPlan | WorldModelTrainingPlan) -> TrainingExecutionResult:
    kind = ModelKind.POLICY if isinstance(plan, PolicyTrainingPlan) else ModelKind.WORLD_MODEL
    return TrainingExecutionResult(
        run_id=str(uuid.UUID(int=101)),
        kind=kind,
        model=VersionedRef(
            resource_type=ResourceType.MODEL,
            resource_id=plan.model_id,
            version=plan.version,
            artifact_id=_artifact_id(f"{plan.model_id}-manifest"),
        ),
        dataset_snapshot_id=plan.dataset.record_id,
        trainer_record_id=canonical_record_id(plan.trainer),
        code_record_id=plan.code.record_id,
        environment_record_id=plan.environment.record_id,
        split_record_id=plan.dataset.split.record_id,
        seed=plan.seed,
        checkpoint=_artifact(f"{plan.model_id}-checkpoint", "application/vnd.safetensors"),
        auxiliary_artifacts=(_artifact(f"{plan.model_id}-config", "application/json"),),
        evidence=(_artifact(f"{plan.model_id}-log", "text/plain"),),
        metrics={"validation_loss": 0.2},
        started_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        completed_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )


def _policy_bundle():  # type: ignore[no-untyped-def]
    snapshot = _snapshot()
    plan = _policy_plan(snapshot)
    return TrainingLifecycle.finalize_policy(plan, _training_result(plan))


def test_dataset_snapshot_freezes_data_code_environment_seed_and_split() -> None:
    snapshot = _snapshot()

    assert snapshot.schema_name == "physrealx.dataset-snapshot/1"
    assert snapshot.status == "VALIDATED"
    assert snapshot.model_dump(mode="json", by_alias=True)["schema"] == snapshot.schema_name
    assert snapshot.dataset.resolved
    assert snapshot.record_id.startswith("rec_")
    assert snapshot.split.record_id.startswith("rec_")
    with pytest.raises(ValidationError, match="frozen"):
        snapshot.seed = 99  # type: ignore[misc]

    payload = snapshot.model_dump(round_trip=True)
    payload["split"]["test"] = (payload["split"]["test"][0],)
    with pytest.raises(ValidationError, match="cover every snapshot episode"):
        DatasetSnapshot.model_validate(payload)


def test_snapshot_rejects_mutable_or_incomplete_reproducibility_pins() -> None:
    with pytest.raises(ValidationError, match="immutable"):
        _code().model_copy(update={"revision": "main"})
    with pytest.raises(ValidationError, match="non-empty"):
        _environment().model_copy(
            update={
                "lockfile": ArtifactRef(
                    artifact_id=_artifact_id("empty-lock"),
                    size_bytes=0,
                    media_type="text/plain",
                )
            }
        )


def test_policy_and_world_model_plans_are_strictly_separated() -> None:
    snapshot = _snapshot()
    policy = _policy_plan(snapshot)
    world = _world_plan(snapshot)

    with pytest.raises(ValidationError, match=r"policy\.train"):
        policy.model_copy(update={"trainer": world.trainer})
    with pytest.raises(ValidationError, match=r"world_model\.rollout"):
        world.model_copy(update={"rollout": policy.inference})


def test_training_lifecycle_accepts_exact_policy_and_world_model_evidence() -> None:
    snapshot = _snapshot()
    policy_plan = _policy_plan(snapshot)
    world_plan = _world_plan(snapshot)

    policy = TrainingLifecycle.finalize_policy(policy_plan, _training_result(policy_plan))
    world = TrainingLifecycle.finalize_world_model(world_plan, _training_result(world_plan))

    assert policy.schema_name == "physrealx.policy-model-bundle/1"
    assert policy.status == "TRAINED"
    assert policy.kind == "policy"
    assert policy.report.kind is ModelKind.POLICY
    assert policy.inference.capability is MethodCapability.POLICY_INFERENCE
    assert world.kind == "world_model"
    assert world.schema_name == "physrealx.world-model-bundle/1"
    assert world.status == "TRAINED"
    assert world.report.kind is ModelKind.WORLD_MODEL
    assert world.rollout.capability is MethodCapability.WORLD_MODEL_ROLLOUT


def test_training_lifecycle_rejects_any_changed_execution_pin() -> None:
    plan = _policy_plan(_snapshot())
    result = _training_result(plan)

    for field_name, replacement in (
        ("seed", plan.seed + 1),
        ("dataset_snapshot_id", _record_id("other-dataset")),
        ("trainer_record_id", _record_id("other-trainer")),
        ("environment_record_id", _record_id("other-env")),
        ("split_record_id", _record_id("other-split")),
    ):
        with pytest.raises(ValueError, match="changed pinned"):
            TrainingLifecycle.finalize_policy(
                plan,
                result.model_copy(update={field_name: replacement}),
            )


def test_benchmark_suite_is_the_exact_frozen_test_split_with_stable_seeds() -> None:
    snapshot = _snapshot()
    evaluator = _method(MethodCapability.EVALUATION_RUN, "libero-adapter")
    suite = BenchmarkSuite.from_snapshot(
        suite_id="cabinet-fixed",
        version="1.0.0",
        dataset=snapshot,
        evaluator=evaluator,
        global_seed=123,
        timeout_s=30.0,
    )
    repeated = BenchmarkSuite.from_snapshot(
        suite_id="cabinet-fixed",
        version="1.0.0",
        dataset=snapshot,
        evaluator=evaluator,
        global_seed=123,
        timeout_s=30.0,
    )

    assert tuple(item.episode_id for item in suite.episodes) == snapshot.split.test
    assert suite.schema_name == "physrealx.benchmark-suite/1"
    assert suite.status == "FROZEN"
    assert tuple(item.seed for item in suite.episodes) == tuple(
        item.seed for item in repeated.episodes
    )
    assert suite.record_id == repeated.record_id

    payload = suite.model_dump(round_trip=True)
    payload["episodes"] = tuple(reversed(payload["episodes"]))
    with pytest.raises(ValidationError, match="test split order"):
        BenchmarkSuite.model_validate(payload)


def _suite_and_results():  # type: ignore[no-untyped-def]
    snapshot = _snapshot()
    suite = BenchmarkSuite.from_snapshot(
        suite_id="cabinet-fixed",
        version="1.0.0",
        dataset=snapshot,
        evaluator=_method(MethodCapability.EVALUATION_RUN, "libero-adapter"),
        global_seed=123,
        timeout_s=10.0,
    )
    success, timeout = suite.episodes
    results = (
        EpisodeExecutionResult(
            episode_id=success.episode_id,
            seed=success.seed,
            elapsed_s=1.0,
            objective_satisfied=True,
            metrics={"return": 1.0},
            evidence=_artifact("success-rollout", "application/vnd.mcap"),
        ),
        EpisodeExecutionResult(
            episode_id=timeout.episode_id,
            seed=timeout.seed,
            elapsed_s=10.0,
            objective_satisfied=True,
            metrics={"return": 1.0},
            evidence=_artifact("timeout-log", "text/plain"),
        ),
    )
    return suite, results


def test_evaluation_timeout_is_failure_and_wilson_ci_uses_all_episodes() -> None:
    suite, results = _suite_and_results()
    report = EvaluationLifecycle.finalize(
        report_id="cabinet-policy-eval",
        version="1.0.0",
        model=_policy_bundle(),
        suite=suite,
        results=results,
    )

    assert report.success_count == 1
    assert report.schema_name == "physrealx.benchmark-report/1"
    assert report.status == "PASSED"
    assert report.failure_count == 1
    assert report.timeout_count == 1
    assert report.success_rate == 0.5
    assert report.episodes[1].outcome is EvaluationOutcome.TIMEOUT
    assert report.success_rate_ci.lower == pytest.approx(0.09453120573423074)
    assert report.success_rate_ci.upper == pytest.approx(0.9054687942657693)
    assert all(item.evidence.size_bytes > 0 for item in report.episodes)


def test_evaluation_rejects_missing_episode_duplicate_episode_and_changed_seed() -> None:
    suite, results = _suite_and_results()
    kwargs = {
        "report_id": "cabinet-policy-eval",
        "version": "1.0.0",
        "model": _policy_bundle(),
        "suite": suite,
    }
    with pytest.raises(ValueError, match="cover the fixed suite exactly"):
        EvaluationLifecycle.finalize(results=results[:1], **kwargs)
    with pytest.raises(ValueError, match="duplicate"):
        EvaluationLifecycle.finalize(results=(results[0], results[0]), **kwargs)
    changed = results[0].model_copy(update={"seed": results[0].seed + 1})
    with pytest.raises(ValueError, match="changed a fixed episode seed"):
        EvaluationLifecycle.finalize(results=(changed, results[1]), **kwargs)


def test_failure_unknown_is_preserved_and_success_cannot_be_relabelled() -> None:
    suite, results = _suite_and_results()
    report = EvaluationLifecycle.finalize(
        report_id="cabinet-policy-eval",
        version="1.0.0",
        model=_policy_bundle(),
        suite=suite,
        results=results,
    )
    timeout = report.episodes[1]
    failure = FailureLifecycle.record(
        report,
        episode_id=timeout.episode_id,
        message="deadline exceeded; subsystem cause is not yet established",
        evidence=(timeout.evidence,),
    )

    assert failure.category is FailureCategory.UNKNOWN
    assert not failure.confirmed
    failure_report = FailureLifecycle.finalize(report, (failure,))
    assert failure_report.schema_name == "physrealx.failure-report/1"
    assert failure_report.status == "ANALYZED"
    assert failure_report.records == (failure,)
    with pytest.raises(ValueError, match="successful episode"):
        FailureLifecycle.record(
            report,
            episode_id=report.episodes[0].episode_id,
            message="invalid relabel",
            evidence=(report.episodes[0].evidence,),
        )


def test_feedback_requires_new_version_and_explicit_immutable_lineage() -> None:
    base = _snapshot()
    suite, results = _suite_and_results()
    report = EvaluationLifecycle.finalize(
        report_id="cabinet-policy-eval",
        version="1.0.0",
        model=_policy_bundle(),
        suite=suite,
        results=results,
    )
    failure = FailureLifecycle.record(
        report,
        episode_id=report.episodes[1].episode_id,
        message="timeout",
        evidence=(report.episodes[1].evidence,),
    )
    method = _method(MethodCapability.FEEDBACK_BUILD_DATASET, "hard-negative-adapter")
    code = _code("feedback-code")
    environment = _environment("feedback-env")
    spec = FeedbackDatasetSpec(
        base=base,
        target=VersionedRef(
            resource_type=ResourceType.DATASET,
            resource_id=base.dataset.resource_id,
            version="1.1.0",
        ),
        failures=(failure,),
        method=method,
        code=code,
        environment=environment,
        seed=99,
    )
    produced = _snapshot(
        version="1.1.0",
        manifest_label="manifest-v2",
        lineage=SnapshotLineage(
            parent_snapshot_id=base.record_id,
            method=method,
            evidence=(_artifact("feedback-build-log", "text/plain"),),
        ),
        code=code,
        environment=environment,
        seed=99,
    )

    finalized = FeedbackLifecycle.finalize(spec, produced)

    assert finalized.schema_name == "physrealx.feedback-dataset/1"
    assert finalized.status == "VALIDATED"
    assert finalized.snapshot.dataset.version == "1.1.0"
    assert finalized.snapshot.dataset.artifact_id != base.dataset.artifact_id
    assert finalized.parent_snapshot_id == base.record_id
    assert base.dataset.version == "1.0.0"


def test_feedback_fails_closed_on_in_place_target_or_missing_parent_lineage() -> None:
    base = _snapshot()
    suite, results = _suite_and_results()
    report = EvaluationLifecycle.finalize(
        report_id="cabinet-policy-eval",
        version="1.0.0",
        model=_policy_bundle(),
        suite=suite,
        results=results,
    )
    failure = FailureLifecycle.record(
        report,
        episode_id=report.episodes[1].episode_id,
        message="timeout",
        evidence=(report.episodes[1].evidence,),
    )
    method = _method(MethodCapability.FEEDBACK_BUILD_DATASET, "hard-negative-adapter")
    with pytest.raises(ValidationError, match="new dataset version"):
        FeedbackDatasetSpec(
            base=base,
            target=VersionedRef(
                resource_type=ResourceType.DATASET,
                resource_id=base.dataset.resource_id,
                version=base.dataset.version,
            ),
            failures=(failure,),
            method=method,
            code=_code("feedback-code"),
            environment=_environment("feedback-env"),
            seed=99,
        )

    spec = FeedbackDatasetSpec(
        base=base,
        target=VersionedRef(
            resource_type=ResourceType.DATASET,
            resource_id=base.dataset.resource_id,
            version="1.1.0",
        ),
        failures=(failure,),
        method=method,
        code=_code("feedback-code"),
        environment=_environment("feedback-env"),
        seed=99,
    )
    with pytest.raises(ValueError, match="lineage"):
        FeedbackLifecycle.finalize(spec, _snapshot(version="1.1.0"))
