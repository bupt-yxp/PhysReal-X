from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from physrealx.domain import (
    ArtifactRef,
    DataValue,
    Episode,
    EpisodeSource,
    EpisodeStep,
    MethodAccelerator,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodRuntime,
    MethodSpec,
    ResourceType,
    SignalDType,
    VersionedRef,
    canonical_json_bytes,
    sha256_digest,
)
from physrealx.execution_lifecycle import (
    EpisodePackResult,
    ExecutionInput,
    ExecutionLifecycle,
    ExecutionPlan,
    ExecutionValidationReport,
    RenderedFrameEvidence,
    RenderedModality,
    RenderStreamEvidence,
    StateFrameEvidence,
    StateTrajectoryEvidence,
    VerifiedEpisodeArtifact,
    VerifiedEpisodeBundle,
    VerifiedEpisodeCollection,
)


def _digest(label: str) -> str:
    return f"sha256:{hashlib.sha256(label.encode()).hexdigest()}"


def _artifact(label: str, media_type: str = "application/octet-stream") -> ArtifactRef:
    return ArtifactRef(digest=_digest(label), size_bytes=128, media_type=media_type)


def _method(capability: MethodCapability, name: str) -> MethodSpec:
    return MethodDescriptor(
        capability=capability,
        name=name,
        version="1.0.0",
        api_version="2",
        source_uri=f"https://example.test/{name}",
        source_revision=_digest(f"revision-{name}")[7:],
        license_spdx="MIT",
        runtime=MethodRuntime(
            mode=MethodExecutionMode.CONTAINER,
            accelerator=MethodAccelerator.CPU,
            python="3.11",
            image_digest=_digest(f"image-{name}"),
        ),
        input_contract="ExecutionInput/1",
        output_contract="ExecutionOutput/1",
    ).bind()


def _plan() -> ExecutionPlan:
    return ExecutionPlan(
        run_id=str(uuid.UUID(int=1)),
        episode_id=str(uuid.UUID(int=2)),
        episode_index=3,
        scene=VersionedRef(
            resource_type=ResourceType.SCENE,
            resource_id="cabinet-scene",
            version="1.0.0",
            digest=_digest("scene"),
        ),
        task=VersionedRef(
            resource_type=ResourceType.TASK,
            resource_id="open-drawer",
            version="1.0.0",
            digest=_digest("task"),
        ),
        simulator=_method(MethodCapability.SIMULATION_RUN, "mujoco-worker"),
        policy=_method(MethodCapability.POLICY_INFERENCE, "act-worker"),
        renderer=_method(MethodCapability.RENDER_GENERATE, "blenderproc-worker"),
        packer=_method(MethodCapability.EPISODE_PACK, "lerobot-pack-worker"),
        seed=42,
        inputs=(
            ExecutionInput(
                name="asset",
                contract="TrainableAsset/1",
                resource=VersionedRef(
                    resource_type=ResourceType.ASSET,
                    resource_id="cabinet2",
                    version="1.0.0",
                    digest=_digest("cabinet2"),
                ),
            ),
            ExecutionInput(
                name="policy_checkpoint",
                contract="PolicyCheckpoint/1",
                artifact=_artifact("policy", "application/vnd.safetensors"),
            ),
        ),
        modalities=("depth", "rgb"),
    )


def _state_value(index: int) -> DataValue:
    return DataValue(
        dtype=SignalDType.FLOAT32,
        shape=(1,),
        unit="rad",
        frame="robot",
        timestamp_seconds=index / 10,
        inline=(index / 10,),
    )


def _action_value(index: int) -> DataValue:
    return DataValue(
        dtype=SignalDType.FLOAT32,
        shape=(1,),
        unit="rad",
        frame="robot",
        timestamp_seconds=index / 10,
        inline=(0.1,),
    )


def _trajectory(plan: ExecutionPlan) -> StateTrajectoryEvidence:
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    frames = tuple(
        StateFrameEvidence(
            index=index,
            timestamp_ns=index * 100_000_000,
            state={"joint.position": _state_value(index)},
            action=_action_value(index) if index < 2 else None,
        )
        for index in range(3)
    )
    return StateTrajectoryEvidence(
        plan_digest=plan.digest,
        scene=plan.scene,
        task=plan.task,
        simulator_digest=sha256_digest(plan.simulator),
        policy_digest=sha256_digest(plan.policy),
        seed=plan.seed,
        input_digest=plan.input_digest,
        trajectory=_artifact("trajectory", "application/x-mcap"),
        frames=frames,
        evidence=(_artifact("simulation-log", "text/plain"),),
        started_at=started,
        completed_at=started + timedelta(seconds=1),
    )


def _render_value(name: str, index: int) -> DataValue:
    if name == "depth":
        return DataValue(
            dtype=SignalDType.FLOAT32,
            shape=(2, 2),
            unit="m",
            frame="camera",
            timestamp_seconds=index / 10,
            artifact=_artifact(f"depth-{index}", "image/tiff"),
        )
    return DataValue(
        dtype=SignalDType.UINT8,
        shape=(2, 2, 3),
        unit="1",
        frame="camera",
        timestamp_seconds=index / 10,
        artifact=_artifact(f"rgb-{index}", "image/png"),
    )


def _rendering(plan: ExecutionPlan, trajectory: StateTrajectoryEvidence) -> RenderStreamEvidence:
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return RenderStreamEvidence(
        plan_digest=plan.digest,
        trajectory_evidence_digest=trajectory.digest,
        scene=plan.scene,
        task=plan.task,
        renderer_digest=sha256_digest(plan.renderer),
        seed=plan.seed,
        input_digest=plan.input_digest,
        stream=_artifact("render-stream", "application/x-mcap"),
        frames=tuple(
            RenderedFrameEvidence(
                index=frame.index,
                timestamp_ns=frame.timestamp_ns,
                state_frame_digest=frame.digest,
                modalities=tuple(
                    RenderedModality(name=name, value=_render_value(name, frame.index))
                    for name in plan.modalities
                ),
            )
            for frame in trajectory.frames
        ),
        evidence=(_artifact("render-log", "text/plain"),),
        started_at=started,
        completed_at=started + timedelta(seconds=1),
    )


def _episode(
    plan: ExecutionPlan,
    trajectory: StateTrajectoryEvidence,
    rendering: RenderStreamEvidence,
) -> Episode:
    steps = []
    for index, (state_frame, render_frame) in enumerate(zip(trajectory.frames, rendering.frames)):
        is_last = index == len(trajectory.frames) - 1
        steps.append(
            EpisodeStep(
                index=index,
                timestamp_seconds=state_frame.timestamp_ns / 1_000_000_000,
                observations={item.name: item.value for item in render_frame.modalities},
                state=state_frame.state,
                action=state_frame.action,
                reward=None if is_last else 0.0,
                discount=None if is_last else 1.0,
                is_first=index == 0,
                is_last=is_last,
                is_terminal=is_last,
            )
        )
    return Episode(
        episode_id=plan.episode_id,
        run_id=plan.run_id,
        index=plan.episode_index,
        source=EpisodeSource.SIMULATOR,
        seed=plan.seed,
        scene=plan.scene,
        task=plan.task,
        backend=plan.simulator,
        policy=plan.policy,
        steps=tuple(steps),
        success=True,
    )


def _packed(
    plan: ExecutionPlan,
    trajectory: StateTrajectoryEvidence,
    rendering: RenderStreamEvidence,
) -> EpisodePackResult:
    episode = _episode(plan, trajectory, rendering)
    started = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return EpisodePackResult(
        plan_digest=plan.digest,
        trajectory_evidence_digest=trajectory.digest,
        render_evidence_digest=rendering.digest,
        packer_digest=sha256_digest(plan.packer),
        input_digest=plan.input_digest,
        scene=plan.scene,
        task=plan.task,
        seed=plan.seed,
        episode_payload_digest=episode.digest,
        episode=episode,
        artifact=_artifact("episode", "application/x-mcap"),
        evidence=(_artifact("pack-log", "text/plain"),),
        started_at=started,
        completed_at=started + timedelta(seconds=1),
    )


def _chain() -> tuple[
    ExecutionPlan,
    StateTrajectoryEvidence,
    RenderStreamEvidence,
    EpisodePackResult,
]:
    plan = _plan()
    trajectory = _trajectory(plan)
    rendering = _rendering(plan, trajectory)
    packed = _packed(plan, trajectory, rendering)
    return plan, trajectory, rendering, packed


def _materialized(bundle: VerifiedEpisodeBundle) -> VerifiedEpisodeArtifact:
    payload = canonical_json_bytes(bundle)
    return VerifiedEpisodeArtifact(
        episode_id=bundle.episode.episode_id,
        bundle=bundle,
        artifact=ArtifactRef(
            digest=sha256_digest(payload),
            size_bytes=len(payload),
            media_type="application/json",
        ),
    )


def test_validated_chain_finalizes_a_versioned_episode_bundle() -> None:
    plan, trajectory, rendering, packed = _chain()

    bundle = ExecutionLifecycle.finalize(plan, trajectory, rendering, packed)

    assert bundle.status == "FINALIZED"
    assert bundle.schema_name == "physrealx.verified-episode-bundle/1"
    assert bundle.validation.status == "VALIDATED"
    assert bundle.episode == packed.episode
    assert bundle.validation.trajectory_evidence_digest == trajectory.digest
    assert bundle.validation.render_evidence_digest == rendering.digest
    assert bundle.validation.episode_digest == packed.episode.digest
    assert bundle.model_dump(mode="json", by_alias=True)["schema"] == bundle.schema_name


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("plan_digest", _digest("other-plan")),
        ("seed", 43),
        ("input_digest", _digest("other-input")),
        (
            "scene",
            VersionedRef(
                resource_type=ResourceType.SCENE,
                resource_id="other-scene",
                version="1.0.0",
                digest=_digest("other-scene"),
            ),
        ),
    ),
)
def test_trajectory_cannot_change_any_execution_pin(field_name: str, replacement: object) -> None:
    plan, trajectory, rendering, packed = _chain()
    changed = trajectory.model_copy(update={field_name: replacement})

    with pytest.raises(ValueError, match="trajectory changed pinned"):
        ExecutionLifecycle.finalize(plan, changed, rendering, packed)


def test_methods_are_bound_by_full_method_spec_digest() -> None:
    plan, trajectory, rendering, packed = _chain()
    changed = trajectory.model_copy(update={"simulator_digest": _digest("other-method")})

    with pytest.raises(ValueError, match="changed pinned simulator"):
        ExecutionLifecycle.finalize(plan, changed, rendering, packed)


def test_rendering_must_replay_every_state_frame_exactly() -> None:
    plan, trajectory, rendering, packed = _chain()
    changed_frame = rendering.frames[1].model_copy(
        update={"state_frame_digest": _digest("wrong-frame")}
    )
    changed = rendering.model_copy(
        update={"frames": (rendering.frames[0], changed_frame, rendering.frames[2])}
    )

    with pytest.raises(ValueError, match="exactly replay"):
        ExecutionLifecycle.finalize(plan, trajectory, changed, packed)


def test_packer_must_preserve_rendered_observations_and_simulation_state() -> None:
    plan, trajectory, rendering, packed = _chain()
    first = packed.episode.steps[0]
    changed_first = first.model_copy(
        update={
            "observations": {
                **first.observations,
                "rgb": _render_value("rgb", 99).model_copy(
                    update={"timestamp_seconds": first.timestamp_seconds}
                ),
            }
        }
    )
    changed_episode = packed.episode.model_copy(
        update={"steps": (changed_first, *packed.episode.steps[1:])}
    )
    changed_packed = packed.model_copy(
        update={
            "episode": changed_episode,
            "episode_payload_digest": changed_episode.digest,
        }
    )

    with pytest.raises(ValueError, match="observations do not match"):
        ExecutionLifecycle.finalize(plan, trajectory, rendering, changed_packed)


def test_manually_constructed_validation_report_cannot_finalize_another_chain() -> None:
    plan, trajectory, rendering, packed = _chain()
    report = ExecutionValidationReport(
        plan_digest=plan.digest,
        input_digest=plan.input_digest,
        trajectory_evidence_digest=trajectory.digest,
        render_evidence_digest=rendering.digest,
        pack_result_digest=packed.digest,
        episode_digest=_digest("different-episode"),
        checks=(
            "artifact_evidence",
            "episode_binding",
            "input_binding",
            "method_binding",
            "resource_binding",
            "timeline_alignment",
        ),
    )

    with pytest.raises(ValidationError, match="report does not match"):
        VerifiedEpisodeBundle(
            plan=plan,
            trajectory=trajectory,
            rendering=rendering,
            packed=packed,
            validation=report,
        )


def test_plan_requires_resolved_inputs_versioned_contracts_and_exact_capabilities() -> None:
    plan = _plan()
    with pytest.raises(ValidationError, match="explicitly versioned"):
        ExecutionInput(
            name="asset",
            contract="TrainableAsset",
            artifact=_artifact("asset"),
        )
    with pytest.raises(ValidationError, match="resolved"):
        ExecutionInput(
            name="asset",
            contract="TrainableAsset/1",
            resource=VersionedRef(
                resource_type=ResourceType.ASSET,
                resource_id="cabinet2",
                version="1.0.0",
            ),
        )
    with pytest.raises(ValidationError, match=r"render\.generate"):
        plan.model_copy(update={"renderer": plan.simulator})


def test_frames_require_nanosecond_aligned_timestamps_and_real_evidence() -> None:
    with pytest.raises(ValidationError, match="integer nanoseconds"):
        StateFrameEvidence(
            index=0,
            timestamp_ns=0,
            state={
                "joint.position": _state_value(0).model_copy(
                    update={"timestamp_seconds": 0.0000000001}
                )
            },
            action=None,
        )
    plan = _plan()
    trajectory = _trajectory(plan)
    with pytest.raises(ValidationError, match="at least one evidence"):
        trajectory.model_copy(update={"evidence": ()})


def test_lifecycle_exposes_validation_and_finalization_not_algorithm_execution() -> None:
    assert callable(ExecutionLifecycle.validate)
    assert callable(ExecutionLifecycle.finalize)
    assert not hasattr(ExecutionLifecycle, "run")
    assert not hasattr(ExecutionLifecycle, "render")
    assert not hasattr(ExecutionLifecycle, "pack")


def test_collect_orders_multiple_materialized_episodes_deterministically() -> None:
    first_plan, first_trajectory, first_rendering, first_packed = _chain()
    first_bundle = ExecutionLifecycle.finalize(
        first_plan, first_trajectory, first_rendering, first_packed
    )
    second_plan = first_plan.model_copy(
        update={"episode_id": str(uuid.UUID(int=3)), "episode_index": 4}
    )
    second_trajectory = _trajectory(second_plan)
    second_rendering = _rendering(second_plan, second_trajectory)
    second_packed = _packed(second_plan, second_trajectory, second_rendering)
    second_bundle = ExecutionLifecycle.finalize(
        second_plan, second_trajectory, second_rendering, second_packed
    )

    collection = ExecutionLifecycle.collect(
        (_materialized(second_bundle), _materialized(first_bundle))
    )

    assert isinstance(collection, VerifiedEpisodeCollection)
    assert collection.status == "FINALIZED"
    assert collection.schema_name == "physrealx.verified-episode-collection/1"
    assert tuple(item.episode_id for item in collection.episodes) == tuple(
        sorted((first_plan.episode_id, second_plan.episode_id))
    )
    assert collection.model_dump(mode="json", by_alias=True)["schema"] == (
        "physrealx.verified-episode-collection/1"
    )


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("digest", _digest("wrong-bundle")),
        ("size_bytes", 1),
        ("media_type", "application/octet-stream"),
    ),
)
def test_verified_episode_artifact_requires_exact_canonical_json(
    field_name: str, replacement: object
) -> None:
    plan, trajectory, rendering, packed = _chain()
    bundle = ExecutionLifecycle.finalize(plan, trajectory, rendering, packed)
    materialized = _materialized(bundle)

    with pytest.raises(ValidationError, match="verified episode artifact"):
        materialized.model_copy(
            update={"artifact": materialized.artifact.model_copy(update={field_name: replacement})}
        )


def test_collection_rejects_duplicate_episode_ids_and_direct_nondeterministic_order() -> None:
    plan, trajectory, rendering, packed = _chain()
    bundle = ExecutionLifecycle.finalize(plan, trajectory, rendering, packed)
    materialized = _materialized(bundle)

    with pytest.raises(ValidationError, match="ids must be unique"):
        VerifiedEpisodeCollection(episodes=(materialized, materialized))

    second_plan = plan.model_copy(update={"episode_id": str(uuid.UUID(int=3)), "episode_index": 4})
    second_trajectory = _trajectory(second_plan)
    second_rendering = _rendering(second_plan, second_trajectory)
    second_bundle = ExecutionLifecycle.finalize(
        second_plan,
        second_trajectory,
        second_rendering,
        _packed(second_plan, second_trajectory, second_rendering),
    )
    with pytest.raises(ValidationError, match="canonical episode-id order"):
        VerifiedEpisodeCollection(episodes=(_materialized(second_bundle), materialized))


def test_collect_requires_at_least_one_verified_episode_artifact() -> None:
    with pytest.raises(ValidationError, match="cannot be empty"):
        ExecutionLifecycle.collect(())
