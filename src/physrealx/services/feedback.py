"""Evaluation, deployment evidence and feedback refinement."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal

from physrealx.domain import (
    ArtifactRef,
    EvaluationReport,
    FailureCase,
    FailureCategory,
    JSONValue,
    MethodCapability,
    MethodSpec,
    OperationManifest,
    ResourceType,
    RunSpec,
    VersionedRef,
    canonical_record_id,
)
from physrealx.errors import (
    DomainValidationError,
    ResourceConflictError,
)

if TYPE_CHECKING:
    from physrealx.platform import PhysRealX


class EvaluationsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def submit(
        self,
        *,
        resource_id: str,
        version: str,
        model: VersionedRef,
        run: RunSpec,
        episode_count: int = 16,
        base_seed: int = 100_000,
    ) -> VersionedRef:
        if model.resource_type is not ResourceType.MODEL or not model.resolved:
            raise DomainValidationError("evaluation requires a resolved model reference")
        registered_model = self._platform.repository.get_model(model.resource_id, model.version)
        if model.artifact_id != registered_model.artifact_id:
            raise ResourceConflictError("evaluation model does not match the registry")
        model_manifest = self._platform.training.get_model(registered_model)
        task = self._platform.tasks.get(run.task)
        task_observations = {item.name: item for item in task.observations}
        incompatible = {
            item.name
            for item in model_manifest.observation_signals
            if task_observations.get(item.name) != item
        }
        if incompatible or model_manifest.action_signal != task.action.signal:
            raise DomainValidationError(
                "model observation/action schema is incompatible with the evaluation task"
            )
        evaluation_payload = run.model_dump(mode="json")
        evaluation_payload.update(
            {
                "policy": model_manifest.policy.model_copy(
                    update={
                        "config": {
                            **model_manifest.policy.config,
                            "model_artifact_id": registered_model.artifact_id or "",
                        }
                    }
                ).model_dump(mode="json"),
                "episode_count": episode_count,
                "base_seed": base_seed,
                "dataset_id": f"{resource_id}.episodes",
                "dataset_version": version,
                "tags": [
                    *(
                        tag
                        for tag in run.tags
                        if tag
                        not in {"expert_demonstration", "failure_recovery", "policy_evaluation"}
                    ),
                    "policy_evaluation",
                ],
            }
        )
        evaluation_payload.pop("run_id", None)
        evaluation_key = canonical_record_id(evaluation_payload)
        evaluation_payload["run_id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, evaluation_key))
        evaluation_run = RunSpec.model_validate(evaluation_payload)
        record = self._platform.runs.submit(evaluation_run)
        if record.result is None or record.result.dataset is None:
            raise DomainValidationError("evaluation run completed without a dataset")
        dataset = self._platform.datasets.get(record.result.dataset)
        failures = tuple(
            FailureCase(
                episode_id=item.episode_id,
                category=(
                    FailureCategory.TIMEOUT
                    if item.failure_reason == "task_timeout"
                    else FailureCategory.UNKNOWN
                ),
                message=item.failure_reason or "task failed",
            )
            for item in dataset.episodes
            if not item.success
        )
        successes = sum(item.success for item in dataset.episodes)
        report = EvaluationReport(
            resource_id=resource_id,
            version=version,
            model=registered_model,
            dataset=record.result.dataset,
            success_rate=successes / len(dataset.episodes),
            metrics={
                "success_count": float(successes),
                "episode_count": float(len(dataset.episodes)),
            },
            failures=failures,
            created_at=dataset.created_at,
        )
        return self._platform._register(
            report,
            report.reference(),
            "application/vnd.physrealx.evaluation+json",
            specialized="evaluation",
        )

    def get_report(self, reference: VersionedRef) -> EvaluationReport:
        return self._platform._load_registered(reference, EvaluationReport)

    def list_reports(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.EVALUATION)

    def run(
        self,
        model: VersionedRef | ArtifactRef,
        suite: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.EVALUATION_RUN,
            method,
            inputs={"model": model, "suite": suite},
            config=config,
        )


class DeploymentsService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def stage(
        self,
        model: VersionedRef | ArtifactRef,
        robot: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.DEPLOYMENT_STAGE,
            method,
            inputs={"model": model, "robot": robot},
            config=config,
        )


class TelemetryService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def record(
        self,
        run: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.TELEMETRY_RECORD,
            method,
            inputs={"run": run},
            config=config,
        )


class FailuresService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def analyze(
        self,
        run: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.FAILURE_ANALYZE,
            method,
            inputs={"run": run},
            config=config,
        )


class FeedbackService:
    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def preview_demonstrations(self, datasets: tuple[VersionedRef, ...]) -> ArtifactRef:
        from physrealx.integrations.feedback_preview import preview_demonstrations

        for reference in datasets:
            self._platform.datasets.get(reference)
        return preview_demonstrations(self._platform.artifacts, datasets)

    def build_dataset(
        self,
        query: VersionedRef | ArtifactRef,
        *,
        method: str | MethodSpec,
        config: Mapping[str, JSONValue] | None = None,
    ) -> OperationManifest:
        return self._platform.methods.execute(
            MethodCapability.FEEDBACK_BUILD_DATASET,
            method,
            inputs={"query": query},
            config=config,
        )

    def plan_physics_calibration(
        self,
        telemetry: ArtifactRef,
        failure_report: ArtifactRef,
    ) -> ArtifactRef:
        """Decide whether verified deployment evidence can support physics calibration."""
        from physrealx.integrations.feedback import build_physics_calibration_plan

        return build_physics_calibration_plan(
            self._platform.artifacts,
            telemetry,
            failure_report,
        )

    def analyze_policy_intervention(self, evidence: ArtifactRef) -> ArtifactRef:
        """Attribute failures using same-condition reruns that change only the policy."""
        from physrealx.integrations.feedback import analyze_policy_intervention

        return analyze_policy_intervention(self._platform.artifacts, evidence)

    def assess_policy_update(
        self,
        initial: VersionedRef,
        updated: VersionedRef,
        *,
        reference_dataset: VersionedRef | None = None,
        config: Mapping[str, JSONValue] | None = None,
    ) -> ArtifactRef:
        """Gate a candidate policy with paired closed-loop evaluation episodes."""
        from physrealx.integrations.feedback import assess_policy_update

        self._platform.evaluations.get_report(initial)
        self._platform.evaluations.get_report(updated)
        if reference_dataset is not None:
            self._platform.datasets.get(reference_dataset)
        return assess_policy_update(
            self._platform.artifacts,
            initial,
            updated,
            config,
            reference_dataset=reference_dataset,
        )

    def prepare_physics_calibration(
        self,
        telemetry: ArtifactRef,
        *,
        joint_name: str,
        joint_type: Literal["slide", "hinge"],
        joint_index: int,
        prior_parameters: tuple[float, float, float],
    ) -> ArtifactRef:
        """Prepare existing inverse-dynamics calibration from attested real episodes."""
        from physrealx.integrations.feedback import prepare_joint_identification_evidence

        return prepare_joint_identification_evidence(
            self._platform.artifacts,
            telemetry,
            joint_name=joint_name,
            joint_type=joint_type,
            joint_index=joint_index,
            prior_parameters=prior_parameters,
        )

    def preview_report(
        self,
        initial: VersionedRef,
        updated: VersionedRef,
        failure_report: ArtifactRef,
        causal_report: ArtifactRef,
        calibration_plan: ArtifactRef,
        policy_update: ArtifactRef | None = None,
    ) -> ArtifactRef:
        """Render closed-loop metrics, attribution evidence, and calibration status."""
        from physrealx.integrations.feedback_preview import preview_feedback_report

        return preview_feedback_report(
            self._platform.artifacts,
            self._platform.evaluations.get_report(initial),
            self._platform.evaluations.get_report(updated),
            failure_report,
            causal_report,
            calibration_plan,
            policy_update,
        )

    def preview_cycle(
        self,
        failed_episode: ArtifactRef,
        recovery_episode: ArtifactRef,
        updated_episode: ArtifactRef,
        *,
        frames: int = 8,
        fps: int = 3,
    ) -> ArtifactRef:
        """Animate a same-condition failure, expert recovery, and updated policy replay."""
        from physrealx.integrations.feedback_preview import preview_feedback_cycle

        return preview_feedback_cycle(
            self._platform.artifacts,
            failed_episode,
            recovery_episode,
            updated_episode,
            frames=frames,
            fps=fps,
        )
