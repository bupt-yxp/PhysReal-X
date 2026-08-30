from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("scipy", reason="MuJoCo SysID runs in its isolated worker environment")

PROJECT_ROOT = Path(__file__).parents[2]
PROJECT_SOURCE = PROJECT_ROOT / "src"
WORKERS_SOURCE = PROJECT_ROOT / "workers" / "src"
sys.path.insert(0, str(PROJECT_SOURCE))
sys.path.insert(0, str(WORKERS_SOURCE))

from physrealx_workers.mujoco_sysid_worker import _handle  # noqa: E402

from physrealx.domain import (  # noqa: E402
    ArtifactRef,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodRuntime,
    OperationInput,
    OperationOutputRequirement,
    OperationSpec,
    ResourceType,
    VersionedRef,
    canonical_json_bytes,
    sha256_digest,
)
from physrealx.external_workers import (  # noqa: E402
    WorkerInputFile,
    WorkerRequest,
    WorkerResourceArtifact,
)


def _signals(*, unexcited: bool = False) -> dict[str, np.ndarray[Any, Any]]:
    timestamp = np.linspace(0.0, 40.0, 2_001, dtype=np.float64)
    if unexcited:
        velocity = np.zeros_like(timestamp)
        acceleration = np.zeros_like(timestamp)
    else:
        velocity = 0.7 * np.sin(0.73 * timestamp) + 0.35 * np.sin(1.91 * timestamp + 0.4)
        acceleration = 0.7 * 0.73 * np.cos(0.73 * timestamp) + 0.35 * 1.91 * np.cos(
            1.91 * timestamp + 0.4
        )
    position = np.cumsum(velocity) * (timestamp[1] - timestamp[0])
    applied_effort = 2.3 * acceleration + 0.7 * velocity + 0.24 * np.tanh(velocity / 0.02)
    split = np.zeros(timestamp.shape, dtype=np.int8)
    split[1_501:] = 1
    return {
        "timestamp_s": timestamp,
        "q_m": position,
        "qdot_m_s": velocity,
        "effort_n": applied_effort,
        "split": split,
    }


def _manifest(artifact: ArtifactRef, *, include_effort_provenance: bool = True) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "schema": "physrealx.joint-system-identification-evidence/1",
        "resource_id": "synthetic.drawer.sysid",
        "version": "1.0.0",
        "joint": {"name": "drawer.slide", "type": "slide"},
        "format": "npz",
        "artifact": artifact.model_dump(mode="json"),
        "signals": {
            "timestamp": "timestamp_s",
            "position": "q_m",
            "velocity": "qdot_m_s",
            "applied_effort": "effort_n",
            "split": "split",
        },
        "units": {
            "timestamp": "s",
            "position": "m",
            "velocity": "m/s",
            "applied_effort": "N",
        },
        "effort_sign_convention": "positive_assists_joint_motion",
        "split": {"train": 0, "holdout": 1},
        "prior": {
            "source_type": "engineering_baseline",
            "source": "fixture_baseline_v1",
            "parameters": {
                "effective_inertia": 0.8,
                "viscous_damping": 0.1,
                "coulomb_friction": 0.05,
            },
        },
        "provenance": {
            "capture_ids": ["fixture-excitation-001", "fixture-holdout-001"],
            "clock": "fixture-monotonic-clock",
        },
        "split_provenance": {
            "train_capture_ids": ["fixture-excitation-001"],
            "holdout_capture_ids": ["fixture-holdout-001"],
        },
    }
    if include_effort_provenance:
        manifest["effort_evidence"] = {
            "kind": "measured_actuator_effort",
            "source": "calibrated-joint-torque-sensor",
        }
    return manifest


def _request(
    tmp_path: Path,
    signals: dict[str, np.ndarray[Any, Any]],
    *,
    include_effort_provenance: bool = True,
    config: dict[str, Any] | None = None,
) -> WorkerRequest:
    inputs = tmp_path / "inputs"
    dependencies = inputs / "000-evidence.artifacts"
    dependencies.mkdir(parents=True)
    evidence_path = dependencies / "000000.npz"
    np.savez(evidence_path, **signals)
    evidence_payload = evidence_path.read_bytes()
    evidence_artifact = ArtifactRef(
        digest=sha256_digest(evidence_payload),
        size_bytes=len(evidence_payload),
        media_type="application/x-npz",
    )
    manifest = _manifest(
        evidence_artifact,
        include_effort_provenance=include_effort_provenance,
    )
    manifest_payload = canonical_json_bytes(manifest)
    manifest_path = inputs / "000-evidence"
    manifest_path.write_bytes(manifest_payload)
    manifest_digest = sha256_digest(manifest_payload)
    descriptor = MethodDescriptor(
        capability=MethodCapability.PHYSICS_CALIBRATE,
        name="physrealx_mujoco_sysid_adapter",
        version="1.0.0",
        api_version="1",
        source_uri="https://github.com/bupt-yxp/PhysReal-X",
        source_revision="fixture",
        license_spdx="MIT",
        runtime=MethodRuntime(
            mode=MethodExecutionMode.PROCESS,
            accelerator="cpu",
            python=">=3.8,<3.13",
            environment_digest=f"sha256:{'e' * 64}",
        ),
        input_contract="JointSystemIdentificationEvidence/1",
        output_contract="IdentifiedJointPhysics/1+PhysicsIdentificationReport/1",
    )
    operation = OperationSpec(
        resource_id="operation.physics.identify.fixture",
        version="1.0.0",
        method=descriptor.bind(
            config={
                "inertia_bounds": [0.1, 5.0],
                "damping_bounds": [0.0, 3.0],
                "friction_bounds": [0.0, 1.0],
                "multi_start_count": 9,
                **(config or {}),
            }
        ),
        inputs=(
            OperationInput(
                name="evidence",
                resource=VersionedRef(
                    resource_type=ResourceType.DATASET,
                    resource_id="synthetic.drawer.sysid",
                    version="1.0.0",
                    digest=manifest_digest,
                ),
            ),
        ),
        outputs=(
            OperationOutputRequirement(
                name="identified_joint_physics",
                contract="IdentifiedJointPhysics/1",
            ),
            OperationOutputRequirement(
                name="identification_report",
                contract="PhysicsIdentificationReport/1",
            ),
        ),
    )
    return WorkerRequest(
        operation=operation,
        inputs=(
            WorkerInputFile(
                name="evidence",
                kind="resource",
                digest=manifest_digest,
                size_bytes=len(manifest_payload),
                media_type="application/vnd.physrealx.sysid-evidence+json",
                path=manifest_path.relative_to(tmp_path).as_posix(),
                resource_type=ResourceType.DATASET,
            ),
        ),
        resource_artifacts=(
            WorkerResourceArtifact(
                owner="evidence",
                json_pointer="/artifact",
                digest=evidence_artifact.digest,
                size_bytes=evidence_artifact.size_bytes,
                media_type=evidence_artifact.media_type,
                path=evidence_path.relative_to(tmp_path).as_posix(),
            ),
        ),
    )


def _execute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    request: WorkerRequest,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    output = tmp_path / "outputs"
    output.mkdir()
    monkeypatch.chdir(tmp_path)
    response = _handle(request, output)
    physics = json.loads((output / "identified_joint_physics.json").read_text(encoding="utf-8"))
    report = json.loads((output / "identification_report.json").read_text(encoding="utf-8"))
    return physics, report, dict(response.metrics)


def test_adapter_recovers_bounded_parameters_and_beats_prior_on_holdout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physics, report, metrics = _execute(
        tmp_path,
        monkeypatch,
        _request(tmp_path, _signals()),
    )

    assert physics["status"] == "IDENTIFIED"
    assert physics["adapter"] == "physrealx_mujoco_sysid_adapter"
    assert physics["upstream_attribution"] == "not_an_upstream_mujoco_component"
    assert physics["parameters"]["effective_inertia"]["value"] == pytest.approx(2.3, rel=0.01)
    assert physics["parameters"]["viscous_damping"]["value"] == pytest.approx(0.7, rel=0.02)
    assert physics["parameters"]["coulomb_friction"]["value"] == pytest.approx(0.24, rel=0.02)
    for parameter in physics["parameters"].values():
        confidence = parameter["confidence_interval"]
        assert confidence["lower"] <= parameter["value"] <= confidence["upper"]
        assert parameter["multi_start_spread"] >= 0.0
    assert metrics["identifiable"] == 1.0
    assert metrics["holdout_rmse"] < metrics["prior_holdout_rmse"]
    assert metrics["prior_improvement_fraction"] > 0.9
    assert report["diagnostics"]["design_rank"] == 3
    assert report["diagnostics"]["converged_starts"] >= 3


def test_adapter_rejects_unexcited_data_instead_of_inventing_physics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physics, report, metrics = _execute(
        tmp_path,
        monkeypatch,
        _request(tmp_path, _signals(unexcited=True)),
    )

    assert physics["status"] == "NOT_IDENTIFIABLE"
    assert "parameters" not in physics
    assert report["status"] == "NOT_IDENTIFIABLE"
    assert metrics["identifiable"] == 0.0
    assert any("velocity" in blocker for blocker in report["blockers"])
    assert any("rank deficient" in blocker for blocker in report["blockers"])


def test_adapter_rejects_missing_measured_effort_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    physics, report, metrics = _execute(
        tmp_path,
        monkeypatch,
        _request(tmp_path, _signals(), include_effort_provenance=False),
    )

    assert physics["status"] == "NOT_IDENTIFIABLE"
    assert metrics["identifiable"] == 0.0
    assert report["blockers"] == [
        "effort_evidence is required; controller commands alone are insufficient"
    ]


def test_adapter_rejects_nonfinite_signal_and_emits_strict_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signals = _signals()
    signals["effort_n"][300] = np.nan
    physics, report, metrics = _execute(
        tmp_path,
        monkeypatch,
        _request(tmp_path, signals),
    )

    assert physics["status"] == "NOT_IDENTIFIABLE"
    assert metrics["identifiable"] == 0.0
    assert "NaN or infinity" in report["blockers"][0]
    json.dumps({"physics": physics, "report": report}, allow_nan=False)


def test_adapter_requires_explicit_holdout_improvement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _request(
        tmp_path,
        _signals(),
        config={"minimum_prior_improvement": 0.999999},
    )
    manifest_path = tmp_path / request.inputs[0].path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prior"]["parameters"] = {
        "effective_inertia": 2.3,
        "viscous_damping": 0.7,
        "coulomb_friction": 0.24,
    }
    payload = canonical_json_bytes(manifest)
    manifest_path.write_bytes(payload)
    request = request.model_copy(
        update={
            "inputs": (
                request.inputs[0].model_copy(
                    update={"digest": sha256_digest(payload), "size_bytes": len(payload)}
                ),
            ),
            "operation": request.operation.model_copy(
                update={
                    "inputs": (
                        request.operation.inputs[0].model_copy(
                            update={
                                "resource": request.operation.inputs[0].resource.model_copy(
                                    update={"digest": sha256_digest(payload)}
                                )
                            }
                        ),
                    )
                }
            ),
        }
    )

    physics, report, _metrics = _execute(tmp_path, monkeypatch, request)

    assert physics["status"] == "NOT_IDENTIFIABLE"
    assert report["blockers"] == [
        "identified parameters do not improve the declared prior on the holdout split"
    ]
