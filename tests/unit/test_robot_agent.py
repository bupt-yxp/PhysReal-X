from __future__ import annotations

import pytest
from pydantic import ValidationError

from physrealx.domain import ArtifactRef, ResourceType, VersionedRef
from physrealx.errors import InvalidStateTransitionError
from physrealx.robot_agent import (
    CommandDisposition,
    DeploymentMode,
    DeploymentPlan,
    JointSafetyLimit,
    MockNodeLocalRobotAgent,
    RobotCommand,
    RobotExecutionEvidence,
    RobotInterfaceSpec,
    RobotObservation,
    SafetyFault,
    SafetyPolicy,
    TelemetryManifest,
    WorkspaceBounds,
)


def _artifact(fill: str, media_type: str = "application/octet-stream") -> ArtifactRef:
    return ArtifactRef(digest=f"sha256:{fill * 64}", size_bytes=64, media_type=media_type)


def _plan(
    *,
    shadow_min_cycles: int = 2,
    canary_min_cycles: int = 2,
    heartbeat_timeout_s: float = 0.05,
    lease_ttl_s: float = 0.20,
) -> DeploymentPlan:
    robot = RobotInterfaceSpec(
        robot_id="cabinet-arm",
        interface_version="1.0.0",
        joint_names=("shoulder", "elbow"),
        base_frame="robot/base_link",
        tool_frame="robot/tool0",
        urdf=_artifact("a", "application/xml"),
        controller_configuration=_artifact("b", "application/yaml"),
        state_interface="/physrealx/joint_states",
        command_interface="/physrealx/joint_commands",
    )
    safety = SafetyPolicy(
        joints=(
            JointSafetyLimit(
                joint="shoulder",
                minimum_position_rad=-1.5,
                maximum_position_rad=1.5,
                maximum_velocity_rad_s=2.0,
                maximum_effort_nm=20.0,
            ),
            JointSafetyLimit(
                joint="elbow",
                minimum_position_rad=-2.0,
                maximum_position_rad=2.0,
                maximum_velocity_rad_s=2.5,
                maximum_effort_nm=15.0,
            ),
        ),
        workspace=WorkspaceBounds(
            minimum_m=(-0.5, -0.5, 0.0),
            maximum_m=(0.8, 0.5, 1.2),
        ),
        maximum_tcp_force_n=40.0,
        control_period_s=0.010,
        control_jitter_s=0.002,
        heartbeat_timeout_s=heartbeat_timeout_s,
        lease_ttl_s=lease_ttl_s,
        maximum_observation_age_s=0.020,
        canary_velocity_scale=0.5,
    )
    return DeploymentPlan(
        deployment_id="cabinet-policy",
        version="1.0.0",
        model=VersionedRef(
            resource_type=ResourceType.MODEL,
            resource_id="drawer-policy",
            version="1.0.0",
            digest=f"sha256:{'c' * 64}",
        ),
        robot=robot,
        safety=safety,
        shadow_min_cycles=shadow_min_cycles,
        canary_min_cycles=canary_min_cycles,
    )


def _observation(sequence: int, now_ns: int, *, estop: bool = False) -> RobotObservation:
    return RobotObservation(
        sequence=sequence,
        observed_at_ns=now_ns,
        joint_position_rad=(0.1, -0.2),
        joint_velocity_rad_s=(0.1, 0.1),
        joint_effort_nm=(1.0, 1.0),
        tcp_position_m=(0.3, 0.0, 0.4),
        tcp_force_n=(1.0, 2.0, 1.0),
        emergency_stop=estop,
    )


def _command(sequence: int) -> RobotCommand:
    return RobotCommand(
        sequence=sequence,
        joint_position_rad=(0.2, -0.1),
        joint_velocity_rad_s=(0.2, 0.2),
        joint_effort_nm=(2.0, 2.0),
        tcp_target_m=(0.35, 0.0, 0.45),
    )


def test_plan_pins_robot_model_and_complete_joint_safety_contract() -> None:
    plan = _plan()

    assert plan.schema_name == "physrealx.deployment-plan/1"
    assert plan.model_dump(mode="json", by_alias=True)["schema"] == plan.schema_name
    assert plan.robot.schema_name == "physrealx.robot-interface-spec/1"
    assert plan.safety.schema_name == "physrealx.robot-safety-policy/1"
    assert plan.robot.requires_ros2
    assert plan.robot.requires_hardware
    assert plan.model.resolved
    assert plan.digest.startswith("sha256:")
    assert plan.digest == _plan().digest

    payload = plan.model_dump(round_trip=True)
    payload["safety"]["joints"] = tuple(reversed(payload["safety"]["joints"]))
    with pytest.raises(ValidationError, match="exactly match robot joints"):
        DeploymentPlan.model_validate(payload)


def test_shadow_canary_active_never_forwards_a_shadow_command() -> None:
    plan = _plan()
    agent = MockNodeLocalRobotAgent()
    start = 1_000_000_000
    agent.acquire(plan, lease_id="lease-one", now_ns=start)

    first = agent.cycle("lease-one", _observation(1, start), _command(1), now_ns=start)
    second_time = start + 10_000_000
    second = agent.cycle(
        "lease-one",
        _observation(2, second_time),
        _command(2),
        now_ns=second_time,
    )
    assert first.disposition is CommandDisposition.OBSERVE_ONLY
    assert second.disposition is CommandDisposition.OBSERVE_ONLY
    assert agent.promote("lease-one", DeploymentMode.CANARY, now_ns=second_time) is (
        DeploymentMode.CANARY
    )

    third_time = start + 20_000_000
    fourth_time = start + 30_000_000
    third = agent.cycle("lease-one", _observation(3, third_time), _command(3), now_ns=third_time)
    fourth = agent.cycle("lease-one", _observation(4, fourth_time), _command(4), now_ns=fourth_time)
    assert third.disposition is CommandDisposition.FORWARD_LIMITED
    assert fourth.disposition is CommandDisposition.FORWARD_LIMITED
    assert agent.promote("lease-one", DeploymentMode.ACTIVE, now_ns=fourth_time) is (
        DeploymentMode.ACTIVE
    )

    fifth_time = start + 40_000_000
    fifth = agent.cycle("lease-one", _observation(5, fifth_time), _command(5), now_ns=fifth_time)
    assert fifth.accepted
    assert fifth.disposition is CommandDisposition.FORWARD


def test_promotion_requires_successful_stage_evidence() -> None:
    agent = MockNodeLocalRobotAgent()
    agent.acquire(_plan(), lease_id="lease-one", now_ns=0)

    with pytest.raises(InvalidStateTransitionError, match="shadow evidence"):
        agent.promote("lease-one", DeploymentMode.CANARY, now_ns=0)
    with pytest.raises(InvalidStateTransitionError, match="illegal deployment transition"):
        agent.promote("lease-one", DeploymentMode.ACTIVE, now_ns=0)


def test_heartbeat_timeout_latches_safe_stop() -> None:
    agent = MockNodeLocalRobotAgent()
    agent.acquire(_plan(), lease_id="lease-one", now_ns=0)

    decision = agent.cycle(
        "lease-one",
        _observation(1, 60_000_000),
        _command(1),
        now_ns=60_000_000,
    )

    assert not decision.accepted
    assert decision.fault is SafetyFault.HEARTBEAT_TIMEOUT
    assert decision.disposition is CommandDisposition.SAFE_STOP
    assert agent.mode is DeploymentMode.FAULTED
    with pytest.raises(InvalidStateTransitionError, match="no live lease"):
        agent.heartbeat("lease-one", now_ns=61_000_000)


@pytest.mark.parametrize(
    ("observation", "command", "expected"),
    [
        (
            _observation(1, 0).model_copy(update={"emergency_stop": True}),
            _command(1),
            SafetyFault.EMERGENCY_STOP,
        ),
        (
            _observation(1, 0),
            _command(1).model_copy(update={"joint_position_rad": (9.0, 0.0)}),
            SafetyFault.JOINT_POSITION_LIMIT,
        ),
        (
            _observation(1, 0).model_copy(update={"tcp_force_n": (100.0, 0.0, 0.0)}),
            _command(1),
            SafetyFault.TCP_FORCE_LIMIT,
        ),
    ],
)
def test_estop_and_hard_limits_are_fail_closed(
    observation: RobotObservation,
    command: RobotCommand,
    expected: SafetyFault,
) -> None:
    agent = MockNodeLocalRobotAgent()
    agent.acquire(_plan(), lease_id="lease-one", now_ns=0)

    decision = agent.cycle("lease-one", observation, command, now_ns=0)

    assert decision.fault is expected
    assert decision.disposition is CommandDisposition.SAFE_STOP
    assert agent.mode is DeploymentMode.FAULTED


def test_nan_wire_value_and_stale_observation_are_fail_closed() -> None:
    command = _command(1).model_dump(mode="python")
    command["joint_velocity_rad_s"] = (float("nan"), 0.0)
    agent = MockNodeLocalRobotAgent()
    agent.acquire(_plan(), lease_id="lease-one", now_ns=0)

    nan_decision = agent.cycle("lease-one", _observation(1, 0), command, now_ns=0)

    assert nan_decision.fault is SafetyFault.NON_FINITE_VALUE
    assert agent.mode is DeploymentMode.FAULTED

    stale_agent = MockNodeLocalRobotAgent()
    stale_agent.acquire(_plan(), lease_id="lease-two", now_ns=0)
    stale = stale_agent.cycle("lease-two", _observation(1, 0), _command(1), now_ns=21_000_000)
    assert stale.fault is SafetyFault.STALE_OBSERVATION
    assert stale_agent.mode is DeploymentMode.FAULTED


def test_control_watchdog_and_replayed_sequence_latch_fault() -> None:
    agent = MockNodeLocalRobotAgent()
    agent.acquire(_plan(), lease_id="lease-one", now_ns=0)
    assert agent.cycle("lease-one", _observation(1, 0), _command(1), now_ns=0).accepted

    too_late = 13_000_000
    decision = agent.cycle("lease-one", _observation(2, too_late), _command(2), now_ns=too_late)
    assert decision.fault is SafetyFault.CONTROL_WATCHDOG

    replay_agent = MockNodeLocalRobotAgent()
    replay_agent.acquire(_plan(), lease_id="lease-two", now_ns=0)
    assert replay_agent.cycle("lease-two", _observation(1, 0), _command(1), now_ns=0).accepted
    replay_time = 10_000_000
    replay = replay_agent.cycle(
        "lease-two",
        _observation(1, replay_time),
        _command(2),
        now_ns=replay_time,
    )
    assert replay.fault is SafetyFault.REPLAYED_OBSERVATION


def test_revoke_is_terminal_and_execution_evidence_only_accepts_artifacts() -> None:
    agent = MockNodeLocalRobotAgent()
    agent.acquire(_plan(), lease_id="lease-one", now_ns=0)
    decision = agent.revoke("lease-one", now_ns=1)
    assert decision.fault is SafetyFault.LEASE_REVOKED
    assert agent.mode is DeploymentMode.REVOKED

    evidence = RobotExecutionEvidence(
        status="COMPLETED",
        deployment_plan=_artifact("d", "application/json"),
        telemetry=_artifact("e", "application/vnd.mcap"),
        safety_events=_artifact("f", "application/jsonl"),
        final_mode=DeploymentMode.REVOKED,
        started_at_ns=0,
        completed_at_ns=1,
    )
    with pytest.raises(ValidationError):
        evidence.telemetry = _artifact("0")  # type: ignore[misc]
    with pytest.raises(ValidationError, match="only be finalized"):
        RobotExecutionEvidence(
            status="COMPLETED",
            deployment_plan=_artifact("d"),
            telemetry=_artifact("e"),
            safety_events=_artifact("f"),
            final_mode=DeploymentMode.ACTIVE,
            started_at_ns=0,
            completed_at_ns=1,
        )

    with pytest.raises(ValidationError, match="FAILED after a fault"):
        RobotExecutionEvidence(
            status="COMPLETED",
            deployment_plan=_artifact("d"),
            telemetry=_artifact("e"),
            safety_events=_artifact("f"),
            final_mode=DeploymentMode.FAULTED,
            started_at_ns=0,
            completed_at_ns=1,
        )


def test_telemetry_manifest_requires_complete_indexed_mcap_evidence() -> None:
    manifest = TelemetryManifest(
        deployment_digest=_plan().digest,
        robot_id="cabinet-arm",
        clock_domain="ros_time",
        started_at_ns=1,
        completed_at_ns=2,
        message_count=10,
        topics=("/joint_states", "/tf"),
        mcap=_artifact("1", "application/vnd.mcap"),
        topic_schemas=_artifact("2", "application/json"),
        crc_verified=True,
        index_verified=True,
        timestamps_monotonic=True,
    )

    assert manifest.schema_name == "physrealx.telemetry-manifest/1"
    assert manifest.status == "COMPLETE"
    assert manifest.model_dump(mode="json", by_alias=True)["schema"] == manifest.schema_name
    with pytest.raises(ValidationError, match="absolute"):
        manifest.model_copy(update={"topics": ("joint_states",)})
