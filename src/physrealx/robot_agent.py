"""Fail-closed protocol for a node-local robot deployment agent.

This module deliberately contains no ROS 2 imports.  It defines the immutable
wire contracts and the safety state machine that a real ``ros2_control`` agent
must host next to the robot.  Consequently, passing these checks is evidence of
protocol behaviour, not evidence that a robot, URDF, controller or ROS 2 runtime
exists on the current machine.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from enum import Enum
from typing import Any, Literal

from pydantic import ConfigDict, Field, ValidationError, field_validator, model_validator

from physrealx.domain import (
    ArtifactRef,
    DomainModel,
    ResourceType,
    VersionedRef,
    canonical_json_bytes,
    sha256_digest,
)
from physrealx.errors import InvalidStateTransitionError

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_FRAME = re.compile(r"^[A-Za-z][A-Za-z0-9_./:-]{0,126}$")
_SEMVER = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


class _StrEnum(str, Enum):
    """Python 3.8-compatible string enumeration."""

    def __str__(self) -> str:
        return str.__str__(self)


class RobotAgentModel(DomainModel):
    """Strict immutable base for robot-agent protocol messages."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
        strict=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class WorkspaceBounds(RobotAgentModel):
    """Axis-aligned tool-centre-point safety envelope in the base frame."""

    minimum_m: tuple[float, float, float]
    maximum_m: tuple[float, float, float]

    @model_validator(mode="after")
    def validate_bounds(self) -> WorkspaceBounds:
        values = self.minimum_m + self.maximum_m
        if any(not math.isfinite(value) for value in values):
            raise ValueError("workspace bounds must be finite")
        if any(low >= high for low, high in zip(self.minimum_m, self.maximum_m)):
            raise ValueError("workspace minimum must be below maximum on every axis")
        return self

    def contains(self, point_m: tuple[float, float, float]) -> bool:
        return all(
            low <= value <= high
            for value, low, high in zip(point_m, self.minimum_m, self.maximum_m)
        )


class JointSafetyLimit(RobotAgentModel):
    """Hard limits for one actuated joint."""

    joint: str
    minimum_position_rad: float
    maximum_position_rad: float
    maximum_velocity_rad_s: float = Field(gt=0)
    maximum_effort_nm: float = Field(gt=0)

    @field_validator("joint")
    @classmethod
    def validate_joint(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("joint must be a lowercase machine identifier")
        return value

    @model_validator(mode="after")
    def validate_limits(self) -> JointSafetyLimit:
        values = (
            self.minimum_position_rad,
            self.maximum_position_rad,
            self.maximum_velocity_rad_s,
            self.maximum_effort_nm,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("joint limits must be finite")
        if self.minimum_position_rad >= self.maximum_position_rad:
            raise ValueError("joint minimum position must be below maximum position")
        return self


class RobotInterfaceSpec(RobotAgentModel):
    """Pinned robot interface required by a real node-local ROS 2 agent."""

    schema_name: Literal["physrealx.robot-interface-spec/1"] = Field(
        default="physrealx.robot-interface-spec/1",
        alias="schema",
        serialization_alias="schema",
    )
    status: Literal["DECLARED"] = "DECLARED"
    schema_version: Literal[1] = 1
    robot_id: str
    interface_version: str
    joint_names: tuple[str, ...]
    base_frame: str
    tool_frame: str
    urdf: ArtifactRef
    controller_configuration: ArtifactRef
    state_interface: str = Field(min_length=1, max_length=255)
    command_interface: str = Field(min_length=1, max_length=255)
    requires_ros2: bool = True
    requires_hardware: bool = True

    @field_validator("robot_id")
    @classmethod
    def validate_robot_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("robot_id must be a lowercase machine identifier")
        return value

    @field_validator("interface_version")
    @classmethod
    def validate_interface_version(cls, value: str) -> str:
        if _SEMVER.fullmatch(value) is None:
            raise ValueError("interface_version must be an explicit semantic version")
        return value

    @field_validator("joint_names")
    @classmethod
    def validate_joint_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("robot interface requires at least one joint")
        if len(value) != len(set(value)):
            raise ValueError("robot joint names must be unique")
        if any(_IDENTIFIER.fullmatch(item) is None for item in value):
            raise ValueError("robot joint names must be lowercase machine identifiers")
        return value

    @field_validator("base_frame", "tool_frame")
    @classmethod
    def validate_frame(cls, value: str) -> str:
        if _FRAME.fullmatch(value) is None or ".." in value:
            raise ValueError("robot frames must be traversal-free logical frame names")
        return value

    @model_validator(mode="after")
    def validate_artifacts(self) -> RobotInterfaceSpec:
        if self.urdf.digest == self.controller_configuration.digest:
            raise ValueError("URDF and controller configuration must be distinct artifacts")
        if not self.requires_ros2 or not self.requires_hardware:
            raise ValueError("a real robot interface must explicitly require ROS 2 and hardware")
        return self


class SafetyPolicy(RobotAgentModel):
    """Complete fail-closed safety envelope for one robot interface."""

    schema_name: Literal["physrealx.robot-safety-policy/1"] = Field(
        default="physrealx.robot-safety-policy/1",
        alias="schema",
        serialization_alias="schema",
    )
    status: Literal["DECLARED"] = "DECLARED"
    schema_version: Literal[1] = 1
    joints: tuple[JointSafetyLimit, ...]
    workspace: WorkspaceBounds
    maximum_tcp_force_n: float = Field(gt=0)
    control_period_s: float = Field(gt=0)
    control_jitter_s: float = Field(ge=0)
    heartbeat_timeout_s: float = Field(gt=0)
    lease_ttl_s: float = Field(gt=0)
    maximum_observation_age_s: float = Field(gt=0)
    canary_velocity_scale: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_policy(self) -> SafetyPolicy:
        values = (
            self.maximum_tcp_force_n,
            self.control_period_s,
            self.control_jitter_s,
            self.heartbeat_timeout_s,
            self.lease_ttl_s,
            self.maximum_observation_age_s,
            self.canary_velocity_scale,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("safety policy values must be finite")
        names = tuple(item.joint for item in self.joints)
        if not names or len(names) != len(set(names)):
            raise ValueError("safety policy joint limits must be non-empty and unique")
        if self.control_jitter_s >= self.control_period_s:
            raise ValueError("control jitter must be smaller than the control period")
        if self.heartbeat_timeout_s > self.lease_ttl_s:
            raise ValueError("heartbeat timeout cannot exceed lease TTL")
        if self.maximum_observation_age_s > self.heartbeat_timeout_s:
            raise ValueError("observation age limit cannot exceed heartbeat timeout")
        if self.control_period_s + self.control_jitter_s > self.heartbeat_timeout_s:
            raise ValueError("control cycle deadline cannot exceed heartbeat timeout")
        return self


class DeploymentPlan(RobotAgentModel):
    """Immutable request to stage a model on one exact robot interface."""

    schema_name: Literal["physrealx.deployment-plan/1"] = Field(
        default="physrealx.deployment-plan/1",
        alias="schema",
        serialization_alias="schema",
    )
    status: Literal["STAGED"] = "STAGED"
    schema_version: Literal[1] = 1
    deployment_id: str
    version: str
    model: VersionedRef
    robot: RobotInterfaceSpec
    safety: SafetyPolicy
    shadow_min_cycles: int = Field(gt=0)
    canary_min_cycles: int = Field(gt=0)

    @field_validator("deployment_id")
    @classmethod
    def validate_deployment_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("deployment_id must be a lowercase machine identifier")
        return value

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        if _SEMVER.fullmatch(value) is None:
            raise ValueError("deployment version must be an explicit semantic version")
        return value

    @model_validator(mode="after")
    def validate_plan(self) -> DeploymentPlan:
        if self.model.resource_type is not ResourceType.MODEL or not self.model.resolved:
            raise ValueError("deployment model must be a resolved immutable model reference")
        policy_joints = tuple(item.joint for item in self.safety.joints)
        if policy_joints != self.robot.joint_names:
            raise ValueError("safety limits must exactly match robot joints and ordering")
        return self

    @property
    def digest(self) -> str:
        return sha256_digest(canonical_json_bytes(self))


class DeploymentMode(_StrEnum):
    IDLE = "idle"
    SHADOW = "shadow"
    CANARY = "canary"
    ACTIVE = "active"
    REVOKED = "revoked"
    FAULTED = "faulted"


class SafetyFault(_StrEnum):
    INVALID_MESSAGE = "invalid_message"
    LEASE_REVOKED = "lease_revoked"
    LEASE_EXPIRED = "lease_expired"
    HEARTBEAT_TIMEOUT = "heartbeat_timeout"
    CONTROL_WATCHDOG = "control_watchdog"
    EMERGENCY_STOP = "emergency_stop"
    STALE_OBSERVATION = "stale_observation"
    REPLAYED_OBSERVATION = "replayed_observation"
    REPLAYED_COMMAND = "replayed_command"
    NON_FINITE_VALUE = "non_finite_value"
    JOINT_POSITION_LIMIT = "joint_position_limit"
    JOINT_VELOCITY_LIMIT = "joint_velocity_limit"
    JOINT_EFFORT_LIMIT = "joint_effort_limit"
    TCP_FORCE_LIMIT = "tcp_force_limit"
    WORKSPACE_LIMIT = "workspace_limit"


class CommandDisposition(_StrEnum):
    OBSERVE_ONLY = "observe_only"
    FORWARD_LIMITED = "forward_limited"
    FORWARD = "forward"
    SAFE_STOP = "safe_stop"


class RobotObservation(RobotAgentModel):
    sequence: int = Field(ge=0)
    observed_at_ns: int = Field(ge=0)
    joint_position_rad: tuple[float, ...]
    joint_velocity_rad_s: tuple[float, ...]
    joint_effort_nm: tuple[float, ...]
    tcp_position_m: tuple[float, float, float]
    tcp_force_n: tuple[float, float, float]
    emergency_stop: bool = False


class RobotCommand(RobotAgentModel):
    sequence: int = Field(ge=0)
    joint_position_rad: tuple[float, ...]
    joint_velocity_rad_s: tuple[float, ...]
    joint_effort_nm: tuple[float, ...]
    tcp_target_m: tuple[float, float, float]


class DeploymentLease(RobotAgentModel):
    lease_id: str
    deployment_digest: str
    acquired_at_ns: int = Field(ge=0)
    expires_at_ns: int = Field(gt=0)

    @field_validator("lease_id")
    @classmethod
    def validate_lease_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("lease_id must be a lowercase machine identifier")
        return value

    @field_validator("deployment_digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("deployment digest must be a SHA-256 digest")
        return value


class SafetyDecision(RobotAgentModel):
    mode: DeploymentMode
    disposition: CommandDisposition
    accepted: bool
    fault: SafetyFault | None = None
    message: str = Field(min_length=1, max_length=512)

    @model_validator(mode="after")
    def validate_decision(self) -> SafetyDecision:
        if self.accepted == (self.fault is not None):
            raise ValueError("accepted decisions have no fault; rejected decisions require one")
        if not self.accepted and self.disposition is not CommandDisposition.SAFE_STOP:
            raise ValueError("a rejected command must request a safe stop")
        return self


class RobotExecutionEvidence(RobotAgentModel):
    """Immutable published evidence; mutable paths and live handles are forbidden."""

    schema_name: Literal["physrealx.deployment-report/1"] = Field(
        default="physrealx.deployment-report/1",
        alias="schema",
        serialization_alias="schema",
    )
    status: Literal["COMPLETED", "FAILED"]
    deployment_plan: ArtifactRef
    telemetry: ArtifactRef
    safety_events: ArtifactRef
    final_mode: DeploymentMode
    started_at_ns: int = Field(ge=0)
    completed_at_ns: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_evidence(self) -> RobotExecutionEvidence:
        if self.final_mode not in {DeploymentMode.REVOKED, DeploymentMode.FAULTED}:
            raise ValueError("execution evidence can only be finalized after revoke or fault")
        if self.completed_at_ns < self.started_at_ns:
            raise ValueError("execution completion cannot precede start")
        expected_status = "COMPLETED" if self.final_mode is DeploymentMode.REVOKED else "FAILED"
        if self.status != expected_status:
            raise ValueError(
                "execution status must be COMPLETED after revoke and FAILED after a fault"
            )
        digests = {
            self.deployment_plan.digest,
            self.telemetry.digest,
            self.safety_events.digest,
        }
        if len(digests) != 3:
            raise ValueError("deployment, telemetry and safety events must be distinct artifacts")
        return self


class TelemetryManifest(RobotAgentModel):
    """Integrity-checked ROS 2 telemetry index emitted by the node-local agent."""

    schema_name: Literal["physrealx.telemetry-manifest/1"] = Field(
        default="physrealx.telemetry-manifest/1",
        alias="schema",
        serialization_alias="schema",
    )
    status: Literal["COMPLETE"] = "COMPLETE"
    schema_version: Literal[1] = 1
    deployment_digest: str
    robot_id: str
    clock_domain: Literal["ros_time", "tai", "utc"]
    started_at_ns: int = Field(ge=0)
    completed_at_ns: int = Field(ge=0)
    message_count: int = Field(gt=0)
    topics: tuple[str, ...]
    mcap: ArtifactRef
    topic_schemas: ArtifactRef
    crc_verified: Literal[True]
    index_verified: Literal[True]
    timestamps_monotonic: Literal[True]

    @field_validator("deployment_digest")
    @classmethod
    def validate_deployment_digest(cls, value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("deployment_digest must be a SHA-256 digest")
        return value

    @field_validator("robot_id")
    @classmethod
    def validate_telemetry_robot_id(cls, value: str) -> str:
        if _IDENTIFIER.fullmatch(value) is None:
            raise ValueError("robot_id must be a lowercase machine identifier")
        return value

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(value) != len(set(value)):
            raise ValueError("telemetry topics must be non-empty and unique")
        if any(not item.startswith("/") or ".." in item for item in value):
            raise ValueError("telemetry topics must be absolute traversal-free ROS names")
        return value

    @model_validator(mode="after")
    def validate_manifest(self) -> TelemetryManifest:
        if self.completed_at_ns < self.started_at_ns:
            raise ValueError("telemetry completion cannot precede start")
        if self.mcap.media_type != "application/vnd.mcap":
            raise ValueError("telemetry payload must use application/vnd.mcap")
        if self.topic_schemas.media_type != "application/json":
            raise ValueError("telemetry topic schemas must use application/json")
        if self.mcap.digest == self.topic_schemas.digest:
            raise ValueError("telemetry payload and topic schema must be distinct artifacts")
        return self


class MockNodeLocalRobotAgent:
    """Deterministic in-memory reference for the ROS 2-side safety state machine.

    It proves protocol behaviour only.  It never opens a ROS graph, loads a
    controller or forwards a command to hardware.
    """

    def __init__(self) -> None:
        self._mode = DeploymentMode.IDLE
        self._plan: DeploymentPlan | None = None
        self._lease: DeploymentLease | None = None
        self._last_heartbeat_ns: int | None = None
        self._last_tick_ns: int | None = None
        self._last_observation_sequence: int | None = None
        self._last_command_sequence: int | None = None
        self._shadow_cycles = 0
        self._canary_cycles = 0

    @property
    def mode(self) -> DeploymentMode:
        return self._mode

    @property
    def lease(self) -> DeploymentLease | None:
        return self._lease

    def acquire(self, plan: DeploymentPlan, *, lease_id: str, now_ns: int) -> DeploymentLease:
        if self._mode is not DeploymentMode.IDLE:
            raise InvalidStateTransitionError("a robot agent can acquire only from idle")
        if now_ns < 0:
            raise ValueError("now_ns must be non-negative")
        ttl_ns = _seconds_to_ns(plan.safety.lease_ttl_s)
        lease = DeploymentLease(
            lease_id=lease_id,
            deployment_digest=plan.digest,
            acquired_at_ns=now_ns,
            expires_at_ns=now_ns + ttl_ns,
        )
        self._plan = plan
        self._lease = lease
        self._last_heartbeat_ns = now_ns
        self._mode = DeploymentMode.SHADOW
        return lease

    def heartbeat(self, lease_id: str, *, now_ns: int) -> DeploymentLease:
        plan, lease = self._require_live_lease(lease_id)
        fault = self._watchdog_fault(now_ns)
        if fault is not None:
            self._trip(fault)
            raise InvalidStateTransitionError(f"heartbeat rejected: {fault.value}")
        if self._last_heartbeat_ns is not None and now_ns < self._last_heartbeat_ns:
            self._trip(SafetyFault.HEARTBEAT_TIMEOUT)
            raise InvalidStateTransitionError("heartbeat time must be monotonic")
        ttl_ns = _seconds_to_ns(plan.safety.lease_ttl_s)
        self._last_heartbeat_ns = now_ns
        self._lease = lease.model_copy(update={"expires_at_ns": now_ns + ttl_ns})
        return self._lease

    def promote(
        self,
        lease_id: str,
        target: DeploymentMode,
        *,
        now_ns: int,
    ) -> DeploymentMode:
        plan, _ = self._require_live_lease(lease_id)
        fault = self._watchdog_fault(now_ns)
        if fault is not None:
            self._trip(fault)
            raise InvalidStateTransitionError(f"promotion rejected: {fault.value}")
        if self._mode is DeploymentMode.SHADOW and target is DeploymentMode.CANARY:
            if self._shadow_cycles < plan.shadow_min_cycles:
                raise InvalidStateTransitionError("shadow evidence threshold has not been met")
            self._mode = target
            return self._mode
        if self._mode is DeploymentMode.CANARY and target is DeploymentMode.ACTIVE:
            if self._canary_cycles < plan.canary_min_cycles:
                raise InvalidStateTransitionError("canary evidence threshold has not been met")
            self._mode = target
            return self._mode
        raise InvalidStateTransitionError(
            f"illegal deployment transition {self._mode.value}->{target.value}"
        )

    def revoke(self, lease_id: str, *, now_ns: int) -> SafetyDecision:
        self._require_live_lease(lease_id)
        if now_ns < 0:
            raise ValueError("now_ns must be non-negative")
        self._mode = DeploymentMode.REVOKED
        return SafetyDecision(
            mode=self._mode,
            disposition=CommandDisposition.SAFE_STOP,
            accepted=False,
            fault=SafetyFault.LEASE_REVOKED,
            message="lease revoked; hardware command path must remain stopped",
        )

    def cycle(
        self,
        lease_id: str,
        observation: RobotObservation | Mapping[str, Any],
        command: RobotCommand | Mapping[str, Any],
        *,
        now_ns: int,
    ) -> SafetyDecision:
        """Validate one cycle and return the only permitted command disposition."""

        if _contains_non_finite(observation) or _contains_non_finite(command):
            return self._reject(SafetyFault.NON_FINITE_VALUE, "non-finite robot value")
        try:
            plan, _ = self._require_live_lease(lease_id)
        except InvalidStateTransitionError:
            return self._reject(SafetyFault.LEASE_EXPIRED, "no live deployment lease")

        watchdog = self._watchdog_fault(now_ns)
        if watchdog is not None:
            return self._reject(watchdog, "lease or heartbeat watchdog expired")

        try:
            parsed_observation = (
                observation
                if isinstance(observation, RobotObservation)
                else RobotObservation.model_validate(observation)
            )
            parsed_command = (
                command
                if isinstance(command, RobotCommand)
                else RobotCommand.model_validate(command)
            )
        except (ValidationError, TypeError, ValueError):
            return self._reject(SafetyFault.INVALID_MESSAGE, "invalid observation or command")

        vectors = (
            parsed_observation.joint_position_rad,
            parsed_observation.joint_velocity_rad_s,
            parsed_observation.joint_effort_nm,
            parsed_observation.tcp_position_m,
            parsed_observation.tcp_force_n,
            parsed_command.joint_position_rad,
            parsed_command.joint_velocity_rad_s,
            parsed_command.joint_effort_nm,
            parsed_command.tcp_target_m,
        )
        if any(not math.isfinite(value) for values in vectors for value in values):
            return self._reject(SafetyFault.NON_FINITE_VALUE, "non-finite robot value")
        if parsed_observation.emergency_stop:
            return self._reject(SafetyFault.EMERGENCY_STOP, "hardware emergency stop is asserted")

        observation_age_ns = now_ns - parsed_observation.observed_at_ns
        maximum_age_ns = _seconds_to_ns(plan.safety.maximum_observation_age_s)
        if observation_age_ns < 0 or observation_age_ns > maximum_age_ns:
            return self._reject(SafetyFault.STALE_OBSERVATION, "observation timestamp is stale")
        if (
            self._last_observation_sequence is not None
            and parsed_observation.sequence <= self._last_observation_sequence
        ):
            return self._reject(SafetyFault.REPLAYED_OBSERVATION, "observation was replayed")
        if (
            self._last_command_sequence is not None
            and parsed_command.sequence <= self._last_command_sequence
        ):
            return self._reject(SafetyFault.REPLAYED_COMMAND, "command was replayed")

        joint_count = len(plan.robot.joint_names)
        joint_vectors = (
            parsed_observation.joint_position_rad,
            parsed_observation.joint_velocity_rad_s,
            parsed_observation.joint_effort_nm,
            parsed_command.joint_position_rad,
            parsed_command.joint_velocity_rad_s,
            parsed_command.joint_effort_nm,
        )
        if any(len(values) != joint_count for values in joint_vectors):
            return self._reject(SafetyFault.INVALID_MESSAGE, "joint vector length mismatch")

        period_fault = self._control_period_fault(now_ns, plan.safety)
        if period_fault is not None:
            return self._reject(period_fault, "control cycle missed the configured period")

        for index, limit in enumerate(plan.safety.joints):
            positions = (
                parsed_observation.joint_position_rad[index],
                parsed_command.joint_position_rad[index],
            )
            if any(
                value < limit.minimum_position_rad or value > limit.maximum_position_rad
                for value in positions
            ):
                return self._reject(SafetyFault.JOINT_POSITION_LIMIT, limit.joint)
            velocity_limit = limit.maximum_velocity_rad_s
            if self._mode is DeploymentMode.CANARY:
                velocity_limit *= plan.safety.canary_velocity_scale
            velocities = (
                parsed_observation.joint_velocity_rad_s[index],
                parsed_command.joint_velocity_rad_s[index],
            )
            if any(abs(value) > velocity_limit for value in velocities):
                return self._reject(SafetyFault.JOINT_VELOCITY_LIMIT, limit.joint)
            efforts = (
                parsed_observation.joint_effort_nm[index],
                parsed_command.joint_effort_nm[index],
            )
            if any(abs(value) > limit.maximum_effort_nm for value in efforts):
                return self._reject(SafetyFault.JOINT_EFFORT_LIMIT, limit.joint)

        if not plan.safety.workspace.contains(parsed_observation.tcp_position_m) or not (
            plan.safety.workspace.contains(parsed_command.tcp_target_m)
        ):
            return self._reject(SafetyFault.WORKSPACE_LIMIT, "TCP is outside workspace")
        force_norm = math.sqrt(sum(value * value for value in parsed_observation.tcp_force_n))
        if force_norm > plan.safety.maximum_tcp_force_n:
            return self._reject(SafetyFault.TCP_FORCE_LIMIT, "TCP force exceeds hard limit")

        self._last_tick_ns = now_ns
        self._last_observation_sequence = parsed_observation.sequence
        self._last_command_sequence = parsed_command.sequence
        if self._mode is DeploymentMode.SHADOW:
            self._shadow_cycles += 1
            disposition = CommandDisposition.OBSERVE_ONLY
        elif self._mode is DeploymentMode.CANARY:
            self._canary_cycles += 1
            disposition = CommandDisposition.FORWARD_LIMITED
        else:
            disposition = CommandDisposition.FORWARD
        return SafetyDecision(
            mode=self._mode,
            disposition=disposition,
            accepted=True,
            message="cycle accepted by fail-closed safety policy",
        )

    def _require_live_lease(self, lease_id: str) -> tuple[DeploymentPlan, DeploymentLease]:
        if self._mode in {DeploymentMode.IDLE, DeploymentMode.REVOKED, DeploymentMode.FAULTED}:
            raise InvalidStateTransitionError("robot agent has no live lease")
        if self._plan is None or self._lease is None or self._lease.lease_id != lease_id:
            raise InvalidStateTransitionError("deployment lease identity mismatch")
        return self._plan, self._lease

    def _watchdog_fault(self, now_ns: int) -> SafetyFault | None:
        if now_ns < 0 or self._plan is None or self._lease is None:
            return SafetyFault.LEASE_EXPIRED
        if now_ns > self._lease.expires_at_ns:
            return SafetyFault.LEASE_EXPIRED
        if self._last_heartbeat_ns is None:
            return SafetyFault.HEARTBEAT_TIMEOUT
        elapsed = now_ns - self._last_heartbeat_ns
        if elapsed < 0 or elapsed > _seconds_to_ns(self._plan.safety.heartbeat_timeout_s):
            return SafetyFault.HEARTBEAT_TIMEOUT
        return None

    def _control_period_fault(
        self,
        now_ns: int,
        policy: SafetyPolicy,
    ) -> SafetyFault | None:
        if self._last_tick_ns is None:
            return None
        elapsed = now_ns - self._last_tick_ns
        lower = _seconds_to_ns(policy.control_period_s - policy.control_jitter_s)
        upper = _seconds_to_ns(policy.control_period_s + policy.control_jitter_s)
        if elapsed < lower or elapsed > upper:
            return SafetyFault.CONTROL_WATCHDOG
        return None

    def _reject(self, fault: SafetyFault, message: str) -> SafetyDecision:
        self._trip(fault)
        return SafetyDecision(
            mode=self._mode,
            disposition=CommandDisposition.SAFE_STOP,
            accepted=False,
            fault=fault,
            message=message,
        )

    def _trip(self, fault: SafetyFault) -> None:
        del fault
        self._mode = DeploymentMode.FAULTED


def _seconds_to_ns(value: float) -> int:
    return round(value * 1_000_000_000)


def _contains_non_finite(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_non_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_non_finite(item) for item in value)
    return False


__all__ = [
    "CommandDisposition",
    "DeploymentLease",
    "DeploymentMode",
    "DeploymentPlan",
    "JointSafetyLimit",
    "RobotCommand",
    "RobotExecutionEvidence",
    "RobotInterfaceSpec",
    "RobotObservation",
    "SafetyDecision",
    "SafetyFault",
    "SafetyPolicy",
    "TelemetryManifest",
    "WorkspaceBounds",
]
