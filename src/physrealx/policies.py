"""Deterministic joint target policy, independent of the simulation engine."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import Field

from physrealx.contracts import Frame
from physrealx.domain import DataValue, DomainModel, JSONValue, MethodCapability, TaskSpec
from physrealx.errors import DomainValidationError, InvalidStateTransitionError
from physrealx.methods import builtin_method


class JointTargetConfig(DomainModel):
    target: tuple[float, ...] = Field(min_length=1)


class JointSequenceConfig(DomainModel):
    commands: tuple[tuple[float, ...], ...] = Field(min_length=1, max_length=10000)


class JointSequencePolicy:
    descriptor = builtin_method(
        MethodCapability.POLICY_INFERENCE,
        "joint_sequence",
        input_contract="Frame/1+TaskSpec/1",
        output_contract="Action/1",
    )

    def __init__(self, config: Mapping[str, JSONValue]) -> None:
        import math

        self.config = JointSequenceConfig.model_validate(config)
        lengths = {len(row) for row in self.config.commands}
        if (
            len(lengths) != 1
            or 0 in lengths
            or any(not math.isfinite(v) for row in self.config.commands for v in row)
        ):
            raise DomainValidationError(
                "joint sequence requires finite, equally sized nonempty commands"
            )
        self._closed = False

    def reset(self, seed: int) -> None:
        del seed
        if self._closed:
            raise InvalidStateTransitionError("joint sequence policy is closed")

    def act(self, frame: Frame, task: TaskSpec, step_index: int) -> DataValue:
        del frame
        if self._closed:
            raise InvalidStateTransitionError("joint sequence policy is closed")
        if task.action.kind != "joint_position" or step_index < 0:
            raise DomainValidationError(
                "joint sequence requires position control and a nonnegative index"
            )
        command = self.config.commands[min(step_index, len(self.config.commands) - 1)]
        if task.action.signal.shape != (len(command),) or any(
            not lo <= v <= hi for v, lo, hi in zip(command, task.action.lower, task.action.upper)
        ):
            raise DomainValidationError("joint sequence command exceeds the task action bounds")
        return DataValue(
            dtype=task.action.signal.dtype,
            shape=task.action.signal.shape,
            unit=task.action.signal.unit,
            frame=task.action.signal.frame,
            inline=command,
        )

    def close(self) -> None:
        self._closed = True


class JointTargetPolicy:
    descriptor = builtin_method(
        MethodCapability.POLICY_INFERENCE,
        "joint_target",
        input_contract="Frame/1+TaskSpec/1",
        output_contract="Action/1",
    )

    def __init__(self, config: Mapping[str, JSONValue]) -> None:
        import math

        self.config = JointTargetConfig.model_validate(config)
        if any(not math.isfinite(value) for value in self.config.target):
            raise DomainValidationError("policy targets must be finite")
        self._closed = False

    def reset(self, seed: int) -> None:
        del seed
        if self._closed:
            raise InvalidStateTransitionError("joint target policy is closed")

    def act(self, frame: Frame, task: TaskSpec, step_index: int) -> DataValue:
        del step_index
        if self._closed:
            raise InvalidStateTransitionError("joint target policy is closed")
        if task.action.kind not in {"joint_position", "joint_delta"}:
            raise DomainValidationError("joint target policy requires position or delta actions")
        target = self.config.target
        if task.action.signal.shape != (len(target),):
            raise DomainValidationError("policy target does not match the action vector")
        if task.action.kind == "joint_delta":
            observation = frame.observations.get("joint.position")
            if (
                observation is None
                or observation.inline is None
                or len(observation.inline) != len(target)
            ):
                raise DomainValidationError("delta policy requires inline joint.position")
            target = tuple(q - float(p) for q, p in zip(target, observation.inline))
        command = tuple(
            max(lo, min(hi, q)) for q, lo, hi in zip(target, task.action.lower, task.action.upper)
        )
        return DataValue(
            dtype=task.action.signal.dtype,
            shape=task.action.signal.shape,
            unit=task.action.signal.unit,
            frame=task.action.signal.frame,
            inline=command,
        )

    def close(self) -> None:
        self._closed = True
