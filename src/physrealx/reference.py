"""Deterministic articulated-drawer backend used as the executable contract."""

from __future__ import annotations

import math
import random
import uuid
from collections.abc import Mapping

from physrealx.contracts import API_VERSION, BackendCapabilities, Frame, StepResult
from physrealx.domain import (
    ActionSpec,
    AssetSpec,
    BodyPhysicsSpec,
    DataValue,
    GeometryRepresentation,
    GeometryRole,
    GeometrySpec,
    JointSpec,
    JointType,
    JSONValue,
    ParameterSource,
    Predicate,
    PredicateOperator,
    RunSpec,
    ScalarParameter,
    SignalDType,
    SignalSpec,
    TaskSpec,
)
from physrealx.errors import DomainValidationError, InvalidStateTransitionError


def _number(
    config: Mapping[str, JSONValue],
    name: str,
    default: float,
    *,
    minimum: float | None = None,
) -> float:
    raw = config.get(name, default)
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise DomainValidationError(f"{name!r} must be numeric")
    value = float(raw)
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise DomainValidationError(f"{name!r} is outside its valid range")
    return value


def _reject_unknown(
    config: Mapping[str, JSONValue],
    allowed: frozenset[str],
    *,
    label: str,
) -> None:
    unknown = set(config) - allowed
    if unknown:
        raise DomainValidationError(f"unknown {label} config fields: {sorted(unknown)!r}")


class ReferenceDrawerSession:
    """Small deterministic dynamics model, not a substitute for a physics engine."""

    def __init__(self, asset: AssetSpec, task: TaskSpec, config: Mapping[str, JSONValue]) -> None:
        _reject_unknown(
            config,
            frozenset({"dt_seconds", "response", "friction", "initial_min", "initial_max"}),
            label="reference_drawer backend",
        )
        joints = [joint for joint in asset.joints if joint.joint_type is JointType.PRISMATIC]
        if len(joints) != 1:
            raise DomainValidationError("reference_drawer requires exactly one prismatic joint")
        if task.action.signal.shape != (1,) or task.action.kind != "joint_delta":
            raise DomainValidationError(
                "reference_drawer requires a one-dimensional joint_delta action"
            )
        self._joint = joints[0]
        self._task = task
        self._dt = _number(config, "dt_seconds", 0.05, minimum=1e-6)
        self._response = _number(config, "response", 1.0, minimum=0.0)
        self._friction = _number(config, "friction", 0.002, minimum=0.0)
        self._initial_min = _number(config, "initial_min", 0.0)
        self._initial_max = _number(config, "initial_max", 0.03)
        if self._initial_min > self._initial_max:
            raise DomainValidationError("initial_min cannot exceed initial_max")
        if (
            self._initial_min < self._joint.lower_limit
            or self._initial_max > self._joint.upper_limit
        ):
            raise DomainValidationError("initial drawer range exceeds joint limits")
        self._closed = False
        self._initialized = False
        self._position = 0.0
        self._velocity = 0.0
        self._step_index = 0

    @property
    def dt_seconds(self) -> float:
        return self._dt

    def reset(self, seed: int) -> Frame:
        self._ensure_open()
        generator = random.Random(seed)
        self._position = generator.uniform(self._initial_min, self._initial_max)
        self._velocity = 0.0
        self._step_index = 0
        self._initialized = True
        frame = self._frame()
        if not all(item.evaluate(self._scalar_state()) for item in self._task.initial_conditions):
            raise DomainValidationError(
                "reference backend reset does not satisfy initial conditions"
            )
        return frame

    def step(self, action: DataValue) -> StepResult:
        self._ensure_open()
        if not self._initialized:
            raise InvalidStateTransitionError("session must be reset before step")
        action.assert_matches(self._task.action.signal)
        if action.inline is None:
            raise DomainValidationError("reference backend requires an inline numeric action")
        command = float(action.inline[0])
        lower = self._task.action.lower[0]
        upper = self._task.action.upper[0]
        if not lower <= command <= upper:
            raise DomainValidationError("action is outside declared bounds")
        effective = math.copysign(max(abs(command) - self._friction, 0.0), command)
        previous = self._position
        self._position = min(
            self._joint.upper_limit,
            max(self._joint.lower_limit, previous + self._response * effective),
        )
        self._velocity = (self._position - previous) / self._dt
        self._step_index += 1
        state = self._scalar_state()
        success = all(item.evaluate(state) for item in self._task.success_conditions)
        terminal_condition = any(item.evaluate(state) for item in self._task.termination_conditions)
        truncated = self._step_index >= self._task.max_steps and not success
        return StepResult(
            frame=self._frame(),
            reward=1.0 if success else -0.01,
            terminated=success or terminal_condition,
            truncated=truncated,
            events=("task_success",) if success else (),
        )

    def close(self) -> None:
        self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise InvalidStateTransitionError("simulation session is closed")

    def _scalar_state(self) -> dict[str, float]:
        return {
            "drawer.position": self._position,
            "drawer.velocity": self._velocity,
        }

    def _frame(self) -> Frame:
        timestamp = self._step_index * self._dt
        values: dict[str, DataValue] = {}
        state = self._scalar_state()
        for spec in self._task.observations:
            try:
                scalar = state[spec.name]
            except KeyError as error:
                raise DomainValidationError(
                    f"reference backend cannot produce observation {spec.name!r}"
                ) from error
            value = DataValue(
                dtype=spec.dtype,
                shape=spec.shape,
                unit=spec.unit,
                frame=spec.frame,
                timestamp_seconds=timestamp,
                inline=(scalar,),
            )
            value.assert_matches(spec)
            values[spec.name] = value
        return Frame(observations=values, state=dict(values))


class ReferenceDrawerBackend:
    name = "reference_drawer"
    plugin_version = "1.0.0"
    api_version = API_VERSION

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            rigid_body=True,
            articulation=True,
            deterministic_reset=True,
            observation_keys=("drawer.position", "drawer.velocity"),
            action_kinds=("joint_delta",),
        )

    def create_session(
        self,
        asset: AssetSpec,
        task: TaskSpec,
        config: Mapping[str, JSONValue],
    ) -> ReferenceDrawerSession:
        return ReferenceDrawerSession(asset, task, config)


class ScriptedDrawerPolicy:
    name = "scripted_drawer"
    plugin_version = "1.0.0"
    api_version = API_VERSION

    def __init__(self, config: Mapping[str, JSONValue]) -> None:
        _reject_unknown(
            config,
            frozenset({"step_size"}),
            label="scripted_drawer policy",
        )
        self._step_size = _number(config, "step_size", 0.06, minimum=1e-6)
        self._closed = False

    def reset(self, seed: int) -> None:
        del seed
        if self._closed:
            raise InvalidStateTransitionError("policy is closed")

    def act(self, frame: Frame, task: TaskSpec, step_index: int) -> DataValue:
        del step_index
        if self._closed:
            raise InvalidStateTransitionError("policy is closed")
        position_value = frame.observations.get("drawer.position")
        if position_value is None or position_value.inline is None:
            raise DomainValidationError("scripted policy requires drawer.position")
        target = max(
            (
                predicate.value
                for predicate in task.success_conditions
                if predicate.signal == "drawer.position"
                and predicate.operator in {PredicateOperator.GE, PredicateOperator.GT}
            ),
            default=task.action.upper[0],
        )
        command = self._step_size if float(position_value.inline[0]) < target else 0.0
        command = min(task.action.upper[0], max(task.action.lower[0], command))
        return DataValue(
            dtype=task.action.signal.dtype,
            shape=task.action.signal.shape,
            unit=task.action.signal.unit,
            frame=task.action.signal.frame,
            timestamp_seconds=position_value.timestamp_seconds,
            inline=(command,),
        )

    def close(self) -> None:
        self._closed = True


def reference_asset() -> AssetSpec:
    measured = ParameterSource.MEASURED

    def parameter(value: float) -> ScalarParameter:
        return ScalarParameter(value=value, uncertainty=0.0, source=measured, confidence=1.0)

    return AssetSpec(
        resource_id="drawer.reference",
        version="1.0.0",
        display_name="Reference articulated drawer",
        root_link="cabinet",
        links=("cabinet", "drawer"),
        geometries=(
            GeometrySpec(
                name="cabinet.visual",
                link="cabinet",
                role=GeometryRole.VISUAL,
                representation=GeometryRepresentation.PRIMITIVE,
                uri="primitive://box/cabinet",
                primitive_dimensions_m=(0.8, 0.6, 0.8),
            ),
            GeometrySpec(
                name="cabinet.collision",
                link="cabinet",
                role=GeometryRole.COLLISION,
                representation=GeometryRepresentation.PRIMITIVE,
                uri="primitive://box/cabinet-collision",
                primitive_dimensions_m=(0.8, 0.6, 0.8),
            ),
            GeometrySpec(
                name="drawer.visual",
                link="drawer",
                role=GeometryRole.VISUAL,
                representation=GeometryRepresentation.PRIMITIVE,
                uri="primitive://box/drawer",
                primitive_dimensions_m=(0.6, 0.5, 0.18),
            ),
            GeometrySpec(
                name="drawer.collision",
                link="drawer",
                role=GeometryRole.COLLISION,
                representation=GeometryRepresentation.PRIMITIVE,
                uri="primitive://box/drawer-collision",
                primitive_dimensions_m=(0.6, 0.5, 0.18),
            ),
        ),
        joints=(
            JointSpec(
                name="drawer.slide",
                parent="cabinet",
                child="drawer",
                joint_type=JointType.PRISMATIC,
                axis=(1.0, 0.0, 0.0),
                lower_limit=0.0,
                upper_limit=0.4,
            ),
        ),
        physics=(
            BodyPhysicsSpec(
                link="cabinet",
                mass_kg=parameter(20.0),
                center_of_mass_m=(0.0, 0.0, 0.4),
                inertia_kg_m2=((1.5, 0.0, 0.0), (0.0, 2.0, 0.0), (0.0, 0.0, 1.8)),
                static_friction=parameter(0.6),
                dynamic_friction=parameter(0.5),
                restitution=parameter(0.0),
            ),
            BodyPhysicsSpec(
                link="drawer",
                mass_kg=parameter(2.0),
                center_of_mass_m=(0.0, 0.0, 0.0),
                inertia_kg_m2=((0.08, 0.0, 0.0), (0.0, 0.07, 0.0), (0.0, 0.0, 0.1)),
                static_friction=parameter(0.35),
                dynamic_friction=parameter(0.3),
                restitution=parameter(0.0),
            ),
        ),
        metadata={"purpose": "executable_contract"},
    )


def reference_task(asset: AssetSpec | None = None) -> TaskSpec:
    target_asset = asset or reference_asset()
    position = SignalSpec(
        name="drawer.position",
        shape=(1,),
        dtype=SignalDType.FLOAT32,
        unit="m",
        frame="drawer_joint",
        frequency_hz=20.0,
    )
    velocity = SignalSpec(
        name="drawer.velocity",
        shape=(1,),
        dtype=SignalDType.FLOAT32,
        unit="m/s",
        frame="drawer_joint",
        frequency_hz=20.0,
    )
    action = SignalSpec(
        name="drawer.delta",
        shape=(1,),
        dtype=SignalDType.FLOAT32,
        unit="m",
        frame="drawer_joint",
        frequency_hz=20.0,
    )
    return TaskSpec(
        resource_id="open_drawer.reference",
        version="1.0.0",
        asset=target_asset.reference(),
        instruction="Open the drawer beyond 30 centimeters.",
        initial_conditions=(
            Predicate(signal="drawer.position", operator=PredicateOperator.LE, value=0.05),
        ),
        success_conditions=(
            Predicate(signal="drawer.position", operator=PredicateOperator.GE, value=0.30),
        ),
        action=ActionSpec(
            kind="joint_delta",
            signal=action,
            lower=(-0.08,),
            upper=(0.08,),
        ),
        observations=(position, velocity),
        max_steps=20,
    )


def reference_run(
    asset: AssetSpec | None = None,
    task: TaskSpec | None = None,
    *,
    episode_count: int = 8,
    base_seed: int = 7,
) -> RunSpec:
    target_asset = asset or reference_asset()
    target_task = task or reference_task(target_asset)
    run_key = (
        f"physrealx:reference:{target_asset.resource_id}@{target_asset.version}:"
        f"{target_task.resource_id}@{target_task.version}:{episode_count}:{base_seed}"
    )
    return RunSpec(
        run_id=str(uuid.uuid5(uuid.NAMESPACE_URL, run_key)),
        asset=target_asset.reference(),
        task=target_task.reference(),
        backend="reference_drawer",
        policy="scripted_drawer",
        episode_count=episode_count,
        base_seed=base_seed,
        dataset_id="drawer.demo",
        dataset_version="1.0.0",
        record_signals=("drawer.position", "drawer.velocity"),
    )


__all__ = [
    "ReferenceDrawerBackend",
    "ReferenceDrawerSession",
    "ScriptedDrawerPolicy",
    "reference_asset",
    "reference_run",
    "reference_task",
]
