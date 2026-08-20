from __future__ import annotations

import pytest

from physrealx.domain import DataValue
from physrealx.errors import (
    DomainValidationError,
    InvalidStateTransitionError,
    PluginError,
    ResourceConflictError,
)
from physrealx.plugins import PluginDescriptor, PluginRegistry
from physrealx.reference import (
    ReferenceDrawerBackend,
    ScriptedDrawerPolicy,
    reference_asset,
    reference_task,
)


def test_reference_reset_is_seed_deterministic() -> None:
    asset = reference_asset()
    task = reference_task(asset)
    backend = ReferenceDrawerBackend()
    first = backend.create_session(asset, task, {})
    second = backend.create_session(asset, task, {})
    try:
        assert first.reset(42) == second.reset(42)
    finally:
        first.close()
        second.close()


def test_reference_session_enforces_lifecycle() -> None:
    asset = reference_asset()
    task = reference_task(asset)
    session = ReferenceDrawerBackend().create_session(asset, task, {})
    policy = ScriptedDrawerPolicy({})
    try:
        with pytest.raises(InvalidStateTransitionError, match="reset"):
            session.step(
                DataValue(
                    dtype=task.action.signal.dtype,
                    shape=task.action.signal.shape,
                    unit=task.action.signal.unit,
                    frame=task.action.signal.frame,
                    inline=(0.01,),
                )
            )
    finally:
        policy.close()
        session.close()
    with pytest.raises(InvalidStateTransitionError, match="closed"):
        session.reset(1)


def test_registry_rejects_duplicate_plugins() -> None:
    registry = PluginRegistry()
    registry.register_backend(ReferenceDrawerBackend())

    with pytest.raises(ResourceConflictError, match="already registered"):
        registry.register_backend(ReferenceDrawerBackend())


def test_scripted_policy_can_complete_reference_task() -> None:
    asset = reference_asset()
    task = reference_task(asset)
    session = ReferenceDrawerBackend().create_session(asset, task, {})
    policy = ScriptedDrawerPolicy({})
    frame = session.reset(5)
    policy.reset(5)
    try:
        result = None
        for index in range(task.max_steps):
            result = session.step(policy.act(frame, task, index))
            frame = result.frame
            if result.terminated:
                break
        assert result is not None
        assert result.terminated
        assert "task_success" in result.events
    finally:
        policy.close()
        session.close()


def test_policy_descriptor_kind_is_checked() -> None:
    registry = PluginRegistry()
    descriptor = PluginDescriptor(
        kind="trainer",
        name="bad",
        version="1.0.0",
        api_version="1",
    )
    with pytest.raises(PluginError, match="policy descriptor"):
        registry.register_policy(descriptor, ScriptedDrawerPolicy)


def test_policy_factory_identity_must_match_descriptor() -> None:
    registry = PluginRegistry()
    descriptor = PluginDescriptor(
        kind="policy",
        name="different_policy",
        version="1.0.0",
        api_version="1",
    )
    registry.register_policy(descriptor, ScriptedDrawerPolicy)

    with pytest.raises(PluginError, match="mismatched plugin identity"):
        registry.policy("different_policy", {})


def test_reference_plugins_reject_unknown_configuration() -> None:
    asset = reference_asset()
    task = reference_task(asset)
    with pytest.raises(DomainValidationError, match="unknown reference_drawer"):
        ReferenceDrawerBackend().create_session(asset, task, {"dt_second": 0.1})
    with pytest.raises(DomainValidationError, match="unknown scripted_drawer"):
        ScriptedDrawerPolicy({"step_sizes": 0.1})
