from __future__ import annotations

import math

import pytest
from pydantic import JsonValue, ValidationError

from physrealx.domain import (
    ActionSpec,
    ArtifactRef,
    AssetEvidenceState,
    DataValue,
    GeometryRole,
    JointSpec,
    JointType,
    Predicate,
    PredicateOperator,
    SignalDType,
    SignalSpec,
    TaskSpec,
    canonical_json_bytes,
    canonical_record_id,
)
from physrealx.reference import reference_asset, reference_scene, reference_task


def test_canonical_record_id_is_mapping_order_independent() -> None:
    left: dict[str, JsonValue] = {"b": [2, 3], "a": 1}
    right: dict[str, JsonValue] = {"a": 1, "b": [2, 3]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_record_id(left) == canonical_record_id(right)


def test_reference_asset_has_separate_visual_and_collision_geometry() -> None:
    asset = reference_asset()
    scene = reference_scene(asset)

    assert {item.role for item in asset.geometries} == {
        GeometryRole.VISUAL,
        GeometryRole.COLLISION,
    }
    assert asset.joints[0].joint_type is JointType.PRISMATIC
    assert scene.entities[0].asset.resource_id == asset.resource_id
    assert reference_task(scene).scene.resource_id == scene.resource_id


def test_asset_evidence_state_cannot_claim_identification_without_evidence() -> None:
    payload = reference_asset().model_dump(mode="json")
    payload["evidence_state"] = AssetEvidenceState.IDENTIFIED.value
    with pytest.raises(ValidationError, match="require immutable evidence"):
        type(reference_asset()).model_validate(payload)

    payload["evidence"] = [
        ArtifactRef(
            artifact_id=f"art_{'e' * 32}",
            size_bytes=10,
            media_type="application/vnd.physrealx.identification+json",
        ).model_dump(mode="json")
    ]
    identified = type(reference_asset()).model_validate(payload)
    assert identified.evidence_state is AssetEvidenceState.IDENTIFIED


def test_asset_requires_positive_definite_inertia_and_primitive_dimensions() -> None:
    payload = reference_asset().model_dump(mode="json")
    payload["physics"][0]["inertia_kg_m2"][0][0] = -1.0
    with pytest.raises(ValidationError, match="positive definite"):
        type(reference_asset()).model_validate(payload)

    payload = reference_asset().model_dump(mode="json")
    payload["geometries"][0]["primitive_dimensions_m"] = None
    with pytest.raises(ValidationError, match="requires finite positive dimensions"):
        type(reference_asset()).model_validate(payload)


def test_joint_axis_must_be_a_unit_vector() -> None:
    with pytest.raises(ValidationError, match="unit vector"):
        JointSpec(
            name="drawer.slide",
            parent="cabinet",
            child="drawer",
            joint_type=JointType.PRISMATIC,
            axis=(2.0, 0.0, 0.0),
            lower_limit=0.0,
            upper_limit=0.4,
        )


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_data_value_rejects_non_finite_inline_values(invalid: float) -> None:
    with pytest.raises(ValidationError):
        DataValue(
            dtype=SignalDType.FLOAT32,
            shape=(1,),
            unit="m",
            frame="world",
            inline=(invalid,),
        )


def test_data_value_requires_exactly_one_payload() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        DataValue(
            dtype=SignalDType.FLOAT32,
            shape=(1,),
            unit="m",
            frame="world",
        )


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (SignalDType.FLOAT32, True),
        (SignalDType.FLOAT32, 1e300),
        (SignalDType.INT32, 2**31),
        (SignalDType.INT64, -(2**63) - 1),
        (SignalDType.UINT8, 256),
    ],
)
def test_data_value_enforces_declared_numeric_range(
    dtype: SignalDType, value: bool | int | float
) -> None:
    with pytest.raises(ValidationError):
        DataValue(dtype=dtype, shape=(1,), unit="1", frame="world", inline=(value,))


def test_task_predicates_cannot_reference_action_signal() -> None:
    asset = reference_asset()
    task = reference_task(reference_scene(asset))
    payload = task.model_dump()
    action = ActionSpec.model_validate(payload["action"])
    payload["success_conditions"][0]["signal"] = action.signal.name

    with pytest.raises(ValidationError, match="observation signals"):
        TaskSpec.model_validate(payload)


def test_absolute_predicate_rejects_negative_threshold() -> None:
    with pytest.raises(ValidationError, match="non-negative"):
        Predicate(signal="error", operator=PredicateOperator.ABS_LE, value=-0.1)


def test_signal_units_and_frames_are_machine_readable() -> None:
    with pytest.raises(ValidationError, match="machine-readable"):
        SignalSpec(
            name="camera.depth",
            shape=(1,),
            dtype=SignalDType.FLOAT32,
            unit="meters per second",
            frame="world",
        )
    with pytest.raises(ValidationError, match="without traversal"):
        SignalSpec(
            name="camera.depth",
            shape=(1,),
            dtype=SignalDType.FLOAT32,
            unit="m",
            frame="../world",
        )


def test_unknown_contract_fields_are_rejected() -> None:
    payload = reference_asset().model_dump(mode="json")
    payload["legacy_name"] = "unsupported"

    with pytest.raises(ValidationError, match="Extra inputs"):
        type(reference_asset()).model_validate(payload)


def test_contracts_are_deeply_frozen_and_round_trip_json() -> None:
    asset = reference_asset()

    assert isinstance(asset.metadata, dict)
    with pytest.raises(TypeError):
        asset.metadata["mutable"] = True
    assert type(asset).model_validate_json(asset.model_dump_json()) == asset

    updated = asset.model_copy(update={"metadata": {"nested": [1, 2]}})
    nested = updated.metadata["nested"]
    assert isinstance(nested, list)
    with pytest.raises(TypeError):
        nested.append(3)


def test_nested_non_finite_metadata_is_rejected() -> None:
    payload = reference_asset().model_dump(mode="json")
    payload["metadata"] = {"nested": [1.0, math.inf]}

    with pytest.raises(ValidationError, match="NaN or infinity"):
        type(reference_asset()).model_validate(payload)


def test_frozen_metadata_remains_canonical_json_compatible() -> None:
    metadata = reference_asset().metadata

    assert canonical_json_bytes(metadata) == b'{"purpose":"executable_contract"}'
