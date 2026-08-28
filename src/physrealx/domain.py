"""Immutable domain contracts for PhysReal-X.

The module is intentionally infrastructure-free: importing it does not touch a
filesystem, database, simulator, Torch, CUDA, or the network.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
import uuid
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from enum import Enum
from functools import reduce
from operator import mul
from typing import Any, Dict, Final, List, Literal, NoReturn, TypeVar, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)
from typing_extensions import Annotated, Self, TypeAlias

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:

    class StrEnum(str, Enum):
        """Python 3.8-3.10 equivalent of :class:`enum.StrEnum`."""

        def __str__(self) -> str:
            return str.__str__(self)


JSONValue: TypeAlias = JsonValue
SCHEMA_VERSION: Final[Literal[1]] = 1

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_VERSION = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_URI = re.compile(r"^[a-z][a-z0-9+.-]*://\S+$")
_UNIT = re.compile(r"^(?:1|[A-Za-zµ°%][A-Za-z0-9µ°%*/^()._-]{0,62})$")
_FRAME = re.compile(r"^[A-Za-z][A-Za-z0-9_./:-]{0,126}$")

PositiveFloat = Annotated[float, Field(gt=0, allow_inf_nan=False)]
NonNegativeFloat = Annotated[float, Field(ge=0, allow_inf_nan=False)]
UnitFloat = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
_PairT = TypeVar("_PairT")


def _pairwise(values: tuple[_PairT, ...]) -> Iterator[tuple[_PairT, _PairT]]:
    return zip(values, values[1:])


def _identifier(value: str, label: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError(
            f"{label} must start with a lowercase letter and contain only "
            "lowercase letters, digits, '.', '_' or '-'"
        )
    return value


def _semver(value: str) -> str:
    if _VERSION.fullmatch(value) is None:
        raise ValueError("version must be an explicit semantic version such as '1.0.0'")
    return value


def _unit(value: str) -> str:
    if _UNIT.fullmatch(value) is None:
        raise ValueError("unit must use a compact machine-readable symbol")
    return value


def _frame(value: str) -> str:
    if _FRAME.fullmatch(value) is None or ".." in value:
        raise ValueError("frame must be an absolute logical frame name without traversal")
    return value


def _count(shape: tuple[int, ...]) -> int:
    return reduce(mul, shape, 1)


class _FrozenDict(Dict[str, Any]):
    """A dict-compatible immutable container used by frozen contracts."""

    @staticmethod
    def _reject(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise TypeError("domain mappings are immutable")

    __setitem__ = _reject
    __delitem__ = _reject
    __ior__ = _reject
    clear = _reject
    pop = _reject
    popitem = _reject
    setdefault = _reject
    update = _reject

    def __copy__(self) -> _FrozenDict:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenDict:
        del memo
        return self


class _FrozenList(List[Any]):
    """A list-compatible immutable container used for nested JSON arrays."""

    @staticmethod
    def _reject(*args: object, **kwargs: object) -> NoReturn:
        del args, kwargs
        raise TypeError("domain sequences are immutable")

    __setitem__ = _reject
    __delitem__ = _reject
    __iadd__ = _reject
    __imul__ = _reject
    append = _reject
    clear = _reject
    extend = _reject
    insert = _reject
    pop = _reject
    remove = _reject
    reverse = _reject
    sort = _reject

    def __copy__(self) -> _FrozenList:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> _FrozenList:
        del memo
        return self


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("domain values must not contain NaN or infinity")
    if isinstance(value, datetime) and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError("domain timestamps must include a UTC offset")
    if isinstance(value, (_FrozenDict, _FrozenList)):
        return value
    if isinstance(value, Mapping):
        return _FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_deep_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_deep_freeze(item) for item in value)
    return value


def canonical_json_bytes(value: BaseModel | JsonValue) -> bytes:
    """Return deterministic UTF-8 JSON and reject NaN/Infinity."""

    payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
    return json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def sha256_digest(value: BaseModel | JsonValue | bytes) -> str:
    payload = value if isinstance(value, bytes) else canonical_json_bytes(value)
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


class DomainModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    @model_validator(mode="after")
    def freeze_nested_values(self) -> DomainModel:
        for field_name in type(self).model_fields:
            object.__setattr__(self, field_name, _deep_freeze(getattr(self, field_name)))
        return self

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        """Return a fully revalidated copy; Pydantic's unchecked update is unsafe here."""

        del deep  # Every nested domain value is immutable; revalidation creates a safe copy.
        payload = self.model_dump(round_trip=True)
        if update:
            payload.update(update)
        return type(self).model_validate(payload)


class ResourceType(StrEnum):
    ASSET = "asset"
    ASSET_BUILD = "asset_build"
    OBSERVATION_SET = "observation_set"
    SCENE = "scene"
    TASK = "task"
    DATASET = "dataset"
    MODEL = "model"
    EVALUATION = "evaluation"


class ArtifactRef(DomainModel):
    digest: str
    size_bytes: int = Field(ge=0)
    media_type: str = Field(min_length=1, max_length=255)

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("digest must be 'sha256:' followed by 64 lowercase hex digits")
        return value


class VersionedRef(DomainModel):
    resource_type: ResourceType
    resource_id: str
    version: str
    digest: str | None = None

    @field_validator("resource_id")
    @classmethod
    def validate_resource_id(cls, value: str) -> str:
        return _identifier(value, "resource_id")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _semver(value)

    @field_validator("digest")
    @classmethod
    def validate_optional_digest(cls, value: str | None) -> str | None:
        if value is not None and _DIGEST.fullmatch(value) is None:
            raise ValueError("digest must be 'sha256:' followed by 64 lowercase hex digits")
        return value

    @property
    def resolved(self) -> bool:
        return self.digest is not None


class GeometryRepresentation(StrEnum):
    MESH = "mesh"
    GAUSSIAN = "gaussian"
    POINT_CLOUD = "point_cloud"
    SDF = "sdf"
    PRIMITIVE = "primitive"


class GeometryRole(StrEnum):
    VISUAL = "visual"
    COLLISION = "collision"


class PoseSpec(DomainModel):
    translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rotation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    @model_validator(mode="after")
    def validate_pose(self) -> PoseSpec:
        values = self.translation_m + self.rotation_xyzw
        if any(not math.isfinite(value) for value in values):
            raise ValueError("pose values must be finite")
        norm = math.sqrt(sum(value * value for value in self.rotation_xyzw))
        if abs(norm - 1.0) > 1e-6:
            raise ValueError("pose rotation_xyzw must be a unit quaternion")
        return self


class GeometrySpec(DomainModel):
    name: str
    link: str
    role: GeometryRole
    representation: GeometryRepresentation
    uri: str
    checksum: str | None = None
    local_pose: PoseSpec = Field(default_factory=PoseSpec)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    primitive_dimensions_m: tuple[float, float, float] | None = None

    @field_validator("name", "link")
    @classmethod
    def validate_names(cls, value: str, info: object) -> str:
        field_name = getattr(info, "field_name", "identifier")
        return _identifier(value, field_name)

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, value: str) -> str:
        if _URI.fullmatch(value) is None:
            raise ValueError("geometry uri must be absolute and include a scheme")
        return value

    @field_validator("checksum")
    @classmethod
    def validate_checksum(cls, value: str | None) -> str | None:
        if value is not None and _DIGEST.fullmatch(value) is None:
            raise ValueError("checksum must be a SHA-256 digest")
        return value

    @field_validator("scale")
    @classmethod
    def validate_scale(cls, value: tuple[float, float, float]) -> tuple[float, float, float]:
        if any(not math.isfinite(item) or item <= 0 for item in value):
            raise ValueError("geometry scale must contain finite positive values")
        return value

    @model_validator(mode="after")
    def validate_external_geometry(self) -> GeometrySpec:
        if self.representation is not GeometryRepresentation.PRIMITIVE and self.checksum is None:
            raise ValueError("non-primitive geometry requires a checksum")
        if self.representation is GeometryRepresentation.PRIMITIVE:
            if self.primitive_dimensions_m is None or any(
                not math.isfinite(item) or item <= 0 for item in self.primitive_dimensions_m
            ):
                raise ValueError("primitive geometry requires finite positive dimensions")
        elif self.primitive_dimensions_m is not None:
            raise ValueError("primitive_dimensions_m is only valid for primitive geometry")
        return self


class JointType(StrEnum):
    FIXED = "fixed"
    REVOLUTE = "revolute"
    PRISMATIC = "prismatic"


class JointSpec(DomainModel):
    name: str
    parent: str
    child: str
    joint_type: JointType
    parent_to_joint: PoseSpec = Field(default_factory=PoseSpec)
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    lower_limit: float = Field(allow_inf_nan=False)
    upper_limit: float = Field(allow_inf_nan=False)
    damping: NonNegativeFloat = 0.0
    friction: NonNegativeFloat = 0.0
    effort_limit: PositiveFloat | None = None
    velocity_limit: PositiveFloat | None = None

    @field_validator("name", "parent", "child")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        return _identifier(value, getattr(info, "field_name", "identifier"))

    @model_validator(mode="after")
    def validate_joint(self) -> JointSpec:
        if self.parent == self.child:
            raise ValueError("a joint cannot connect a link to itself")
        norm = math.sqrt(sum(component * component for component in self.axis))
        if not math.isfinite(norm) or abs(norm - 1.0) > 1e-6:
            raise ValueError("joint axis must be a finite unit vector")
        if self.lower_limit > self.upper_limit:
            raise ValueError("joint lower_limit cannot exceed upper_limit")
        if self.joint_type is JointType.FIXED and (
            self.lower_limit != 0.0 or self.upper_limit != 0.0
        ):
            raise ValueError("fixed joints must have zero limits")
        return self


class ParameterSource(StrEnum):
    MEASURED = "measured"
    INFERRED = "inferred"
    DEFAULT = "default"


class ScalarParameter(DomainModel):
    value: float = Field(allow_inf_nan=False)
    uncertainty: NonNegativeFloat = 0.0
    source: ParameterSource
    confidence: UnitFloat


class BodyPhysicsSpec(DomainModel):
    link: str
    mass_kg: ScalarParameter
    center_of_mass_m: tuple[float, float, float]
    inertia_kg_m2: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]
    static_friction: ScalarParameter
    dynamic_friction: ScalarParameter
    restitution: ScalarParameter

    @field_validator("link")
    @classmethod
    def validate_link(cls, value: str) -> str:
        return _identifier(value, "link")

    @model_validator(mode="after")
    def validate_values(self) -> BodyPhysicsSpec:
        if self.mass_kg.value <= 0:
            raise ValueError("mass_kg must be positive")
        if self.static_friction.value < 0 or self.dynamic_friction.value < 0:
            raise ValueError("friction coefficients cannot be negative")
        if not 0 <= self.restitution.value <= 1:
            raise ValueError("restitution must be in [0, 1]")
        if any(not math.isfinite(value) for value in self.center_of_mass_m):
            raise ValueError("center_of_mass_m must be finite")
        matrix = self.inertia_kg_m2
        if any(not math.isfinite(value) for row in matrix for value in row):
            raise ValueError("inertia tensor must be finite")
        if any(
            abs(matrix[row][column] - matrix[column][row]) > 1e-9
            for row in range(3)
            for column in range(3)
        ):
            raise ValueError("inertia tensor must be symmetric")
        first_minor = matrix[0][0]
        second_minor = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        determinant = (
            matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
            - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
            + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
        )
        if first_minor <= 0 or second_minor <= 0 or determinant <= 0:
            raise ValueError("inertia tensor must be positive definite")
        return self


class AssetSpec(DomainModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    resource_id: str
    version: str
    display_name: str = Field(min_length=1, max_length=255)
    coordinate_frame: str = "asset"
    up_axis: Literal["x", "y", "z"] = "z"
    handedness: Literal["right"] = "right"
    root_pose: PoseSpec = Field(default_factory=PoseSpec)
    root_link: str
    links: tuple[str, ...]
    geometries: tuple[GeometrySpec, ...]
    joints: tuple[JointSpec, ...] = ()
    physics: tuple[BodyPhysicsSpec, ...]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("resource_id", "root_link")
    @classmethod
    def validate_identifier_fields(cls, value: str, info: object) -> str:
        return _identifier(value, getattr(info, "field_name", "identifier"))

    @field_validator("coordinate_frame")
    @classmethod
    def validate_coordinate_frame(cls, value: str) -> str:
        return _frame(value)

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _semver(value)

    @field_validator("links")
    @classmethod
    def validate_links(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("an asset requires at least one link")
        for value in values:
            _identifier(value, "link")
        if len(set(values)) != len(values):
            raise ValueError("asset links must be unique")
        return values

    @model_validator(mode="after")
    def validate_asset_graph(self) -> AssetSpec:
        links = set(self.links)
        if self.root_link not in links:
            raise ValueError("root_link must be present in links")
        if not any(item.role is GeometryRole.VISUAL for item in self.geometries):
            raise ValueError("a trainable asset requires visual geometry")
        if not any(item.role is GeometryRole.COLLISION for item in self.geometries):
            raise ValueError("a trainable asset requires collision geometry")
        if any(item.link not in links for item in self.geometries):
            raise ValueError("all geometry links must exist in the asset")
        if len({item.name for item in self.geometries}) != len(self.geometries):
            raise ValueError("geometry names must be unique")
        if len({item.name for item in self.joints}) != len(self.joints):
            raise ValueError("joint names must be unique")
        children: set[str] = set()
        parents: dict[str, str] = {}
        for joint in self.joints:
            if joint.parent not in links or joint.child not in links:
                raise ValueError("all joint links must exist in the asset")
            if joint.child in children:
                raise ValueError("a link cannot have more than one parent joint")
            children.add(joint.child)
            parents[joint.child] = joint.parent
        if self.root_link in children:
            raise ValueError("root_link cannot be a joint child")
        if children != links - {self.root_link}:
            raise ValueError("every non-root link must be connected by exactly one parent joint")
        for link in links:
            seen: set[str] = set()
            current = link
            while current in parents:
                if current in seen:
                    raise ValueError("asset joint graph contains a cycle")
                seen.add(current)
                current = parents[current]
        if len({item.link for item in self.physics}) != len(self.physics):
            raise ValueError("physics entries must have unique links")
        if any(item.link not in links for item in self.physics):
            raise ValueError("all physics links must exist in the asset")
        if {item.link for item in self.physics} != links:
            raise ValueError("every asset link requires an explicit physics entry")
        return self

    def reference(self, digest: str | None = None) -> VersionedRef:
        return VersionedRef(
            resource_type=ResourceType.ASSET,
            resource_id=self.resource_id,
            version=self.version,
            digest=digest,
        )


class SceneEntityRole(StrEnum):
    ROBOT = "robot"
    OBJECT = "object"
    ENVIRONMENT = "environment"


class SceneEntitySpec(DomainModel):
    name: str
    asset: VersionedRef
    role: SceneEntityRole
    pose: PoseSpec = Field(default_factory=PoseSpec)
    tags: tuple[str, ...] = ()

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _identifier(value, "scene entity name")

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            _identifier(value, "scene entity tag")
        if len(values) != len(set(values)):
            raise ValueError("scene entity tags must be unique")
        return values

    @model_validator(mode="after")
    def validate_asset_reference(self) -> SceneEntitySpec:
        if self.asset.resource_type is not ResourceType.ASSET:
            raise ValueError("SceneEntitySpec.asset must reference an asset")
        return self


class SceneSpec(DomainModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    resource_id: str
    version: str
    display_name: str = Field(min_length=1, max_length=255)
    entities: tuple[SceneEntitySpec, ...]
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("resource_id")
    @classmethod
    def validate_resource_id(cls, value: str) -> str:
        return _identifier(value, "resource_id")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _semver(value)

    @model_validator(mode="after")
    def validate_scene(self) -> SceneSpec:
        if not self.entities:
            raise ValueError("a scene requires at least one entity")
        names = [item.name for item in self.entities]
        if len(names) != len(set(names)):
            raise ValueError("scene entity names must be unique")
        if not any(item.role is not SceneEntityRole.ENVIRONMENT for item in self.entities):
            raise ValueError("a scene requires at least one robot or object entity")
        return self

    def reference(self, digest: str | None = None) -> VersionedRef:
        return VersionedRef(
            resource_type=ResourceType.SCENE,
            resource_id=self.resource_id,
            version=self.version,
            digest=digest,
        )


class SignalDType(StrEnum):
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    BOOL = "bool"


class SignalSpec(DomainModel):
    name: str
    shape: tuple[int, ...]
    dtype: SignalDType
    unit: str = Field(min_length=1, max_length=63)
    frame: str = Field(min_length=1, max_length=127)
    frequency_hz: PositiveFloat | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        return _identifier(value, "signal name")

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(item <= 0 for item in value):
            raise ValueError("signal shape must contain positive dimensions")
        return value

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, value: str) -> str:
        return _unit(value)

    @field_validator("frame")
    @classmethod
    def validate_frame(cls, value: str) -> str:
        return _frame(value)


InlineScalar: TypeAlias = Union[bool, int, float]


class DataValue(DomainModel):
    dtype: SignalDType
    shape: tuple[int, ...]
    unit: str = Field(min_length=1, max_length=63)
    frame: str = Field(min_length=1, max_length=127)
    timestamp_seconds: NonNegativeFloat | None = None
    inline: tuple[InlineScalar, ...] | None = None
    artifact: ArtifactRef | None = None

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or any(item <= 0 for item in value):
            raise ValueError("data shape must contain positive dimensions")
        return value

    @field_validator("unit")
    @classmethod
    def validate_unit(cls, value: str) -> str:
        return _unit(value)

    @field_validator("frame")
    @classmethod
    def validate_frame(cls, value: str) -> str:
        return _frame(value)

    @model_validator(mode="after")
    def validate_payload(self) -> DataValue:
        if (self.inline is None) == (self.artifact is None):
            raise ValueError("exactly one of inline or artifact must be set")
        if self.inline is not None:
            if len(self.inline) != _count(self.shape):
                raise ValueError("inline value count does not match shape")
            for item in self.inline:
                if isinstance(item, float) and not math.isfinite(item):
                    raise ValueError("inline values must be finite")
                if self.dtype is SignalDType.BOOL and not isinstance(item, bool):
                    raise ValueError("bool data requires boolean inline values")
                if self.dtype in {SignalDType.FLOAT32, SignalDType.FLOAT64} and isinstance(
                    item, bool
                ):
                    raise ValueError("floating-point data cannot contain boolean values")
                if self.dtype in {SignalDType.FLOAT32, SignalDType.FLOAT64}:
                    try:
                        numeric = float(item)
                    except OverflowError as error:
                        raise ValueError(
                            "floating-point data is outside its declared range"
                        ) from error
                    maximum = (
                        3.4028234663852886e38
                        if self.dtype is SignalDType.FLOAT32
                        else 1.7976931348623157e308
                    )
                    if not math.isfinite(numeric) or abs(numeric) > maximum:
                        raise ValueError("floating-point data is outside its declared range")
                if self.dtype in {SignalDType.INT32, SignalDType.INT64, SignalDType.UINT8}:
                    if isinstance(item, bool) or not isinstance(item, int):
                        raise ValueError("integer data requires integer inline values")
                    if self.dtype is SignalDType.UINT8 and not 0 <= item <= 255:
                        raise ValueError("uint8 values must be in [0, 255]")
                    if self.dtype is SignalDType.INT32 and not -(2**31) <= item < 2**31:
                        raise ValueError("int32 values must fit the signed 32-bit range")
                    if self.dtype is SignalDType.INT64 and not -(2**63) <= item < 2**63:
                        raise ValueError("int64 values must fit the signed 64-bit range")
        return self

    def assert_matches(self, spec: SignalSpec) -> None:
        if (self.dtype, self.shape, self.unit, self.frame) != (
            spec.dtype,
            spec.shape,
            spec.unit,
            spec.frame,
        ):
            raise ValueError(f"data value does not match signal specification {spec.name!r}")


class PredicateOperator(StrEnum):
    GE = "ge"
    GT = "gt"
    LE = "le"
    LT = "lt"
    EQ = "eq"
    ABS_LE = "abs_le"


class Predicate(DomainModel):
    signal: str
    operator: PredicateOperator
    value: float = Field(allow_inf_nan=False)
    tolerance: NonNegativeFloat = 0.0

    @field_validator("signal")
    @classmethod
    def validate_signal(cls, value: str) -> str:
        return _identifier(value, "predicate signal")

    @model_validator(mode="after")
    def validate_operator_value(self) -> Predicate:
        if self.operator is PredicateOperator.ABS_LE and self.value < 0:
            raise ValueError("abs_le predicates require a non-negative value")
        return self

    def evaluate(self, signals: dict[str, float]) -> bool:
        actual = signals[self.signal]
        if not math.isfinite(actual):
            return False
        if self.operator is PredicateOperator.GE:
            return actual >= self.value - self.tolerance
        if self.operator is PredicateOperator.GT:
            return actual > self.value - self.tolerance
        if self.operator is PredicateOperator.LE:
            return actual <= self.value + self.tolerance
        if self.operator is PredicateOperator.LT:
            return actual < self.value + self.tolerance
        if self.operator is PredicateOperator.EQ:
            return abs(actual - self.value) <= self.tolerance
        return abs(actual) <= self.value + self.tolerance


class ActionSpec(DomainModel):
    kind: str
    signal: SignalSpec
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        return _identifier(value, "action kind")

    @model_validator(mode="after")
    def validate_limits(self) -> ActionSpec:
        count = _count(self.signal.shape)
        if len(self.lower) != count or len(self.upper) != count:
            raise ValueError("action bounds must match the signal shape")
        for lower, upper in zip(self.lower, self.upper):
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError("action bounds must be finite and ordered")
        return self


class TaskSpec(DomainModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    resource_id: str
    version: str
    scene: VersionedRef
    subject_entity: str
    instruction: str = Field(min_length=1, max_length=1024)
    initial_conditions: tuple[Predicate, ...]
    success_conditions: tuple[Predicate, ...]
    termination_conditions: tuple[Predicate, ...] = ()
    action: ActionSpec
    observations: tuple[SignalSpec, ...]
    max_steps: int = Field(gt=0, le=1_000_000)
    discount: UnitFloat = 1.0
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("resource_id")
    @classmethod
    def validate_resource_id(cls, value: str) -> str:
        return _identifier(value, "resource_id")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _semver(value)

    @field_validator("subject_entity")
    @classmethod
    def validate_subject_entity(cls, value: str) -> str:
        return _identifier(value, "subject_entity")

    @model_validator(mode="after")
    def validate_task(self) -> TaskSpec:
        if self.scene.resource_type is not ResourceType.SCENE:
            raise ValueError("TaskSpec.scene must reference a scene")
        if not self.success_conditions:
            raise ValueError("a task requires at least one success condition")
        names = [item.name for item in self.observations]
        if len(names) != len(set(names)):
            raise ValueError("observation signal names must be unique")
        if self.action.signal.name in names:
            raise ValueError("action and observation signals must have distinct names")
        known = set(names)
        predicates = self.initial_conditions + self.success_conditions + self.termination_conditions
        if any(item.signal not in known for item in predicates):
            raise ValueError("task predicates must reference declared observation signals")
        observations = {item.name: item for item in self.observations}
        non_scalar = {
            item.signal
            for item in predicates
            if observations[item.signal].shape != (1,)
            or observations[item.signal].dtype is SignalDType.BOOL
        }
        if non_scalar:
            raise ValueError(
                f"task predicates require scalar numeric observations: {sorted(non_scalar)!r}"
            )
        return self

    def reference(self, digest: str | None = None) -> VersionedRef:
        return VersionedRef(
            resource_type=ResourceType.TASK,
            resource_id=self.resource_id,
            version=self.version,
            digest=digest,
        )


class RunSpec(DomainModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scene: VersionedRef
    task: VersionedRef
    backend: str
    backend_version: str | None = None
    policy: str
    policy_version: str | None = None
    episode_count: int = Field(gt=0, le=1_000_000)
    base_seed: int = Field(ge=0, lt=2**63)
    dataset_id: str
    dataset_version: str
    record_signals: tuple[str, ...]
    backend_config: dict[str, JsonValue] = Field(default_factory=dict)
    policy_config: dict[str, JsonValue] = Field(default_factory=dict)
    tags: tuple[str, ...] = ()

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        try:
            return str(uuid.UUID(value))
        except ValueError as error:
            raise ValueError("run_id must be a UUID") from error

    @field_validator("backend", "policy", "dataset_id")
    @classmethod
    def validate_names(cls, value: str, info: object) -> str:
        return _identifier(value, getattr(info, "field_name", "identifier"))

    @field_validator("dataset_version")
    @classmethod
    def validate_dataset_version(cls, value: str) -> str:
        return _semver(value)

    @field_validator("backend_version", "policy_version")
    @classmethod
    def validate_optional_plugin_versions(cls, value: str | None) -> str | None:
        return _semver(value) if value is not None else None

    @field_validator("record_signals")
    @classmethod
    def validate_record_signals(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("at least one signal must be recorded")
        for value in values:
            _identifier(value, "record signal")
        if len(values) != len(set(values)):
            raise ValueError("record_signals must be unique")
        return values

    @model_validator(mode="after")
    def validate_refs(self) -> RunSpec:
        if self.scene.resource_type is not ResourceType.SCENE:
            raise ValueError("RunSpec.scene must reference a scene")
        if self.task.resource_type is not ResourceType.TASK:
            raise ValueError("RunSpec.task must reference a task")
        return self


class EpisodeSource(StrEnum):
    SIMULATOR = "simulator"
    REAL = "real"
    WORLD_MODEL = "world_model"


class EpisodeStep(DomainModel):
    index: int = Field(ge=0)
    timestamp_seconds: NonNegativeFloat
    observations: dict[str, DataValue]
    state: dict[str, DataValue]
    action: DataValue | None
    reward: float | None = Field(default=None, allow_inf_nan=False)
    discount: UnitFloat | None = None
    is_first: bool = False
    is_last: bool = False
    is_terminal: bool = False
    events: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_step_semantics(self) -> EpisodeStep:
        if self.is_terminal and not self.is_last:
            raise ValueError("a terminal step must also be the last step")
        if self.is_last:
            if self.action is not None or self.reward is not None or self.discount is not None:
                raise ValueError("the last observation cannot carry action, reward, or discount")
        elif self.action is None or self.reward is None or self.discount is None:
            raise ValueError("non-last steps require action, reward, and discount")
        return self


class Episode(DomainModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    episode_id: str
    run_id: str
    index: int = Field(ge=0)
    source: EpisodeSource
    seed: int = Field(ge=0, lt=2**63)
    scene: VersionedRef
    task: VersionedRef
    backend: str
    backend_version: str
    policy: str
    policy_version: str
    steps: tuple[EpisodeStep, ...]
    success: bool
    failure_reason: str | None = Field(default=None, max_length=1024)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("episode_id")
    @classmethod
    def validate_episode_id(cls, value: str) -> str:
        try:
            return str(uuid.UUID(value))
        except ValueError as error:
            raise ValueError("episode_id must be a UUID") from error

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        try:
            return str(uuid.UUID(value))
        except ValueError as error:
            raise ValueError("run_id must be a UUID") from error

    @field_validator("backend", "policy")
    @classmethod
    def validate_plugin_names(cls, value: str, info: object) -> str:
        return _identifier(value, getattr(info, "field_name", "plugin"))

    @field_validator("backend_version", "policy_version")
    @classmethod
    def validate_plugin_versions(cls, value: str) -> str:
        return _semver(value)

    @model_validator(mode="after")
    def validate_episode(self) -> Episode:
        if len(self.steps) < 2:
            raise ValueError("an episode requires an initial and final observation")
        if [item.index for item in self.steps] != list(range(len(self.steps))):
            raise ValueError("episode step indices must be contiguous from zero")
        if not self.steps[0].is_first or any(item.is_first for item in self.steps[1:]):
            raise ValueError("only the first episode step may set is_first")
        if not self.steps[-1].is_last or any(item.is_last for item in self.steps[:-1]):
            raise ValueError("only the final episode step may set is_last")
        timestamps = [item.timestamp_seconds for item in self.steps]
        if any(right <= left for left, right in _pairwise(tuple(timestamps))):
            raise ValueError("episode timestamps must be strictly increasing")
        sensor_timestamps: dict[tuple[str, str], float] = {}
        for step in self.steps:
            for channel, values in (("observation", step.observations), ("state", step.state)):
                for name, value in values.items():
                    if value.timestamp_seconds is None:
                        continue
                    key = (channel, name)
                    previous = sensor_timestamps.get(key)
                    if previous is not None and value.timestamp_seconds < previous:
                        raise ValueError("sensor timestamps must be monotonic per signal")
                    sensor_timestamps[key] = value.timestamp_seconds
        if self.success and self.failure_reason is not None:
            raise ValueError("successful episodes cannot have a failure_reason")
        if not self.success and not self.failure_reason:
            raise ValueError("failed episodes require a failure_reason")
        if self.success and not self.steps[-1].is_terminal:
            raise ValueError("successful episodes must end in a terminal observation")
        if not self.scene.resolved or not self.task.resolved:
            raise ValueError("episodes require resolved scene and task references")
        if self.scene.resource_type is not ResourceType.SCENE:
            raise ValueError("Episode.scene must reference a scene")
        if self.task.resource_type is not ResourceType.TASK:
            raise ValueError("Episode.task must reference a task")

        observation_keys = set(self.steps[0].observations)
        state_keys = set(self.steps[0].state)
        if not observation_keys:
            raise ValueError("episode steps require at least one observation")

        def schema(value: DataValue) -> tuple[SignalDType, tuple[int, ...], str, str]:
            return value.dtype, value.shape, value.unit, value.frame

        observation_schema = {
            name: schema(value) for name, value in self.steps[0].observations.items()
        }
        state_schema = {name: schema(value) for name, value in self.steps[0].state.items()}
        action_schema: tuple[SignalDType, tuple[int, ...], str, str] | None = None
        for step in self.steps:
            if set(step.observations) != observation_keys or set(step.state) != state_keys:
                raise ValueError("episode signal sets must remain constant across steps")
            if any(
                schema(value) != observation_schema[name]
                for name, value in step.observations.items()
            ) or any(schema(value) != state_schema[name] for name, value in step.state.items()):
                raise ValueError("episode signal schemas must remain constant across steps")
            if step.action is not None:
                current_action_schema = schema(step.action)
                if action_schema is None:
                    action_schema = current_action_schema
                elif current_action_schema != action_schema:
                    raise ValueError("episode action schema must remain constant across steps")
        return self

    @property
    def digest(self) -> str:
        return sha256_digest(self)


class EpisodeSummary(DomainModel):
    episode_id: str
    run_id: str
    index: int = Field(ge=0)
    success: bool
    step_count: int = Field(gt=1)
    digest: str
    artifact: ArtifactRef | None = None
    failure_reason: str | None = None

    @field_validator("digest")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        if _DIGEST.fullmatch(value) is None:
            raise ValueError("episode digest must be a SHA-256 digest")
        return value

    @field_validator("episode_id", "run_id")
    @classmethod
    def validate_ids(cls, value: str, info: object) -> str:
        try:
            return str(uuid.UUID(value))
        except ValueError as error:
            raise ValueError(f"{getattr(info, 'field_name', 'id')} must be a UUID") from error

    @model_validator(mode="after")
    def validate_summary(self) -> EpisodeSummary:
        if self.success == (self.failure_reason is not None):
            raise ValueError("only failed episode summaries may have a failure_reason")
        if self.artifact is not None and self.artifact.digest != self.digest:
            raise ValueError("episode summary and artifact digests must match")
        return self


class DatasetManifest(DomainModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    resource_id: str
    version: str
    episodes: tuple[EpisodeSummary, ...]
    source_runs: tuple[str, ...]
    observation_schema: tuple[SignalSpec, ...]
    action_schema: SignalSpec
    total_steps: int = Field(gt=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("resource_id")
    @classmethod
    def validate_resource_id(cls, value: str) -> str:
        return _identifier(value, "resource_id")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _semver(value)

    @model_validator(mode="after")
    def validate_manifest(self) -> DatasetManifest:
        if not self.episodes:
            raise ValueError("a dataset requires at least one episode")
        if not self.observation_schema:
            raise ValueError("a dataset requires at least one observation signal")
        if len({item.episode_id for item in self.episodes}) != len(self.episodes):
            raise ValueError("dataset episode ids must be unique")
        if sum(item.step_count for item in self.episodes) != self.total_steps:
            raise ValueError("total_steps does not match episode summaries")
        if not self.source_runs or len(set(self.source_runs)) != len(self.source_runs):
            raise ValueError("dataset source_runs must be non-empty and unique")
        try:
            source_runs = {str(uuid.UUID(run_id)) for run_id in self.source_runs}
        except ValueError as error:
            raise ValueError("dataset source_runs must contain UUIDs") from error
        if any(item.run_id not in source_runs for item in self.episodes):
            raise ValueError("every dataset episode must belong to a source run")
        if any(item.artifact is None for item in self.episodes):
            raise ValueError("published datasets require an artifact for every episode")
        observation_names = [item.name for item in self.observation_schema]
        if len(observation_names) != len(set(observation_names)):
            raise ValueError("dataset observation signal names must be unique")
        return self

    def reference(self, digest: str | None = None) -> VersionedRef:
        return VersionedRef(
            resource_type=ResourceType.DATASET,
            resource_id=self.resource_id,
            version=self.version,
            digest=digest,
        )


class ModelManifest(DomainModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    resource_id: str
    version: str
    trainer: str
    trainer_version: str
    policy: str
    policy_version: str
    dataset: VersionedRef
    weights: ArtifactRef
    observation_signals: tuple[SignalSpec, ...]
    action_signal: SignalSpec
    training_config: dict[str, JsonValue]
    metrics: dict[str, float]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("resource_id", "trainer", "policy")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        return _identifier(value, getattr(info, "field_name", "identifier"))

    @field_validator("version", "trainer_version", "policy_version")
    @classmethod
    def validate_versions(cls, value: str) -> str:
        return _semver(value)

    @model_validator(mode="after")
    def validate_model(self) -> ModelManifest:
        if self.dataset.resource_type is not ResourceType.DATASET or not self.dataset.resolved:
            raise ValueError("model dataset must be a resolved dataset reference")
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("model metrics must be finite")
        names = [item.name for item in self.observation_signals]
        if not names or len(names) != len(set(names)):
            raise ValueError("model observation signals must be non-empty and unique")
        return self

    def reference(self, digest: str | None = None) -> VersionedRef:
        return VersionedRef(
            resource_type=ResourceType.MODEL,
            resource_id=self.resource_id,
            version=self.version,
            digest=digest,
        )


class TrainingSpec(DomainModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    resource_id: str
    version: str
    dataset: VersionedRef
    trainer: str
    config: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("resource_id", "trainer")
    @classmethod
    def validate_identifiers(cls, value: str, info: object) -> str:
        return _identifier(value, getattr(info, "field_name", "identifier"))

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _semver(value)

    @model_validator(mode="after")
    def validate_training(self) -> TrainingSpec:
        if self.dataset.resource_type is not ResourceType.DATASET or not self.dataset.resolved:
            raise ValueError("training requires a resolved dataset reference")
        return self


class FailureCategory(StrEnum):
    PERCEPTION = "perception"
    PLANNING = "planning"
    CONTROL = "control"
    CONTACT = "contact"
    TIMEOUT = "timeout"
    DATA_QUALITY = "data_quality"
    UNKNOWN = "unknown"


class FailureCase(DomainModel):
    episode_id: str
    category: FailureCategory
    message: str = Field(min_length=1, max_length=1024)

    @field_validator("episode_id")
    @classmethod
    def validate_episode_id(cls, value: str) -> str:
        try:
            return str(uuid.UUID(value))
        except ValueError as error:
            raise ValueError("failure episode_id must be a UUID") from error


class EvaluationReport(DomainModel):
    schema_version: Literal[1] = SCHEMA_VERSION
    resource_id: str
    version: str
    model: VersionedRef
    dataset: VersionedRef | None = None
    success_rate: UnitFloat
    metrics: dict[str, float]
    failures: tuple[FailureCase, ...]
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("resource_id")
    @classmethod
    def validate_resource_id(cls, value: str) -> str:
        return _identifier(value, "resource_id")

    @field_validator("version")
    @classmethod
    def validate_version(cls, value: str) -> str:
        return _semver(value)

    @model_validator(mode="after")
    def validate_report(self) -> EvaluationReport:
        if self.model.resource_type is not ResourceType.MODEL or not self.model.resolved:
            raise ValueError("evaluation model must be a resolved model reference")
        if self.dataset is not None and (
            self.dataset.resource_type is not ResourceType.DATASET or not self.dataset.resolved
        ):
            raise ValueError("evaluation dataset must be a resolved dataset reference")
        if any(not math.isfinite(value) for value in self.metrics.values()):
            raise ValueError("evaluation metrics must be finite")
        if len({item.episode_id for item in self.failures}) != len(self.failures):
            raise ValueError("evaluation failure episode ids must be unique")
        return self

    def reference(self, digest: str | None = None) -> VersionedRef:
        return VersionedRef(
            resource_type=ResourceType.EVALUATION,
            resource_id=self.resource_id,
            version=self.version,
            digest=digest,
        )


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunResult(DomainModel):
    run_id: str
    status: RunStatus
    episode_count: int = Field(ge=0)
    dataset: VersionedRef | None = None
    error_code: str | None = None
    error_message: str | None = Field(default=None, max_length=2048)

    @field_validator("run_id")
    @classmethod
    def validate_run_id(cls, value: str) -> str:
        try:
            return str(uuid.UUID(value))
        except ValueError as error:
            raise ValueError("run_id must be a UUID") from error

    @field_validator("error_code")
    @classmethod
    def validate_error_code(cls, value: str | None) -> str | None:
        return _identifier(value, "error_code") if value is not None else None

    @model_validator(mode="after")
    def validate_result(self) -> RunResult:
        terminal = {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
        if self.status not in terminal:
            raise ValueError("RunResult status must be terminal")
        if self.status is RunStatus.SUCCEEDED:
            if (
                self.dataset is None
                or self.dataset.resource_type is not ResourceType.DATASET
                or not self.dataset.resolved
            ):
                raise ValueError("a successful run requires a dataset reference")
            if self.error_code is not None or self.error_message is not None:
                raise ValueError("a successful run cannot contain an error")
        elif not self.error_code or not self.error_message:
            raise ValueError("failed or cancelled runs require a structured error")
        return self


class RunRecord(DomainModel):
    run_id: str
    spec: RunSpec
    status: RunStatus
    result: RunResult | None
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_record(self) -> RunRecord:
        if self.run_id != self.spec.run_id:
            raise ValueError("RunRecord.run_id must match RunSpec.run_id")
        terminal = self.status in {
            RunStatus.SUCCEEDED,
            RunStatus.FAILED,
            RunStatus.CANCELLED,
        }
        if terminal != (self.result is not None):
            raise ValueError("only terminal runs may contain a result")
        if self.result is not None and self.result.status is not self.status:
            raise ValueError("RunResult status must match RunRecord status")
        if self.updated_at < self.created_at:
            raise ValueError("RunRecord.updated_at cannot precede created_at")
        if self.status is RunStatus.PENDING and (
            self.started_at is not None or self.completed_at is not None
        ):
            raise ValueError("pending runs cannot have lifecycle timestamps")
        if self.status is RunStatus.RUNNING and (
            self.started_at is None or self.completed_at is not None
        ):
            raise ValueError("running runs require started_at and no completed_at")
        if terminal and (self.started_at is None or self.completed_at is None):
            raise ValueError("terminal runs require started_at and completed_at")
        if self.started_at is not None and self.started_at < self.created_at:
            raise ValueError("RunRecord.started_at cannot precede created_at")
        if self.completed_at is not None and (
            self.started_at is None or self.completed_at < self.started_at
        ):
            raise ValueError("RunRecord.completed_at cannot precede started_at")
        return self


__all__ = [
    "ActionSpec",
    "ArtifactRef",
    "AssetSpec",
    "BodyPhysicsSpec",
    "DataValue",
    "DatasetManifest",
    "Episode",
    "EpisodeSource",
    "EpisodeStep",
    "EpisodeSummary",
    "EvaluationReport",
    "FailureCase",
    "FailureCategory",
    "GeometryRepresentation",
    "GeometryRole",
    "GeometrySpec",
    "JSONValue",
    "JointSpec",
    "JointType",
    "ModelManifest",
    "ParameterSource",
    "PoseSpec",
    "Predicate",
    "PredicateOperator",
    "ResourceType",
    "RunRecord",
    "RunResult",
    "RunSpec",
    "RunStatus",
    "ScalarParameter",
    "SceneEntityRole",
    "SceneEntitySpec",
    "SceneSpec",
    "SignalDType",
    "SignalSpec",
    "TaskSpec",
    "TrainingSpec",
    "VersionedRef",
    "canonical_json_bytes",
    "sha256_digest",
]
