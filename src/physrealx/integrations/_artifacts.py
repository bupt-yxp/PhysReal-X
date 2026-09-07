"""Small, shared I/O boundary for scientific capability adapters."""

from __future__ import annotations

import io
from importlib import import_module
from importlib.metadata import version
from typing import Any, TypeVar

from pydantic import BaseModel

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import ArtifactRef, OperationOutput, OperationSpec, VersionedRef
from physrealx.errors import DomainValidationError
from physrealx.evidence import strict_json_object


def dependency_probe(requirements: dict[str, tuple[str, str]]) -> MethodProbe:
    try:
        for module, (distribution, expected) in requirements.items():
            actual = version(distribution)
            if actual != expected:
                return MethodProbe(False, f"{distribution} requires {expected}, found {actual}")
            import_module(module)
    except (ImportError, OSError) as error:
        return MethodProbe(False, str(error))
    return MethodProbe(True, "Pinned dependencies imported successfully")


def input_artifact(spec: OperationSpec, name: str, store: ArtifactStore) -> ArtifactRef:
    matches = [item for item in spec.inputs if item.name == name]
    if len(matches) != 1 or matches[0].artifact is None:
        raise DomainValidationError(f"{name} requires one immutable ArtifactRef")
    reference = matches[0].artifact
    if not store.verify(reference):
        raise DomainValidationError(f"{name} artifact is unavailable")
    return reference


_ModelT = TypeVar("_ModelT", bound=BaseModel)


def load_resource(store: ArtifactStore, reference: VersionedRef, model: type[_ModelT]) -> _ModelT:
    """Resolve a pinned resource without a second catalog or path-based convention."""
    if reference.artifact_id is None or not store.verify(reference.artifact_id):
        raise DomainValidationError("resource requires an available immutable artifact identity")
    result = model.model_validate_json(store.get_bytes(reference.artifact_id))
    if (getattr(result, "resource_id", None), getattr(result, "version", None)) != (
        reference.resource_id,
        reference.version,
    ):
        raise DomainValidationError("resource identity differs from its pinned payload")
    return result


def input_model(
    spec: OperationSpec, name: str, store: ArtifactStore, model: type[_ModelT]
) -> _ModelT:
    matches = [item for item in spec.inputs if item.name == name]
    if len(matches) != 1:
        raise DomainValidationError(f"missing unique input: {name}")
    item = matches[0]
    if item.resource is not None:
        return load_resource(store, item.resource, model)
    if item.artifact is None:
        raise DomainValidationError(f"{name} requires an immutable input")
    return model.model_validate_json(store.get_bytes(item.artifact))


def read_json(store: ArtifactStore, reference: ArtifactRef, schema: str) -> dict[str, Any]:
    if reference.size_bytes > 64 * 1024 * 1024:
        raise DomainValidationError("JSON evidence exceeds 64 MiB")
    try:
        value = strict_json_object(store.get_bytes(reference), schema)
    except (ValueError, UnicodeError) as error:
        raise DomainValidationError("invalid JSON evidence") from error
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise DomainValidationError(f"evidence requires schema {schema}")
    return value


def read_array(store: ArtifactStore, reference: ArtifactRef) -> Any:
    import numpy as np

    if reference.media_type != "application/x-npy" or reference.size_bytes > 512 * 1024 * 1024:
        raise DomainValidationError("array requires an NPY artifact of at most 512 MiB")
    try:
        array = np.load(io.BytesIO(store.get_bytes(reference)), allow_pickle=False)
    except (ValueError, OSError) as error:
        raise DomainValidationError("invalid numeric NPY artifact") from error
    if not isinstance(array, np.ndarray) or array.dtype.kind not in "fiu":
        raise DomainValidationError("array must have a real numeric dtype")
    if not np.isfinite(array).all():
        raise DomainValidationError("array contains non-finite values")
    return array


def publish(
    spec: OperationSpec,
    store: ArtifactStore,
    documents: dict[str, object],
    *,
    artifacts: dict[str, ArtifactRef] | None = None,
    metrics: dict[str, float] | None = None,
) -> MethodExecutionOutput:
    references = dict(artifacts or {})
    if set(documents) & set(references) or set(documents) | set(references) != {
        item.name for item in spec.outputs
    }:
        raise DomainValidationError("adapter output names differ from the canonical contract")
    references.update({name: store.put_json(value) for name, value in documents.items()})
    return MethodExecutionOutput(
        outputs=tuple(
            OperationOutput(name=item.name, contract=item.contract, artifact=references[item.name])
            for item in spec.outputs
        ),
        metrics=metrics or {},
    )
