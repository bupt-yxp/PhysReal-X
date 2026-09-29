"""Immutable operation execution and publication."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from physrealx.contracts import MethodExecutionOutput
from physrealx.domain import (
    OperationManifest,
    OperationOutput,
    OperationSpec,
    ResourceType,
    VersionedRef,
)
from physrealx.errors import (
    DomainValidationError,
    MethodExecutionError,
    PhysRealXError,
    ResourceConflictError,
    ResourceNotFoundError,
)
from physrealx.operation_validation import validate_operation_output

if TYPE_CHECKING:
    from physrealx.platform import PhysRealX


class OperationsService:
    """Execute capability methods and publish immutable, validated evidence."""

    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def get(self, reference: VersionedRef) -> OperationManifest:
        if reference.resource_type is not ResourceType.OPERATION:
            raise DomainValidationError("operations.get requires an operation reference")
        return self._platform._load_registered(reference, OperationManifest)

    def list(self) -> list[VersionedRef]:
        return self._platform._list_resources(ResourceType.OPERATION)

    def execute(self, spec: OperationSpec) -> OperationManifest:
        """Execute and atomically publish one capability-method invocation."""

        try:
            existing_ref = self._platform.repository.get_resource(
                ResourceType.OPERATION.value,
                spec.resource_id,
                spec.version,
            )
        except ResourceNotFoundError:
            pass
        else:
            existing = self._platform._load_registered(existing_ref, OperationManifest)
            if existing.spec != spec:
                raise ResourceConflictError(
                    "operation identity already exists for a different specification"
                )
            if any(
                not self._platform.artifacts.verify(output.artifact) for output in existing.outputs
            ):
                raise DomainValidationError(
                    "operation references an unavailable or corrupt output artifact"
                )
            return existing

        resolved_inputs = []
        for item in spec.inputs:
            if item.resource is not None:
                registered = self._platform.repository.get_resource(
                    item.resource.resource_type.value,
                    item.resource.resource_id,
                    item.resource.version,
                )
                if registered.artifact_id != item.resource.artifact_id:
                    raise ResourceConflictError(
                        f"operation input {item.name!r} does not match the registry"
                    )
                resolved_inputs.append(item.model_copy(update={"resource": registered}))
            else:
                if item.artifact is None or not self._platform.artifacts.verify(item.artifact):
                    raise DomainValidationError(
                        f"operation input artifact {item.name!r} is unavailable or corrupt"
                    )
                resolved_inputs.append(item)

        resolved = spec.model_copy(update={"inputs": tuple(resolved_inputs)})
        adapter = self._platform.plugins.adapter(resolved.method)
        checkpoint = resolved.method.checkpoint
        if checkpoint is not None and not self._platform.artifacts.verify(checkpoint):
            raise DomainValidationError("method checkpoint is unavailable or corrupt")

        started_at = datetime.now(timezone.utc)
        try:
            result = adapter.execute(resolved, self._platform.artifacts)
        except PhysRealXError:
            raise
        except Exception as error:
            raise MethodExecutionError(
                f"method {resolved.method.capability.value}:{resolved.method.name!r} failed"
            ) from error
        completed_at = datetime.now(timezone.utc)
        if not isinstance(result, MethodExecutionOutput) or any(
            not isinstance(item, OperationOutput) for item in result.outputs
        ):
            raise MethodExecutionError("method adapter returned an incompatible result")
        expected_outputs = tuple((item.name, item.contract) for item in resolved.outputs)
        actual_outputs = tuple((item.name, item.contract) for item in result.outputs)
        if actual_outputs != expected_outputs:
            raise MethodExecutionError(
                "method adapter output names or contracts do not match the operation"
            )
        if any(not self._platform.artifacts.verify(item.artifact) for item in result.outputs):
            raise MethodExecutionError("method adapter returned unpublished or invalid artifacts")
        for requirement, output in zip(resolved.outputs, result.outputs):
            validate_operation_output(self._platform.artifacts, output, requirement)

        try:
            manifest = OperationManifest(
                resource_id=resolved.resource_id,
                version=resolved.version,
                spec=resolved,
                outputs=result.outputs,
                metrics=dict(result.metrics),
                started_at=started_at,
                completed_at=completed_at,
            )
        except ValueError as error:
            raise MethodExecutionError(
                "method adapter returned invalid execution evidence"
            ) from error
        self._platform._register(
            manifest,
            manifest.reference(),
            "application/vnd.physrealx.operation+json",
        )
        return manifest
