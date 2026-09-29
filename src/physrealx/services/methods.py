"""Method discovery, binding and canonical capability requests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from physrealx.domain import (
    ArtifactRef,
    JSONValue,
    MethodCapability,
    MethodDescriptor,
    MethodSpec,
    OperationInput,
    OperationManifest,
    OperationSpec,
    VersionedRef,
    canonical_record_id,
)
from physrealx.errors import (
    DomainValidationError,
)
from physrealx.plugins import MethodInstallation

if TYPE_CHECKING:
    from physrealx.platform import PhysRealX


from physrealx.services._method_contracts import SERVICE_METHOD_CONTRACTS


class MethodsService:
    """Inspect and bind the exact implementations installed in this workspace."""

    def __init__(self, platform: PhysRealX) -> None:
        self._platform = platform

    def register(self, descriptor: MethodDescriptor) -> None:
        self._platform.plugins.register_method(descriptor)

    def get(self, capability: MethodCapability, name: str) -> MethodDescriptor:
        return self._platform.plugins.method_descriptor(capability, self._canonical_name(name))

    @staticmethod
    def _canonical_name(name: str) -> str:
        """Normalize a human-entered selector before immutable method binding."""

        canonical = name.strip().casefold()
        if not canonical:
            raise DomainValidationError("method name cannot be empty")
        return canonical

    def list(
        self,
        capability: MethodCapability | None = None,
    ) -> tuple[MethodDescriptor, ...]:
        return self._platform.plugins.descriptors(capability)

    def inspect(self, capability: MethodCapability, name: str) -> MethodInstallation:
        return self._platform.plugins.installation(capability, self._canonical_name(name))

    def installations(
        self,
        capability: MethodCapability | None = None,
    ) -> tuple[MethodInstallation, ...]:
        return self._platform.plugins.installations(capability)

    def bind(
        self,
        capability: MethodCapability,
        name: str,
        *,
        checkpoint: ArtifactRef | None = None,
        config: Mapping[str, JSONValue] | None = None,
    ) -> MethodSpec:
        descriptor = self.get(capability, name)
        if checkpoint is not None and not self._platform.artifacts.verify(checkpoint):
            raise DomainValidationError("method checkpoint is not present in artifact storage")
        return descriptor.bind(checkpoint=checkpoint, config=config)

    def execute(
        self,
        capability: MethodCapability,
        method: str | MethodSpec,
        *,
        inputs: Mapping[str, VersionedRef | ArtifactRef] | None = None,
        config: Mapping[str, JSONValue] | None = None,
        arguments: Mapping[str, JSONValue] | None = None,
        resource_id: str | None = None,
        version: str = "1.0.0",
    ) -> OperationManifest:
        """Execute the canonical contract owned by one public capability service."""

        if isinstance(method, MethodSpec) and config is not None:
            raise DomainValidationError(
                "config must be bound in MethodSpec when an explicit MethodSpec is supplied"
            )
        selected = (
            self.bind(capability, method, config=config) if isinstance(method, str) else method
        )
        if selected.capability is not capability:
            raise DomainValidationError(
                f"service requires capability {capability.value!r}, got "
                f"{selected.capability.value!r}"
            )
        try:
            service_contract = SERVICE_METHOD_CONTRACTS[capability]
        except KeyError as error:
            raise DomainValidationError(
                f"capability {capability.value!r} has no public service contract; "
                "execute it through an explicit method recipe"
            ) from error
        descriptor = selected.descriptor
        if descriptor.input_contract != service_contract.input_contract:
            raise DomainValidationError(
                f"method {capability.value}:{selected.name!r} declares input contract "
                f"{descriptor.input_contract!r}; the service requires "
                f"{service_contract.input_contract!r}"
            )
        declared_outputs = tuple(descriptor.output_contract.split("+"))
        required_outputs = tuple(item.contract for item in service_contract.outputs)
        if sorted(declared_outputs) != sorted(required_outputs):
            raise DomainValidationError(
                f"method {capability.value}:{selected.name!r} declares output contracts "
                f"{declared_outputs!r}; the service requires {required_outputs!r}"
            )
        supplied = inputs or {}
        supplied_inputs = tuple(supplied)
        if set(supplied_inputs) != set(service_contract.input_names):
            raise DomainValidationError(
                f"service {capability.value!r} requires inputs "
                f"{service_contract.input_names!r}, got {supplied_inputs!r}"
            )
        for name, value, expected_kind in zip(
            service_contract.input_names,
            (supplied[name] for name in service_contract.input_names),
            service_contract.input_kinds,
        ):
            actual_kind = "resource" if isinstance(value, VersionedRef) else "artifact"
            if expected_kind != "either" and actual_kind != expected_kind:
                raise DomainValidationError(
                    f"service input {name!r} must be an immutable {expected_kind}, "
                    f"got {actual_kind}"
                )
        self._platform.plugins.resolve_method(selected)
        operation_inputs = tuple(
            OperationInput(
                name=name,
                resource=supplied[name] if isinstance(supplied[name], VersionedRef) else None,
                artifact=supplied[name] if isinstance(supplied[name], ArtifactRef) else None,
            )
            for name in service_contract.input_names
        )
        output_requirements = service_contract.outputs
        identity_payload: dict[str, JSONValue] = {
            "method": selected.model_dump(mode="json"),
            "inputs": [item.model_dump(mode="json") for item in operation_inputs],
            "arguments": dict(arguments or {}),
            "outputs": [item.model_dump(mode="json") for item in output_requirements],
        }
        identity = resource_id or (
            f"operation.{capability.value}.{canonical_record_id(identity_payload)[-16:]}"
        )
        return self._platform.operations.execute(
            OperationSpec(
                resource_id=identity,
                version=version,
                method=selected,
                inputs=operation_inputs,
                arguments=dict(arguments or {}),
                outputs=output_requirements,
            )
        )
