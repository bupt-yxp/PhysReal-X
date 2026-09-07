"""Capability-oriented plugin and method registry."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Callable
from typing import Mapping as TypingMapping

from typing_extensions import TypeAlias

from physrealx.contracts import (
    API_VERSION,
    MethodAdapter,
    MethodProbe,
    ObservationImporter,
    Policy,
    SimulationBackend,
    Trainer,
)
from physrealx.domain import (
    JSONValue,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodSpec,
)
from physrealx.errors import (
    MethodUnavailableError,
    PluginError,
    ResourceConflictError,
    ResourceNotFoundError,
)

PolicyFactory: TypeAlias = Callable[[TypingMapping[str, JSONValue]], Policy]
ENTRY_POINT_GROUP = "physrealx.methods"


@dataclass(frozen=True)
class MethodInstallation:
    """Current local execution state for one immutable method descriptor."""

    descriptor: MethodDescriptor
    execution_kind: str
    executable: bool
    ready: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = dict(self.descriptor.model_dump(mode="json"))
        value.update(
            {
                "execution_kind": self.execution_kind,
                "executable": self.executable,
                "ready": self.ready,
                "detail": self.detail,
            }
        )
        return value


class PluginRegistry:
    """Application-owned registry of executable adapters and method metadata.

    Typed accessors keep the runtime boundary explicit, while the shared method
    catalog provides one auditable identity model across every capability.
    Importing :mod:`physrealx` never creates a process-global registry.
    """

    def __init__(self) -> None:
        self._descriptors: dict[tuple[MethodCapability, str], MethodDescriptor] = {}
        self._backends: dict[str, SimulationBackend] = {}
        self._importers: dict[str, ObservationImporter] = {}
        self._policies: dict[str, tuple[MethodDescriptor, PolicyFactory]] = {}
        self._trainers: dict[str, Trainer] = {}
        self._adapters: dict[tuple[MethodCapability, str], MethodAdapter] = {}
        self._execution_kinds: dict[tuple[MethodCapability, str], str] = {}

    @staticmethod
    def _validate_descriptor(
        descriptor: MethodDescriptor,
        expected: MethodCapability | None = None,
    ) -> None:
        if descriptor.api_version != API_VERSION:
            raise PluginError(
                f"method API version {descriptor.api_version!r} is incompatible with "
                f"{API_VERSION!r}"
            )
        if expected is not None and descriptor.capability is not expected:
            raise PluginError(
                f"registration requires capability {expected.value!r}, got "
                f"{descriptor.capability.value!r}"
            )

    @staticmethod
    def _descriptor(plugin: object, expected: MethodCapability) -> MethodDescriptor:
        descriptor = getattr(plugin, "descriptor", None)
        if not isinstance(descriptor, MethodDescriptor):
            raise PluginError("plugin must expose a MethodDescriptor as 'descriptor'")
        PluginRegistry._validate_descriptor(descriptor, expected)
        return descriptor

    def _record(self, descriptor: MethodDescriptor) -> None:
        self._validate_descriptor(descriptor)
        key = (descriptor.capability, descriptor.name)
        if key in self._descriptors:
            raise ResourceConflictError(
                f"method {descriptor.capability.value}:{descriptor.name!r} is already registered"
            )
        self._descriptors[key] = descriptor

    def _record_executable(self, descriptor: MethodDescriptor) -> None:
        """Publish an executable descriptor or match prior metadata exactly."""

        self._validate_descriptor(descriptor)
        key = (descriptor.capability, descriptor.name)
        existing = self._descriptors.get(key)
        if existing is None:
            self._descriptors[key] = descriptor
        elif existing != descriptor:
            raise ResourceConflictError(
                f"method {descriptor.capability.value}:{descriptor.name!r} metadata "
                "does not match the executable method"
            )

    @staticmethod
    def _require_in_process(descriptor: MethodDescriptor, adapter_kind: str) -> None:
        if descriptor.runtime.mode is not MethodExecutionMode.IN_PROCESS:
            raise PluginError(
                f"{adapter_kind} uses an in-process protocol and cannot declare "
                f"runtime mode {descriptor.runtime.mode.value!r}"
            )

    def register_method(self, descriptor: MethodDescriptor) -> None:
        """Register metadata for a method that is not executable in this installation."""

        self._record(descriptor)

    def register_adapter(self, adapter: MethodAdapter) -> None:
        """Register one executable method adapter."""

        if not isinstance(adapter, MethodAdapter):
            raise PluginError("method adapter does not implement the required protocol")
        descriptor = self._descriptor(adapter, adapter.descriptor.capability)
        self._require_in_process(descriptor, "method adapter")
        key = (descriptor.capability, descriptor.name)
        if key in self._adapters:
            raise ResourceConflictError(
                f"method adapter {descriptor.capability.value}:{descriptor.name!r} "
                "is already registered"
            )
        self._record_executable(descriptor)
        self._adapters[key] = adapter
        self._execution_kinds[key] = "method_adapter"

    def register_backend(self, backend: SimulationBackend) -> None:
        if not isinstance(backend, SimulationBackend):
            raise PluginError("simulation backend does not implement the required protocol")
        descriptor = self._descriptor(backend, MethodCapability.SIMULATION_RUN)
        self._require_in_process(descriptor, "simulation backend")
        if descriptor.name in self._backends:
            raise ResourceConflictError(
                f"simulation backend {descriptor.name!r} is already registered"
            )
        self._record(descriptor)
        self._backends[descriptor.name] = backend
        self._execution_kinds[(descriptor.capability, descriptor.name)] = "simulation_backend"

    def register_importer(self, importer: ObservationImporter) -> None:
        if not isinstance(importer, ObservationImporter):
            raise PluginError("observation importer does not implement the required protocol")
        descriptor = self._descriptor(importer, MethodCapability.OBSERVATION_IMPORT)
        self._require_in_process(descriptor, "observation importer")
        if descriptor.name in self._importers:
            raise ResourceConflictError(
                f"observation importer {descriptor.name!r} is already registered"
            )
        self._record(descriptor)
        self._importers[descriptor.name] = importer
        self._execution_kinds[(descriptor.capability, descriptor.name)] = "observation_importer"

    def register_policy(
        self,
        descriptor: MethodDescriptor,
        factory: PolicyFactory,
    ) -> None:
        self._validate_descriptor(descriptor, MethodCapability.POLICY_INFERENCE)
        self._require_in_process(descriptor, "policy factory")
        if not callable(factory):
            raise PluginError("policy factory must be callable")
        if descriptor.name in self._policies:
            raise ResourceConflictError(f"policy {descriptor.name!r} is already registered")
        self._record(descriptor)
        self._policies[descriptor.name] = (descriptor, factory)
        self._execution_kinds[(descriptor.capability, descriptor.name)] = "policy_factory"

    def register_trainer(self, trainer: Trainer) -> None:
        if not isinstance(trainer, Trainer):
            raise PluginError("trainer does not implement the required protocol")
        descriptor = self._descriptor(trainer, MethodCapability.POLICY_TRAIN)
        self._require_in_process(descriptor, "trainer")
        if descriptor.name in self._trainers:
            raise ResourceConflictError(f"trainer {descriptor.name!r} is already registered")
        self._record(descriptor)
        self._trainers[descriptor.name] = trainer
        self._execution_kinds[(descriptor.capability, descriptor.name)] = "trainer"

    def backend(self, name: str) -> SimulationBackend:
        try:
            backend = self._backends[name]
        except KeyError as error:
            raise ResourceNotFoundError(f"simulation backend {name!r} is not registered") from error
        probe = self._probe(backend)
        if not probe.ready:
            raise MethodUnavailableError(
                f"simulation backend {name!r} is unavailable", details={"reason": probe.detail}
            )
        return backend

    def importer(self, name: str) -> ObservationImporter:
        try:
            return self._importers[name]
        except KeyError as error:
            raise ResourceNotFoundError(
                f"observation importer {name!r} is not registered"
            ) from error

    def policy(self, name: str, config: Mapping[str, JSONValue]) -> Policy:
        try:
            descriptor, factory = self._policies[name]
        except KeyError as error:
            raise ResourceNotFoundError(f"policy {name!r} is not registered") from error
        policy = factory(config)
        if not isinstance(policy, Policy):
            raise PluginError(f"policy factory {name!r} returned an incompatible object")
        if policy.descriptor != descriptor:
            raise PluginError(f"policy factory {name!r} returned mismatched method identity")
        return policy

    def policy_descriptor(self, name: str) -> MethodDescriptor:
        try:
            descriptor, _ = self._policies[name]
        except KeyError as error:
            raise ResourceNotFoundError(f"policy {name!r} is not registered") from error
        return descriptor

    def trainer(self, name: str) -> Trainer:
        try:
            return self._trainers[name]
        except KeyError as error:
            raise ResourceNotFoundError(f"trainer {name!r} is not registered") from error

    def method_descriptor(
        self,
        capability: MethodCapability,
        name: str,
    ) -> MethodDescriptor:
        try:
            return self._descriptors[(capability, name)]
        except KeyError as error:
            raise ResourceNotFoundError(
                f"method {capability.value}:{name!r} is not registered"
            ) from error

    def resolve_method(self, method: MethodSpec) -> MethodDescriptor:
        installed = self.method_descriptor(method.capability, method.name)
        if installed != method.descriptor:
            raise ResourceConflictError(
                f"method {method.capability.value}:{method.name!r} does not match the "
                "installed immutable descriptor"
            )
        return installed

    def adapter(self, method: MethodSpec) -> MethodAdapter:
        """Resolve an executable adapter and verify its full immutable identity."""

        self.resolve_method(method)
        key = (method.capability, method.name)
        try:
            adapter = self._adapters[key]
        except KeyError as error:
            raise ResourceNotFoundError(
                f"method {method.capability.value}:{method.name!r} is registered as "
                "metadata only; install its optional dependency and register its implementation"
            ) from error
        probe = self._probe(adapter)
        if not probe.ready:
            raise MethodUnavailableError(
                f"method {method.capability.value}:{method.name!r} is installed but unavailable",
                details={"reason": probe.detail},
            )
        return adapter

    @staticmethod
    def _probe(adapter: object) -> MethodProbe:
        probe_method = getattr(adapter, "probe", None)
        if not callable(probe_method):
            return MethodProbe(True, "in-process method adapter is registered")
        try:
            result = probe_method()
        except Exception as error:
            return MethodProbe(False, f"readiness probe failed: {error}")
        if not isinstance(result, MethodProbe):
            return MethodProbe(False, "readiness probe returned an incompatible result")
        return result

    def installation(
        self,
        capability: MethodCapability,
        name: str,
    ) -> MethodInstallation:
        descriptor = self.method_descriptor(capability, name)
        key = (capability, name)
        execution_kind = self._execution_kinds.get(key)
        if execution_kind is None:
            return MethodInstallation(
                descriptor=descriptor,
                execution_kind="metadata",
                executable=False,
                ready=False,
                detail="descriptor is registered without an executable adapter",
            )
        adapter: object = self._adapters.get(key)
        if adapter is None:
            if execution_kind == "simulation_backend":
                adapter = self._backends.get(name)
            elif execution_kind == "observation_importer":
                adapter = self._importers.get(name)
            elif execution_kind == "trainer":
                adapter = self._trainers.get(name)
        probe = (
            self._probe(adapter) if adapter is not None else MethodProbe(True, "adapter is ready")
        )
        return MethodInstallation(
            descriptor=descriptor,
            execution_kind=execution_kind,
            executable=True,
            ready=probe.ready,
            detail=probe.detail,
        )

    def installations(
        self,
        capability: MethodCapability | None = None,
    ) -> tuple[MethodInstallation, ...]:
        return tuple(
            self.installation(descriptor.capability, descriptor.name)
            for descriptor in self.descriptors(capability)
        )

    def descriptors(
        self,
        capability: MethodCapability | None = None,
    ) -> tuple[MethodDescriptor, ...]:
        selected = (
            self._descriptors.values()
            if capability is None
            else (
                descriptor
                for (registered_capability, _), descriptor in self._descriptors.items()
                if registered_capability is capability
            )
        )
        return tuple(sorted(selected, key=lambda item: (item.capability.value, item.name)))

    def discover(self) -> None:
        """Atomically load registration functions from ``physrealx.methods`` entry points."""

        if sys.version_info >= (3, 10):
            selected = entry_points(group=ENTRY_POINT_GROUP)
        else:
            selected = entry_points().get(ENTRY_POINT_GROUP, ())
        for entry_point in sorted(selected, key=lambda item: (item.name, item.value)):
            snapshot = (
                self._descriptors.copy(),
                self._backends.copy(),
                self._importers.copy(),
                self._policies.copy(),
                self._trainers.copy(),
                self._adapters.copy(),
                self._execution_kinds.copy(),
            )
            try:
                registration = entry_point.load()
                if not callable(registration):
                    raise PluginError(f"entry point {entry_point.name!r} is not callable")
                registration(self)
                current = (
                    self._descriptors,
                    self._backends,
                    self._importers,
                    self._policies,
                    self._trainers,
                    self._adapters,
                    self._execution_kinds,
                )
                if current == snapshot:
                    raise PluginError(
                        f"entry point {entry_point.name!r} did not register any methods"
                    )
            except Exception as error:
                (
                    self._descriptors,
                    self._backends,
                    self._importers,
                    self._policies,
                    self._trainers,
                    self._adapters,
                    self._execution_kinds,
                ) = snapshot
                if isinstance(error, PluginError):
                    raise
                raise PluginError(
                    f"entry point {entry_point.name!r} failed during registration"
                ) from error


__all__ = ["ENTRY_POINT_GROUP", "MethodInstallation", "PluginRegistry", "PolicyFactory"]
