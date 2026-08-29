"""Capability-oriented plugin and method registry."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from importlib.metadata import entry_points
from typing import Callable
from typing import Mapping as TypingMapping

from typing_extensions import TypeAlias

from physrealx.contracts import (
    API_VERSION,
    DigitalTwinPipeline,
    ObservationImporter,
    Policy,
    SimulationBackend,
    Trainer,
)
from physrealx.domain import JSONValue, MethodCapability, MethodDescriptor, MethodSpec
from physrealx.errors import PluginError, ResourceConflictError, ResourceNotFoundError

PolicyFactory: TypeAlias = Callable[[TypingMapping[str, JSONValue]], Policy]


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
        self._asset_pipelines: dict[str, DigitalTwinPipeline] = {}
        self._policies: dict[str, tuple[MethodDescriptor, PolicyFactory]] = {}
        self._trainers: dict[str, Trainer] = {}

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

    def register_method(self, descriptor: MethodDescriptor) -> None:
        """Register a method executed inside a composite or remote worker."""

        self._record(descriptor)

    def register_backend(self, backend: SimulationBackend) -> None:
        if not isinstance(backend, SimulationBackend):
            raise PluginError("simulation backend does not implement the required protocol")
        descriptor = self._descriptor(backend, MethodCapability.SIMULATION_RUN)
        if descriptor.name in self._backends:
            raise ResourceConflictError(
                f"simulation backend {descriptor.name!r} is already registered"
            )
        self._record(descriptor)
        self._backends[descriptor.name] = backend

    def register_importer(self, importer: ObservationImporter) -> None:
        if not isinstance(importer, ObservationImporter):
            raise PluginError("observation importer does not implement the required protocol")
        descriptor = self._descriptor(importer, MethodCapability.OBSERVATION_IMPORT)
        if descriptor.name in self._importers:
            raise ResourceConflictError(
                f"observation importer {descriptor.name!r} is already registered"
            )
        self._record(descriptor)
        self._importers[descriptor.name] = importer

    def register_asset_pipeline(self, pipeline: DigitalTwinPipeline) -> None:
        if not isinstance(pipeline, DigitalTwinPipeline):
            raise PluginError("asset pipeline does not implement the required protocol")
        descriptor = self._descriptor(pipeline, MethodCapability.ASSET_BUILD)
        if descriptor.name in self._asset_pipelines:
            raise ResourceConflictError(f"asset pipeline {descriptor.name!r} is already registered")
        self._record(descriptor)
        self._asset_pipelines[descriptor.name] = pipeline

    def register_policy(
        self,
        descriptor: MethodDescriptor,
        factory: PolicyFactory,
    ) -> None:
        self._validate_descriptor(descriptor, MethodCapability.POLICY_INFERENCE)
        if not callable(factory):
            raise PluginError("policy factory must be callable")
        if descriptor.name in self._policies:
            raise ResourceConflictError(f"policy {descriptor.name!r} is already registered")
        self._record(descriptor)
        self._policies[descriptor.name] = (descriptor, factory)

    def register_trainer(self, trainer: Trainer) -> None:
        if not isinstance(trainer, Trainer):
            raise PluginError("trainer does not implement the required protocol")
        descriptor = self._descriptor(trainer, MethodCapability.POLICY_TRAIN)
        if descriptor.name in self._trainers:
            raise ResourceConflictError(f"trainer {descriptor.name!r} is already registered")
        self._record(descriptor)
        self._trainers[descriptor.name] = trainer

    def backend(self, name: str) -> SimulationBackend:
        try:
            return self._backends[name]
        except KeyError as error:
            raise ResourceNotFoundError(f"simulation backend {name!r} is not registered") from error

    def importer(self, name: str) -> ObservationImporter:
        try:
            return self._importers[name]
        except KeyError as error:
            raise ResourceNotFoundError(
                f"observation importer {name!r} is not registered"
            ) from error

    def asset_pipeline(self, name: str) -> DigitalTwinPipeline:
        try:
            return self._asset_pipelines[name]
        except KeyError as error:
            raise ResourceNotFoundError(f"asset pipeline {name!r} is not registered") from error

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
        """Load external registration functions from ``physrealx.plugins`` entry points."""

        if sys.version_info >= (3, 10):
            selected = entry_points(group="physrealx.plugins")
        else:
            selected = entry_points().get("physrealx.plugins", ())
        for entry_point in selected:
            try:
                registration = entry_point.load()
                if not callable(registration):
                    raise PluginError(f"entry point {entry_point.name!r} is not callable")
                registration(self)
            except PluginError:
                raise
            except Exception as error:
                raise PluginError(
                    f"entry point {entry_point.name!r} failed during registration"
                ) from error


__all__ = ["PluginRegistry", "PolicyFactory"]
