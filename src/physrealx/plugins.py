"""Explicit plugin registry with optional Python entry-point discovery."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import Literal

from physrealx.contracts import API_VERSION, Policy, SimulationBackend, Trainer
from physrealx.domain import JSONValue
from physrealx.errors import PluginError, ResourceConflictError, ResourceNotFoundError

PolicyFactory = Callable[[Mapping[str, JSONValue]], Policy]
_PLUGIN_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_PLUGIN_VERSION = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?"
    r"(?:\+[0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*)?$"
)


@dataclass(frozen=True, slots=True)
class PluginDescriptor:
    kind: Literal["simulation_backend", "policy", "trainer"]
    name: str
    version: str
    api_version: str

    def __post_init__(self) -> None:
        if self.kind not in {"simulation_backend", "policy", "trainer"}:
            raise PluginError(f"unsupported plugin kind {self.kind!r}")
        if _PLUGIN_NAME.fullmatch(self.name) is None:
            raise PluginError(f"invalid plugin name {self.name!r}")
        if _PLUGIN_VERSION.fullmatch(self.version) is None:
            raise PluginError(f"invalid plugin version {self.version!r}")


class PluginRegistry:
    """An application-owned registry; importing the package creates no global state."""

    def __init__(self) -> None:
        self._backends: dict[str, SimulationBackend] = {}
        self._policies: dict[str, tuple[PluginDescriptor, PolicyFactory]] = {}
        self._trainers: dict[str, Trainer] = {}

    @staticmethod
    def _validate(plugin: object) -> None:
        name = getattr(plugin, "name", None)
        version = getattr(plugin, "plugin_version", getattr(plugin, "version", None))
        if not isinstance(name, str) or _PLUGIN_NAME.fullmatch(name) is None:
            raise PluginError(f"plugin has an invalid name {name!r}")
        if not isinstance(version, str) or _PLUGIN_VERSION.fullmatch(version) is None:
            raise PluginError(f"plugin {name!r} has an invalid version {version!r}")
        api_version = getattr(plugin, "api_version", None)
        if api_version != API_VERSION:
            raise PluginError(
                f"plugin API version {api_version!r} is incompatible with {API_VERSION!r}"
            )

    def register_backend(self, backend: SimulationBackend) -> None:
        if not isinstance(backend, SimulationBackend):
            raise PluginError("simulation backend does not implement the required protocol")
        self._validate(backend)
        if backend.name in self._backends:
            raise ResourceConflictError(
                f"simulation backend {backend.name!r} is already registered"
            )
        self._backends[backend.name] = backend

    def register_policy(
        self,
        descriptor: PluginDescriptor,
        factory: PolicyFactory,
    ) -> None:
        if descriptor.kind != "policy":
            raise PluginError("policy registration requires a policy descriptor")
        if not callable(factory):
            raise PluginError("policy factory must be callable")
        self._validate(descriptor)
        if descriptor.name in self._policies:
            raise ResourceConflictError(f"policy {descriptor.name!r} is already registered")
        self._policies[descriptor.name] = (descriptor, factory)

    def register_trainer(self, trainer: Trainer) -> None:
        if not isinstance(trainer, Trainer):
            raise PluginError("trainer does not implement the required protocol")
        self._validate(trainer)
        if trainer.name in self._trainers:
            raise ResourceConflictError(f"trainer {trainer.name!r} is already registered")
        self._trainers[trainer.name] = trainer

    def backend(self, name: str) -> SimulationBackend:
        try:
            return self._backends[name]
        except KeyError as error:
            raise ResourceNotFoundError(f"simulation backend {name!r} is not registered") from error

    def policy(self, name: str, config: Mapping[str, JSONValue]) -> Policy:
        try:
            descriptor, factory = self._policies[name]
        except KeyError as error:
            raise ResourceNotFoundError(f"policy {name!r} is not registered") from error
        policy = factory(config)
        if not isinstance(policy, Policy):
            raise PluginError(f"policy factory {name!r} returned an incompatible object")
        self._validate(policy)
        if policy.name != descriptor.name or policy.plugin_version != descriptor.version:
            raise PluginError(f"policy factory {name!r} returned mismatched plugin identity")
        return policy

    def policy_descriptor(self, name: str) -> PluginDescriptor:
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

    def descriptors(self) -> tuple[PluginDescriptor, ...]:
        descriptors = [
            PluginDescriptor(
                kind="simulation_backend",
                name=item.name,
                version=item.plugin_version,
                api_version=item.api_version,
            )
            for item in self._backends.values()
        ]
        descriptors.extend(descriptor for descriptor, _ in self._policies.values())
        descriptors.extend(
            PluginDescriptor(
                kind="trainer",
                name=item.name,
                version=item.plugin_version,
                api_version=item.api_version,
            )
            for item in self._trainers.values()
        )
        return tuple(sorted(descriptors, key=lambda item: (item.kind, item.name)))

    def discover(self) -> None:
        """Load external registration functions from ``physrealx.plugins`` entry points."""

        for entry_point in entry_points(group="physrealx.plugins"):
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
