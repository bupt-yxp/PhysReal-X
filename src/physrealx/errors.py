"""Stable exception hierarchy for PhysReal-X.

Expected operational failures are represented by these exceptions so callers can
handle them without parsing human-readable messages.  Validation errors raised by
Pydantic remain ``pydantic.ValidationError``; ``DomainValidationError`` is for
validation performed while compiling or combining already-created domain objects.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar

from pydantic import JsonValue


class PhysRealXError(Exception):
    """Base class for stable, user-facing PhysReal-X failures."""

    code: ClassVar[str] = "physrealx_error"

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, JsonValue] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})


class DomainValidationError(PhysRealXError):
    """Multiple valid domain objects cannot form a valid aggregate."""

    code = "domain_validation_error"


class ResourceNotFoundError(PhysRealXError):
    """A requested catalog resource does not exist."""

    code = "resource_not_found"


class ResourceConflictError(PhysRealXError):
    """An immutable resource already exists with different content."""

    code = "resource_conflict"


class ArtifactIntegrityError(PhysRealXError):
    """Artifact bytes are unavailable or do not match their declared size."""

    code = "artifact_integrity_error"


class InvalidStateTransitionError(PhysRealXError):
    """A persisted job was asked to perform an illegal state transition."""

    code = "invalid_state_transition"


class PluginError(PhysRealXError):
    """Base class for plugin discovery or execution failures."""

    code = "plugin_error"


class PluginCompatibilityError(PluginError):
    """A plugin does not implement the requested API or capability."""

    code = "plugin_compatibility_error"


class SourcePackageError(PhysRealXError):
    """An offline source package is unsafe, incomplete or inconsistent."""

    code = "source_package_error"


class MethodExecutionError(PhysRealXError):
    """A capability method failed or returned invalid execution evidence."""

    code = "method_execution_error"


class MethodUnavailableError(MethodExecutionError):
    """An installed method cannot run in the current project environment."""

    code = "method_unavailable"


__all__ = [
    "ArtifactIntegrityError",
    "DomainValidationError",
    "InvalidStateTransitionError",
    "MethodExecutionError",
    "MethodUnavailableError",
    "PhysRealXError",
    "PluginCompatibilityError",
    "PluginError",
    "ResourceConflictError",
    "ResourceNotFoundError",
    "SourcePackageError",
]
