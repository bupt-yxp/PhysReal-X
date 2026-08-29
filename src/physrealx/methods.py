"""Factories for immutable method descriptors.

This module contains no registry state.  Applications own their
:class:`~physrealx.plugins.PluginRegistry` and may register additional worker
descriptors through entry points.
"""

from __future__ import annotations

from physrealx.domain import (
    MethodAccelerator,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodMaturity,
    MethodRuntime,
)

_PROJECT_SOURCE = "https://github.com/bupt-yxp/PhysReal-X"
_PROJECT_REVISION = "0.1.0"


def builtin_method(
    capability: MethodCapability,
    name: str,
    *,
    input_contract: str,
    output_contract: str,
    accelerator: MethodAccelerator = MethodAccelerator.ANY,
) -> MethodDescriptor:
    """Describe one method shipped in the PhysReal-X distribution."""

    return MethodDescriptor(
        capability=capability,
        name=name,
        version="1.0.0",
        api_version="1",
        source_uri=_PROJECT_SOURCE,
        source_revision=_PROJECT_REVISION,
        license_spdx="MIT",
        maturity=MethodMaturity.STABLE,
        runtime=MethodRuntime(
            mode=MethodExecutionMode.IN_PROCESS,
            accelerator=accelerator,
            python=">=3.8,<3.13",
        ),
        input_contract=input_contract,
        output_contract=output_contract,
    )


__all__ = ["builtin_method"]
