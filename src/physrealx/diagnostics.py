"""Runtime diagnostics. Heavy dependencies are imported only on explicit request."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class DoctorReport:
    python: str
    platform: str
    conda_environment: str | None
    torch_installed: bool
    torch_version: str | None
    torch_cuda_runtime: str | None
    cuda_available: bool
    cuda_device_count: int
    cuda_devices: tuple[str, ...]
    cu121_ready: bool
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_runtime(*, probe_cuda: bool = True) -> DoctorReport:
    """Inspect Python and optional Torch/CUDA without import-time GPU side effects."""

    issues: list[str] = []
    torch_version: str | None = None
    torch_cuda_runtime: str | None = None
    cuda_available = False
    cuda_device_count = 0
    cuda_devices: tuple[str, ...] = ()
    try:
        import torch

        torch_installed = True
        torch_version = torch.__version__
        torch_cuda_runtime = torch.version.cuda
        if probe_cuda:
            cuda_available = bool(torch.cuda.is_available())
            if cuda_available:
                cuda_device_count = int(torch.cuda.device_count())
                cuda_devices = tuple(
                    torch.cuda.get_device_name(index) for index in range(cuda_device_count)
                )
    except ImportError:
        torch_installed = False
        issues.append("PyTorch is not installed; GPU training adapters are unavailable")

    if torch_installed and torch_cuda_runtime != "12.1":
        issues.append(
            f"PyTorch was not compiled for CUDA 12.1 (reported {torch_cuda_runtime or 'CPU-only'})"
        )
    if probe_cuda and torch_installed and not cuda_available:
        issues.append("CUDA is not available to PyTorch")

    conda_environment = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_environment and conda_environment != "PhysReal-X":
        issues.append(f"active Conda environment is {conda_environment!r}, expected 'PhysReal-X'")

    return DoctorReport(
        python=sys.version.split()[0],
        platform=platform.platform(),
        conda_environment=conda_environment,
        torch_installed=torch_installed,
        torch_version=torch_version,
        torch_cuda_runtime=torch_cuda_runtime,
        cuda_available=cuda_available,
        cuda_device_count=cuda_device_count,
        cuda_devices=cuda_devices,
        cu121_ready=torch_installed and torch_cuda_runtime == "12.1" and cuda_available,
        issues=tuple(issues),
    )
