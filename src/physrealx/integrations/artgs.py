"""In-repository ArtGS integration for metric articulated visual geometry.

The adapter deliberately keeps Torch, CUDA and ArtGS imports outside the core
process.  Training still runs in a child process for fault containment, but the
child uses the same Python executable and package environment as PhysReal-X.
"""

from __future__ import annotations

import csv
import fcntl
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, NoReturn, Protocol

from pydantic import Field, model_validator

from physrealx.contracts import ArtifactStore, MethodExecutionOutput, MethodProbe
from physrealx.domain import (
    ArtifactRef,
    DomainModel,
    JSONValue,
    MethodAccelerator,
    MethodCapability,
    MethodDescriptor,
    MethodExecutionMode,
    MethodMaturity,
    MethodRuntime,
    OperationOutput,
    OperationSpec,
    canonical_json_bytes,
)
from physrealx.errors import DomainValidationError, MethodExecutionError
from physrealx.integrations.artgs_capture import ArtGSPreparedInput, prepare_artgs_capture
from physrealx.integrations.rgbd import MetricRGBDInput
from physrealx.reconstruction import ReconstructedPart, ReconstructionPrediction

_BUNDLE_SCHEMA: Final = "physrealx.visual-geometry-bundle/1"
_REPORT_SCHEMA: Final = "physrealx.geometry-reconstruction-report/1"
_UPSTREAM_REVISION: Final = "7c1f41be2cb8b96abca13c6a9668dcf7e06d8c2a"
_SOURCE_REVISION: Final = "7c1f41be2cb8b96a.physrealx10"
_SOURCE_REVISION_LINE = re.compile(r'^revision\s*=\s*"([0-9a-f]{40})"\s*$', re.MULTILINE)
_GENERATED_SOURCE_DIRECTORIES: Final = frozenset({"__pycache__", "build", "dist"})
_GENERATED_SOURCE_SUFFIXES: Final = frozenset({".o", ".pyc", ".pyd", ".so"})
_NATIVE_IMPORTS: Final = (
    (
        "diff_gaussian_rasterization._C",
        "submodules/diff-gaussian-rasterization",
    ),
    ("simple_knn._C", "submodules/simple-knn"),
    ("pointnet2_cuda", "utils/pointnet_lib"),
)


def _python_version() -> tuple[int, int]:
    """Return the active runtime version without making later probe code static."""

    return sys.version_info.major, sys.version_info.minor


def _strict_json(payload: bytes, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"{label} contains duplicate JSON key {key!r}")
            value[key] = item
        return value

    def reject_constant(value: str) -> NoReturn:
        raise ValueError(f"{label} contains forbidden non-finite value {value}")

    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(f"{label} must be unambiguous finite UTF-8 JSON") from error


class ArtGSRunConfig(DomainModel):
    """Bound, finite ArtGS execution configuration."""

    num_slots: int = Field(ge=2, le=32)
    joint_types: tuple[Literal["hinge", "slide"], ...] | None = None
    reference_joint: str | None = Field(default=None, min_length=1)

    coarse_iterations: int = Field(default=10_000, ge=100, le=200_000)
    joint_iterations: int = Field(default=5_000, ge=100, le=100_000)
    reconstruction_iterations: int = Field(default=20_000, ge=100, le=500_000)
    training_resolution: int = Field(default=1, ge=1, le=16)
    evaluation_resolution: int = Field(default=2, ge=1, le=16)
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    max_stage_seconds: int = Field(default=86_400, ge=60, le=604_800)
    state_streams: tuple[str, str] | None = None
    training_frames_per_state: int | None = Field(default=None, ge=2, le=2048, strict=True)
    max_initial_points: int = Field(default=1_000_000, ge=4096, le=2_000_000, strict=True)
    point_loss_weight: float = Field(default=100.0, ge=0.0, le=10_000.0)
    metric_depth_loss_weight: float = Field(default=1.0, ge=0.0, le=1_000.0)
    mono_depth_loss_weight: float = Field(default=0.0, ge=0.0, le=1_000.0)

    @model_validator(mode="after")
    def validate_joint_types(self) -> ArtGSRunConfig:
        if self.joint_types is not None and len(self.joint_types) != self.num_slots - 1:
            raise ValueError("joint_types must describe every non-root motion slot")
        if self.reference_joint is not None and (
            self.num_slots != 2 or not self.reference_joint.strip()
        ):
            raise ValueError("reference_joint requires one root and one moving slot")
        return self


ARTGS_DESCRIPTOR = MethodDescriptor(
    capability=MethodCapability.GEOMETRY_RECONSTRUCT,
    name="artgs",
    version="2.5.0",
    api_version="1",
    source_uri="https://github.com/bupt-yxp/PhysReal-X/tree/main/algorithms/artgs",
    source_revision=_SOURCE_REVISION,
    license_spdx="LicenseRef-ArtGS-Research-Only",
    maturity=MethodMaturity.EXPERIMENTAL,
    runtime=MethodRuntime(
        mode=MethodExecutionMode.IN_PROCESS,
        accelerator=MethodAccelerator.CUDA,
        python=">=3.10,<3.13",
    ),
    input_contract="MetricMultiStateRGBDBundle/1",
    output_contract="VisualGeometryBundle/1+GeometryReconstructionReport/1",
)


def artgs_source_root() -> Path:
    """Resolve the vendored source in editable and installed distributions."""

    module = Path(__file__).resolve()
    checkout = module.parents[3] / "algorithms" / "artgs"
    installed = module.parents[1] / "_embedded" / "artgs"
    for candidate in (checkout, installed):
        if (candidate / "SOURCE.toml").is_file() and (candidate / "train.py").is_file():
            return candidate
    raise FileNotFoundError("the embedded ArtGS source snapshot is unavailable")


def _validate_source_snapshot(root: Path) -> str:
    manifest = root / "SOURCE.toml"
    try:
        text = manifest.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise MethodExecutionError("embedded ArtGS source manifest is unreadable") from error
    revision_match = _SOURCE_REVISION_LINE.search(text)
    if revision_match is None or revision_match.group(1) != _UPSTREAM_REVISION:
        raise MethodExecutionError("embedded ArtGS source revision does not match its descriptor")
    return revision_match.group(1)


def _media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    return {
        ".json": "application/json",
        ".png": "image/png",
        ".ply": "model/ply",
        ".npy": "application/x-npy",
    }.get(suffix, "application/octet-stream")


def _fixed_zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info


@dataclass(frozen=True)
class _ArtGSRunResult:
    model_root: Path
    mesh_paths: tuple[Path, ...]
    metrics: Mapping[str, float]
    evaluation_scope: Literal["held_out_views"]
    stage_diagnostics: tuple[Path, ...] = ()


class _ArtGSRunner(Protocol):
    def probe(self) -> MethodProbe: ...

    def run(
        self,
        dataset_root: Path,
        output_root: Path,
        manifest: ArtGSPreparedInput,
        config: ArtGSRunConfig,
    ) -> _ArtGSRunResult: ...


class _SubprocessArtGSRunner:
    """Run the vendored implementation with the active project interpreter."""

    def __init__(self, source_root: Path | None = None) -> None:
        self.source_root = (source_root or artgs_source_root()).resolve(strict=True)

    def _python_paths(self) -> tuple[Path, ...]:
        return (
            self.source_root,
            self.source_root / "submodules" / "diff-gaussian-rasterization",
            self.source_root / "submodules" / "simple-knn",
            self.source_root / "utils" / "pointnet_lib",
        )

    def _environment(self, seed: int) -> dict[str, str]:
        environment = dict(os.environ)
        existing = environment.get("PYTHONPATH")
        paths = [str(item) for item in self._python_paths()]
        if existing:
            paths.append(existing)
        environment.update(
            {
                "PYTHONPATH": os.pathsep.join(paths),
                "PYTHONHASHSEED": str(seed),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        return environment

    @staticmethod
    def _nvcc() -> Path | None:
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            candidate = Path(cuda_home) / "bin" / "nvcc"
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
        discovered = shutil.which("nvcc")
        return Path(discovered) if discovered else None

    def probe(self) -> MethodProbe:
        if not (3, 10) <= _python_version() < (3, 13):
            return MethodProbe(
                False, "ArtGS requires Python 3.10-3.12 in the current PhysReal-X environment"
            )
        if self._nvcc() is None:
            return MethodProbe(False, "ArtGS requires nvcc in CUDA_HOME or PATH")
        required_sources = (
            self.source_root / "train.py",
            self.source_root / "submodules" / "diff-gaussian-rasterization" / "setup.py",
            self.source_root / "submodules" / "simple-knn" / "setup.py",
            self.source_root / "utils" / "pointnet_lib" / "setup.py",
        )
        if any(not item.is_file() for item in required_sources):
            return MethodProbe(False, "embedded ArtGS or CUDA extension source is incomplete")
        try:
            _validate_source_snapshot(self.source_root)
        except MethodExecutionError as error:
            return MethodProbe(False, str(error))
        command = [
            sys.executable,
            "-c",
            """
import sys
try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('no usable CUDA device in the current PhysReal-X environment')
    import open3d, pytorch3d, tinycudann, torchvision
    value = torch.eye(3, device='cuda')
    assert torch.isfinite(value @ value).all().item()
    torch.cuda.synchronize()
    print(f'{torch.cuda.get_device_name(0)}; Torch {torch.__version__}; CUDA {torch.version.cuda}')
except Exception as error:
    print(f'{type(error).__name__}: {error}')
    sys.exit(1)
""",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=self.source_root,
                env=self._environment(0),
                check=False,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=30,
                text=True,
            )
        except (OSError, subprocess.SubprocessError) as error:
            return MethodProbe(False, f"ArtGS dependency probe failed: {error}")
        detail = completed.stdout[-2048:].strip()
        if completed.returncode != 0:
            return MethodProbe(False, detail or "ArtGS dependency probe failed")
        return MethodProbe(
            True,
            f"embedded ArtGS is ready in {sys.executable}; CUDA device: {detail}",
        )

    def _native_imports_ready(self, environment: Mapping[str, str]) -> bool:
        validation_script = """
import importlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1]).resolve(strict=True)
for module_name, relative_root in json.loads(sys.argv[2]):
    module = importlib.import_module(module_name)
    origin_value = getattr(module, "__file__", None)
    if not isinstance(origin_value, str):
        raise RuntimeError(f"{module_name} has no import origin")
    origin = pathlib.Path(origin_value).resolve(strict=True)
    expected = (root / relative_root).resolve(strict=True)
    try:
        origin.relative_to(expected)
    except ValueError as error:
        raise RuntimeError(
            f"{module_name} resolved outside embedded ArtGS: {origin}"
        ) from error
"""
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                validation_script,
                str(self.source_root),
                json.dumps(_NATIVE_IMPORTS, separators=(",", ":")),
            ],
            cwd=self.source_root,
            env=dict(environment),
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return completed.returncode == 0

    def _ensure_native_extensions(self, environment: Mapping[str, str]) -> None:
        if self._native_imports_ready(environment):
            return
        lock_path = Path(tempfile.gettempdir()) / "physrealx-artgs-native-build.lock"
        with lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if self._native_imports_ready(environment):
                return
            projects = (
                self.source_root / "submodules" / "diff-gaussian-rasterization",
                self.source_root / "submodules" / "simple-knn",
                self.source_root / "utils" / "pointnet_lib",
            )
            for project in projects:
                completed = subprocess.run(
                    [sys.executable, "setup.py", "build_ext", "--inplace"],
                    cwd=project,
                    env=dict(environment),
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                    text=True,
                )
                if completed.returncode != 0:
                    raise MethodExecutionError(
                        f"failed to build embedded ArtGS CUDA extension in {project.name}: "
                        f"{completed.stdout[-4096:]}"
                    )
            if not self._native_imports_ready(environment):
                raise MethodExecutionError(
                    "embedded ArtGS CUDA extensions failed import validation"
                )

    def _run_stage(
        self,
        arguments: Sequence[str],
        log_path: Path,
        environment: Mapping[str, str],
        timeout: int,
    ) -> None:
        with log_path.open("wb") as log:
            try:
                completed = subprocess.run(
                    [sys.executable, *arguments],
                    cwd=self.source_root,
                    env=dict(environment),
                    check=False,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired as error:
                raise MethodExecutionError(f"ArtGS stage timed out: {arguments[0]}") from error
        if completed.returncode != 0:
            tail = log_path.read_bytes()[-8192:].decode("utf-8", errors="replace")
            raise MethodExecutionError(f"ArtGS stage failed: {arguments[0]}\n{tail}")

    @staticmethod
    def _metrics(path: Path, prefix: str) -> dict[str, float]:
        if not path.is_file():
            raise MethodExecutionError(f"ArtGS evaluation result is missing: {path}")
        values: dict[str, float] = {}
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                if len(row) < 2 or row[0] not in {"PSNR", "SSIM", "LPIPS", "FG_PSNR", "ALPHA_IOU"}:
                    continue
                value = float(row[1])
                if not math.isfinite(value) or value < 0.0:
                    raise MethodExecutionError(f"ArtGS produced an invalid {row[0]} metric")
                values[f"{prefix}_{row[0].lower()}"] = value
        if set(values) != {
            f"{prefix}_{name}" for name in ("psnr", "ssim", "lpips", "fg_psnr", "alpha_iou")
        }:
            raise MethodExecutionError("ArtGS evaluation must include image and foreground metrics")
        return values

    @staticmethod
    def _stage_diagnostics(
        model_root: Path,
        *,
        coarse_iterations: int,
        joint_types: str,
    ) -> tuple[Path, ...]:
        """Render the native output of each ArtGS stage without changing it."""
        import numpy as np
        from matplotlib.figure import Figure
        from PIL import Image

        diagnostics = model_root / "diagnostics"
        diagnostics.mkdir(parents=True, exist_ok=False)
        coarse = model_root / "coarse" / "point_cloud" / f"iteration_{coarse_iterations}"

        def points(name: str) -> tuple[Any, int]:
            path = coarse / name
            if not path.is_file():
                raise MethodExecutionError(f"ArtGS coarse diagnostic is missing: {path}")
            values = np.load(path, allow_pickle=False)
            if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
                raise MethodExecutionError(f"ArtGS coarse diagnostic is invalid: {path}")
            total = len(values)
            if total > 30_000:
                values = values[np.linspace(0, len(values) - 1, 30_000, dtype=int)]
            return values, total

        def equal_axes(axis: Any, groups: Sequence[Any]) -> None:
            cloud = np.concatenate([group for group in groups if len(group)])
            lower, upper = cloud.min(0), cloud.max(0)
            center = (lower + upper) / 2
            radius = max(float((upper - lower).max()) / 2, 1e-3)
            axis.set_xlim(center[0] - radius, center[0] + radius)
            axis.set_ylim(center[1] - radius, center[1] + radius)
            axis.set_zlim(center[2] - radius, center[2] + radius)
            axis.set_box_aspect((1, 1, 1))
            axis.view_init(elev=24, azim=-58)
            axis.set_axis_off()

        coarse_groups = {
            state_index: (
                points(f"xyz_static_{state_index}.npy"),
                points(f"xyz_dynamic_{state_index}.npy"),
            )
            for state_index in (0, 1)
        }
        figure = Figure(figsize=(10, 4), constrained_layout=True)
        for index, (state_index, (static_result, dynamic_result)) in enumerate(
            coarse_groups.items(), start=1
        ):
            static, static_count = static_result
            dynamic, dynamic_count = dynamic_result
            axis = figure.add_subplot(1, 2, index, projection="3d")
            axis.scatter(*static.T, s=0.35, c="#667085", alpha=0.45, depthshade=False)
            axis.scatter(*dynamic.T, s=0.45, c="#f79009", alpha=0.65, depthshade=False)
            equal_axes(axis, (static, dynamic))
            axis.set_title(
                f"State {state_index}: static {static_count:,} / moving {dynamic_count:,}"
            )
        figure.suptitle("Stage 1 - independent-state coarse reconstruction and matching")
        coarse_preview = diagnostics / "01-coarse.png"
        figure.savefig(coarse_preview, dpi=150)

        static, _ = points("xyz_static.npy")
        dynamic, _ = points("xyz_dynamic.npy")
        center_path = coarse / "center_info.npy"
        centers = np.load(center_path, allow_pickle=False)
        if (
            centers.ndim != 2
            or centers.shape[1] != 4
            or not np.isfinite(centers).all()
            or len(centers) != len(joint_types.split(","))
        ):
            raise MethodExecutionError("ArtGS predicted joint centers are invalid")
        figure = Figure(figsize=(6, 5), constrained_layout=True)
        axis = figure.add_subplot(1, 1, 1, projection="3d")
        axis.scatter(*static.T, s=0.35, c="#667085", alpha=0.35, depthshade=False)
        axis.scatter(*dynamic.T, s=0.45, c="#f79009", alpha=0.65, depthshade=False)
        axis.scatter(*centers[:, :3].T, s=90, c="#d92d20", marker="x", depthshade=False)
        equal_axes(axis, (static, dynamic, centers[:, :3]))
        axis.set_title(f"Predicted native joint types: {joint_types}")
        figure.suptitle("Stage 2 - motion-type prediction")
        joint_preview = diagnostics / "02-joints.png"
        figure.savefig(joint_preview, dpi=150)

        interpolation = model_root / "reconstruction" / "test" / "ours_best" / "interpolation"
        images = []
        try:
            for index in (0, 2, 4):
                with Image.open(interpolation / f"{index:02d}.png") as source_image:
                    images.append(np.asarray(source_image.convert("RGB")))
        except (OSError, ValueError) as error:
            raise MethodExecutionError("ArtGS final interpolation preview is invalid") from error
        if len({image.shape for image in images}) != 1:
            raise MethodExecutionError("ArtGS final interpolation previews have different sizes")
        figure = Figure(figsize=(12, 4), constrained_layout=True)
        for panel, (amount, pixels) in enumerate(zip((0.0, 0.5, 1.0), images), start=1):
            axis = figure.add_subplot(1, 3, panel)
            axis.imshow(pixels)
            axis.set_title(f"Observed-state interpolation: {amount:.1f}")
            axis.set_axis_off()
        figure.suptitle("Stage 3 - joint reconstruction")
        reconstruction_preview = diagnostics / "03-reconstruction.png"
        figure.savefig(reconstruction_preview, dpi=150)

        metadata = diagnostics / "stages.json"
        joint_type_values: list[JSONValue] = list(joint_types.split(","))
        document: dict[str, JSONValue] = {
            "schema": "physrealx.artgs-stage-diagnostics/1",
            "joint_types": joint_type_values,
            "stages": [
                {
                    "name": "coarse",
                    "preview": coarse_preview.name,
                    "source": "independent-state Gaussian matching",
                },
                {
                    "name": "joint_prediction",
                    "preview": joint_preview.name,
                    "source": "predicted_joint_types.json",
                },
                {
                    "name": "reconstruction",
                    "preview": reconstruction_preview.name,
                    "source": "held-out observed-state interpolation",
                },
            ],
        }
        metadata.write_bytes(canonical_json_bytes(document))
        return coarse_preview, joint_preview, reconstruction_preview, metadata

    def run(
        self,
        dataset_root: Path,
        output_root: Path,
        manifest: ArtGSPreparedInput,
        config: ArtGSRunConfig,
    ) -> _ArtGSRunResult:
        _validate_source_snapshot(self.source_root)
        environment = self._environment(config.seed)
        self._ensure_native_extensions(environment)
        logs = output_root / "logs"
        logs.mkdir(parents=True, exist_ok=False)
        model_root = output_root / "model"
        coarse_root = model_root / "coarse"
        prediction_root = model_root / "prediction"
        reconstruction_root = model_root / "reconstruction"
        common = [
            "--data_device",
            "cpu",
            "--source_path",
            str(dataset_root.parent.parent.parent),
            "--dataset",
            "artgs",
            "--subset",
            "realscan",
            "--scene_name",
            manifest.scene_id,
        ]
        joint_initialization = []
        if manifest.joint_initialization_file is not None:
            initialization_path = dataset_root / manifest.joint_initialization_file
            if not initialization_path.is_file():
                raise MethodExecutionError("ArtGS reference joint initialization is missing")
            joint_initialization = [
                "--joint_initialization_path",
                str(initialization_path),
            ]
        self._run_stage(
            [
                str(self.source_root / "train_coarse.py"),
                *common,
                "--model_path",
                str(coarse_root),
                "--resolution",
                str(config.training_resolution),
                "--num_slots",
                str(manifest.num_slots),
                "--iterations",
                str(config.coarse_iterations),
                "--eval",
                "--init_from_pcd",
                "--opacity_reg_weight",
                "0.1",
                "--random_bg_color",
                "--seed",
                str(config.seed),
            ],
            logs / "01-coarse.log",
            environment,
            config.max_stage_seconds,
        )
        if config.joint_types is not None:
            joint_types = ",".join(
                ["s", *({"hinge": "r", "slide": "p"}[kind] for kind in config.joint_types)]
            )
        else:
            self._run_stage(
                [
                    str(self.source_root / "train_predict.py"),
                    *common,
                    "--model_path",
                    str(prediction_root),
                    "--coarse_model_path",
                    str(coarse_root),
                    "--eval",
                    "--resolution",
                    str(config.evaluation_resolution),
                    "--num_slots",
                    str(manifest.num_slots),
                    *joint_initialization,
                    "--iterations",
                    str(config.joint_iterations),
                    "--densify_grad_threshold",
                    "0.001",
                    "--random_bg_color",
                    "--metric_depth_loss_weight",
                    str(config.metric_depth_loss_weight),
                    "--cd_loss_weight",
                    str(config.point_loss_weight),
                    "--seed",
                    str(config.seed),
                ],
                logs / "02-joints.log",
                environment,
                config.max_stage_seconds,
            )
            joint_file = prediction_root / "predicted_joint_types.json"
            joint_document = _strict_json(joint_file.read_bytes(), "ArtGS joint prediction")
            inferred_joint_types = (
                joint_document.get("joint_types") if isinstance(joint_document, dict) else None
            )
            if (
                not isinstance(inferred_joint_types, str)
                or len(inferred_joint_types.split(",")) != manifest.num_slots
            ):
                raise MethodExecutionError("ArtGS joint prediction has an invalid slot count")
            joint_types = inferred_joint_types
        self._run_stage(
            [
                str(self.source_root / "train.py"),
                *common,
                "--model_path",
                str(reconstruction_root),
                "--coarse_model_path",
                str(coarse_root),
                "--eval",
                "--resolution",
                str(config.training_resolution),
                "--joint_types",
                joint_types,
                *joint_initialization,
                "--iterations",
                str(config.reconstruction_iterations),
                "--seed",
                str(config.seed),
                "--use_art_type_prior",
                "--random_bg_color",
                "--densify_grad_threshold",
                "0.001",
                "--metric_depth_loss_weight",
                str(config.metric_depth_loss_weight),
                "--cd_loss_weight",
                str(config.point_loss_weight),
                "--mono_depth_loss_weight",
                str(config.mono_depth_loss_weight),
            ],
            logs / "03-reconstruction.log",
            environment,
            config.max_stage_seconds,
        )
        render_arguments = [
            str(self.source_root / "render.py"),
            *common,
            "--model_path",
            str(reconstruction_root),
            "--resolution",
            str(config.evaluation_resolution),
            "--iteration",
            "best",
            "--eval_app",
            "--load2gpu_on_the_fly",
            "--seed",
            str(config.seed),
        ]
        self._run_stage(
            render_arguments,
            logs / "04-evaluation.log",
            environment,
            config.max_stage_seconds,
        )

        result_root = reconstruction_root / "train" / "ours_best"
        mesh_paths = tuple(
            result_root / "meshes" / f"start_{index}.ply" for index in range(manifest.num_slots)
        )
        if any(not path.is_file() or path.stat().st_size == 0 for path in mesh_paths):
            raise MethodExecutionError("ArtGS did not export one non-empty visual mesh per slot")
        metrics = self._metrics(result_root / "result.csv", "train")
        metrics.update(
            self._metrics(
                reconstruction_root / "test" / "ours_best" / "result.csv",
                "validation",
            )
        )
        stage_diagnostics = list(
            self._stage_diagnostics(
                model_root,
                coarse_iterations=config.coarse_iterations,
                joint_types=joint_types,
            )
        )
        if manifest.joint_initialization_file is not None:
            audit_path = model_root / "diagnostics" / "reference-joint-initialization.json"
            shutil.copyfile(dataset_root / manifest.joint_initialization_file, audit_path)
            stage_diagnostics.append(audit_path)
        return _ArtGSRunResult(
            model_root=model_root,
            mesh_paths=mesh_paths,
            metrics=metrics,
            evaluation_scope="held_out_views",
            stage_diagnostics=tuple(stage_diagnostics),
        )


def normalize_artgs_prediction(
    entries: Any, mesh_refs: Sequence[ArtifactRef], state_pair: tuple[str, str]
) -> ReconstructionPrediction:
    """Translate native ArtGS output once, at the adapter boundary."""
    try:
        if not isinstance(entries, list) or len(entries) != len(mesh_refs):
            raise ValueError("every mesh needs a native joint record")
        parts = {}
        for item in entries:
            index = item["id"]
            if type(index) is not int or not 0 <= index < len(mesh_refs) or str(index) in parts:
                raise ValueError("invalid or duplicate native motion group identity")
            if item["visuals"] != [f"meshes/start_{index}.ply"]:
                raise ValueError("native motion group references a different mesh")
            kind = {"heavy": "fixed", "slider": "slide", "hinge": "hinge"}[item["joint"]]
            data = item["jointData"]
            motion = {}
            if kind != "fixed":
                observed = data["observed_span"]
                if observed["unit"] != ("m" if kind == "slide" else "deg"):
                    raise ValueError("predicted observed span unit differs from joint type")
                motion = dict(
                    axis=data["axis"]["direction"],
                    origin=data["axis"]["origin"],
                    span=math.radians(observed["value"]) if kind == "hinge" else observed["value"],
                )
            parts[str(index)] = ReconstructedPart(
                kind=kind, parent=None if item["parent"] == -1 else str(item["parent"]), **motion
            )
        return ReconstructionPrediction(
            parts=parts,
            states={
                state_pair[0]: {str(i): ref for i, ref in enumerate(mesh_refs)},
                state_pair[1]: {},
            },
            motion_pair=state_pair,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise MethodExecutionError(f"invalid ArtGS reconstruction output: {error}") from error


def _bundle_outputs(result: _ArtGSRunResult, destination: Path) -> tuple[Path, int]:
    previews = result.model_root / "reconstruction" / "test" / "ours_best" / "interpolation"
    configuration = _strict_json(
        (result.model_root / "reconstruction" / "cfg_args.json").read_bytes(),
        "ArtGS model configuration",
    )
    if not isinstance(configuration, dict):
        raise MethodExecutionError("ArtGS model configuration must be an object")
    # Save architecture and rendering parameters without expired temporary paths.
    for key in (
        "source_path",
        "model_path",
        "coarse_model_path",
        "joint_initialization_path",
    ):
        configuration.pop(key, None)
    inference_config = result.model_root / "reconstruction" / "inference.json"
    inference_config.write_bytes(canonical_json_bytes(configuration))
    candidates = [
        inference_config,
        result.model_root / "reconstruction" / "point_cloud" / "iteration_best" / "point_cloud.ply",
        result.model_root / "reconstruction" / "deform" / "iteration_best" / "deform.pth",
        result.model_root / "reconstruction" / "train" / "ours_best" / "joint_info.json",
        *result.mesh_paths,
        *(previews / f"{index:02d}.png" for index in range(5)),
        previews / "states.json",
        *result.stage_diagnostics,
    ]
    for candidate in candidates:
        if not candidate.is_file() or candidate.stat().st_size == 0:
            raise MethodExecutionError(f"ArtGS required output is missing: {candidate}")
    records: list[JSONValue] = []
    payloads: list[tuple[str, Path]] = []
    for candidate in candidates:
        if candidate == inference_config:
            name = "model/config.json"
        elif candidate in result.mesh_paths:
            name = f"meshes/{candidate.name}"
        elif candidate.parent == previews:
            name = f"appearance/{candidate.name}"
        elif candidate in result.stage_diagnostics:
            name = f"diagnostics/{candidate.name}"
        elif candidate.suffix == ".pth":
            name = "model/deform.pth"
        elif candidate.name == "joint_info.json":
            name = "model/joint_info.json"
        else:
            name = "model/point_cloud.ply"
        record: dict[str, JSONValue] = {
            "path": name,
            "size_bytes": candidate.stat().st_size,
            "media_type": _media_type(candidate),
        }
        records.append(record)
        payloads.append((name, candidate))
    manifest: dict[str, JSONValue] = {
        "schema": _BUNDLE_SCHEMA,
        "status": "COMPLETED",
        "files": records,
    }
    with zipfile.ZipFile(
        destination,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
        allowZip64=True,
    ) as archive:
        archive.writestr(_fixed_zip_info("bundle-manifest.json"), canonical_json_bytes(manifest))
        for name, path in payloads:
            archive.writestr(_fixed_zip_info(name), path.read_bytes())
    return destination, len(result.mesh_paths)


class ArtGSMethodAdapter:
    """Registered `geometry.reconstruct:artgs` implementation."""

    descriptor = ARTGS_DESCRIPTOR

    def __init__(self, runner: _ArtGSRunner | None = None) -> None:
        self._runner = runner or _SubprocessArtGSRunner()

    def probe(self) -> MethodProbe:
        return self._runner.probe()

    def execute(self, spec: OperationSpec, artifacts: ArtifactStore) -> MethodExecutionOutput:
        if spec.method.descriptor != self.descriptor:
            raise DomainValidationError("ArtGS operation uses a mismatched method descriptor")
        if len(spec.inputs) != 1 or spec.inputs[0].name != "source":
            raise DomainValidationError("ArtGS requires exactly one source artifact")
        source = spec.inputs[0].artifact
        if (
            source is None
            or source.media_type != "application/json"
            or not artifacts.verify(source)
        ):
            raise DomainValidationError("ArtGS source must be a canonical RGB-D JSON artifact")
        rgbd = MetricRGBDInput.model_validate_json(artifacts.get_bytes(source))
        try:
            config = ArtGSRunConfig.model_validate(dict(spec.method.config))
        except ValueError as error:
            raise DomainValidationError("ArtGS method configuration is invalid") from error

        with tempfile.TemporaryDirectory(prefix="physrealx-artgs-run-") as temporary:
            root = Path(temporary)
            extracted = root / "dataset" / "artgs" / "realscan" / "source"
            manifest = prepare_artgs_capture(
                rgbd,
                artifacts,
                extracted,
                num_slots=config.num_slots,
                state_streams=config.state_streams,
                training_frames_per_state=config.training_frames_per_state,
                max_initial_points=config.max_initial_points,
                reference_joint=config.reference_joint,
                seed=config.seed,
            )
            if (
                config.joint_types is not None
                and manifest.reference_depth_alignment is not None
                and config.joint_types != (manifest.reference_depth_alignment.joint_type,)
            ):
                raise DomainValidationError(
                    "ArtGS joint type prior conflicts with the selected reference joint"
                )
            output_root = root / "output"
            output_root.mkdir()
            result = self._runner.run(extracted, output_root, manifest, config)
            if result.evaluation_scope != "held_out_views":
                raise MethodExecutionError("ArtGS runner omitted required held-out evaluation")
            if any(not math.isfinite(float(value)) for value in result.metrics.values()):
                raise MethodExecutionError("ArtGS returned non-finite reconstruction metrics")

            bundle_path, mesh_count = _bundle_outputs(result, root / "visual-geometry.zip")
            bundle_ref = artifacts.put_file(bundle_path, media_type="application/zip")
            mesh_refs = tuple(
                artifacts.put_file(path, media_type="model/ply") for path in result.mesh_paths
            )
            prediction = normalize_artgs_prediction(
                _strict_json(
                    (
                        result.model_root / "reconstruction/train/ours_best/joint_info.json"
                    ).read_bytes(),
                    "ArtGS joints",
                ),
                mesh_refs,
                manifest.state_pair,
            )
            heldout = result.evaluation_scope == "held_out_views"
            independently_scaled = manifest.metric_evidence_scope == "independent_metric_alignment"
            independent_depth = manifest.depth_evaluation_scope == "independent_reference"
            reference_guided = manifest.reference_depth_alignment is not None
            passed = heldout and independently_scaled and independent_depth and not reference_guided
            report = {
                "schema": _REPORT_SCHEMA,
                "status": "PASSED" if passed else "MODEL_CONDITIONED",
                "metric_evidence_scope": manifest.metric_evidence_scope,
                "depth_evaluation_scope": manifest.depth_evaluation_scope,
                "training_frame_ids": manifest.training_frame_ids,
                "validation_frame_ids": manifest.validation_frame_ids,
                "initial_point_count": manifest.initial_point_count,
                "rectification": "calibrated pinhole warp; nearest metric depth",
                "independent_physical_validation": False,
                "part_scope": "observable motion groups; stationary reference links may be merged",
                "reference_link_count": manifest.reference_link_count,
                "motion_group_count": manifest.num_slots,
                "joint_type_source": "explicit_prior"
                if config.joint_types is not None
                else (
                    "reference_initialized_capture_inference"
                    if reference_guided
                    else "capture_inference"
                ),
                "joint_type_prior": config.joint_types,
                "unit": "m",
                "prediction": prediction.model_dump(mode="json"),
                "coordinate_frame": manifest.coordinate_frame,
                "source_camera_artifact_id": manifest.source_camera.artifact_id,
                "source_depth_artifact_id": manifest.source_depth.artifact_id,
                "method": {
                    "name": self.descriptor.name,
                    "version": self.descriptor.version,
                    "source_revision": self.descriptor.source_revision,
                    "upstream_revision": _validate_source_snapshot(artgs_source_root()),
                },
                "evaluation_scope": result.evaluation_scope,
                "bundle": bundle_ref.model_dump(mode="json"),
                "meshes": [
                    {
                        "link": f"link_{index}",
                        "artifact": reference.model_dump(mode="json"),
                    }
                    for index, reference in enumerate(mesh_refs)
                ],
                "metrics": {key: float(value) for key, value in sorted(result.metrics.items())},
                "gates": {
                    "metric_input": True,
                    "two_state_capture": manifest.training_frames >= 4,
                    "complete_link_meshes": len(mesh_refs) == manifest.num_slots,
                    "held_out_evaluation": heldout,
                    "reference_guided_initialization": reference_guided,
                },
            }
            if manifest.reference_depth_alignment is not None:
                report["reference_depth_alignment"] = manifest.reference_depth_alignment.model_dump(
                    mode="json", by_alias=True
                )
            report_ref = artifacts.put_json(report, media_type="application/json")
            metrics = {key: float(value) for key, value in result.metrics.items()}
            metrics.update(
                {
                    "mesh_count": float(mesh_count),
                    "training_frames": float(manifest.training_frames),
                    "validation_frames": float(manifest.validation_frames),
                }
            )
            if manifest.reference_depth_alignment is not None:
                metrics.update(
                    {
                        f"reference_alignment_{key}": float(value)
                        for key, value in manifest.reference_depth_alignment.metrics.items()
                    }
                )
            return MethodExecutionOutput(
                outputs=(
                    OperationOutput(
                        name="visual_geometry_bundle",
                        contract="VisualGeometryBundle/1",
                        artifact=bundle_ref,
                    ),
                    OperationOutput(
                        name="geometry_reconstruction_report",
                        contract="GeometryReconstructionReport/1",
                        artifact=report_ref,
                    ),
                ),
                metrics=metrics,
            )


__all__ = [
    "ARTGS_DESCRIPTOR",
    "ArtGSMethodAdapter",
    "ArtGSRunConfig",
    "artgs_source_root",
]
