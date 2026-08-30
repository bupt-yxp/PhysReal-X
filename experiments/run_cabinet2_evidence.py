#!/usr/bin/env python3
"""Run the evidence-backed portion of the cabinet2 lifecycle.

This experiment deliberately stops at the first unmet scientific prerequisite.
It never substitutes existing simulation annotations for camera scale,
articulation evidence, measured system-identification signals, demonstrations,
or robot hardware evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

from physrealx import (
    LifecycleEvidenceKey,
    LifecycleStage,
    MethodCapability,
    MethodSpec,
    PhysRealX,
)
from physrealx.errors import ResourceNotFoundError
from physrealx.methods import builtin_method

for _font_path in (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
):
    if _font_path.is_file():
        font_manager.fontManager.addfont(str(_font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(_font_path)).get_name()
        break
plt.rcParams["axes.unicode_minus"] = False


_STAGE_LABELS = {
    "ingestion": "观测数据导入",
    "camera": "相机恢复",
    "metric_alignment": "米制尺度对齐",
    "depth": "深度增强",
    "geometry": "米制视觉几何",
    "articulation": "部件与关节",
    "collision": "碰撞代理",
    "physics": "物性辨识",
    "assembly": "资产组装",
    "usd": "USD 封装",
    "physical_validation": "物理验证",
    "dataset": "数据集",
    "policy": "策略训练",
    "world_model": "世界模型",
    "benchmark": "固定基准评测",
    "ros_deploy": "ROS 2 部署",
    "failure": "失败归因",
    "feedback": "数据回流",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-state", default="can.start")
    parser.add_argument("--camera-frame-stride", type=int, default=20)
    parser.add_argument("--camera-max-frames", type=int, default=24)
    parser.add_argument("--depth-frame-stride", type=int, default=20)
    return parser.parse_args()


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _depth_quality(path: Path, stride: int) -> dict[str, Any]:
    packed = np.load(path, mmap_mode="r", allow_pickle=False)
    if packed.ndim != 1 or packed.dtype.kind != "S":
        raise ValueError(f"expected one-dimensional packed PNG depth NPY: {path}")
    coverage: list[float] = []
    medians: list[float] = []
    p95: list[float] = []
    shape: tuple[int, int] | None = None
    sampled_indices = list(range(0, len(packed), stride))
    for index in sampled_indices:
        payload = bytes(packed[index])
        with Image.open(io.BytesIO(payload)) as image:
            depth = np.asarray(image)
        if depth.ndim != 2 or depth.dtype != np.uint16:
            raise ValueError(f"packed depth frame {path}:{index} is not uint16 single-channel PNG")
        current_shape = (int(depth.shape[0]), int(depth.shape[1]))
        if shape is None:
            shape = current_shape
        elif shape != current_shape:
            raise ValueError(f"packed depth frame sizes are inconsistent: {path}")
        valid = (depth != 0) & (depth != np.iinfo(np.uint16).max)
        coverage.append(float(np.mean(valid)))
        if np.any(valid):
            values = depth[valid]
            medians.append(float(np.median(values)))
            p95.append(float(np.percentile(values, 95)))
    if shape is None or not coverage:
        raise ValueError(f"packed depth stream contains no frames: {path}")
    return {
        "path": str(path),
        "frame_count": len(packed),
        "sampled_frames": len(sampled_indices),
        "sample_stride": stride,
        "height": shape[0],
        "width": shape[1],
        "dtype": "uint16",
        "invalid_values": [0, 65535],
        "unit_status": "unresolved",
        "coverage_mean": float(np.mean(coverage)),
        "coverage_min": float(np.min(coverage)),
        "raw_median_mean": float(np.mean(medians)),
        "raw_p95_mean": float(np.mean(p95)),
    }


def _planned_method(capability: MethodCapability, name: str) -> MethodSpec:
    return builtin_method(
        capability,
        name,
        input_contract="LifecycleEvidence/1",
        output_contract="LifecycleEvidence/1",
    ).bind()


def _lifecycle_methods(platform: PhysRealX) -> dict[LifecycleStage, MethodSpec]:
    installed = {
        LifecycleStage.INGESTION: platform.plugins.importer("artmodeling").descriptor.bind(),
        LifecycleStage.CAMERA: platform.methods.get(
            MethodCapability.CAMERA_RECOVER, "colmap"
        ).bind(),
    }
    try:
        installed[LifecycleStage.METRIC_ALIGNMENT] = platform.methods.get(
            MethodCapability.CAMERA_ALIGN_METRIC, "metric_alignment"
        ).bind()
    except ResourceNotFoundError:
        installed[LifecycleStage.METRIC_ALIGNMENT] = _planned_method(
            MethodCapability.CAMERA_ALIGN_METRIC, "metric_alignment"
        )
    optional_installed = {
        LifecycleStage.COLLISION: (MethodCapability.COLLISION_BUILD, "coacd"),
        LifecycleStage.PHYSICS: (
            MethodCapability.PHYSICS_CALIBRATE,
            "physrealx_mujoco_sysid_adapter",
        ),
        LifecycleStage.USD: (MethodCapability.ASSET_PACKAGE, "openusd"),
    }
    for stage, (capability, name) in optional_installed.items():
        try:
            installed[stage] = platform.methods.get(capability, name).bind()
        except ResourceNotFoundError:
            installed[stage] = _planned_method(capability, name)
    desired = {
        LifecycleStage.DEPTH: (MethodCapability.DEPTH_ENHANCE, "omni_dc"),
        LifecycleStage.GEOMETRY: (
            MethodCapability.GEOMETRY_RECONSTRUCT,
            "metric_geometry",
        ),
        LifecycleStage.ARTICULATION: (MethodCapability.ARTICULATION_RECOVER, "ditto"),
        LifecycleStage.ASSEMBLY: (MethodCapability.ASSET_ASSEMBLE, "asset_assembly"),
        LifecycleStage.PHYSICAL_VALIDATION: (
            MethodCapability.ASSET_VALIDATE,
            "mujoco_physics_probe",
        ),
        LifecycleStage.DATASET: (MethodCapability.DATASET_INGEST, "lerobot_v3"),
        LifecycleStage.POLICY: (MethodCapability.POLICY_TRAIN, "lerobot_act"),
        LifecycleStage.WORLD_MODEL: (MethodCapability.WORLD_MODEL_TRAIN, "dino_wm"),
        LifecycleStage.BENCHMARK: (MethodCapability.EVALUATION_RUN, "libero"),
        LifecycleStage.ROS_DEPLOY: (MethodCapability.DEPLOYMENT_STAGE, "ros2_control"),
        LifecycleStage.FAILURE: (MethodCapability.FAILURE_ANALYZE, "rules_v1"),
        LifecycleStage.FEEDBACK: (
            MethodCapability.FEEDBACK_BUILD_DATASET,
            "hard_negative",
        ),
    }
    for stage, (capability, name) in desired.items():
        installed[stage] = _planned_method(capability, name)
    return installed


def _render_status(report: dict[str, Any], destination: Path) -> None:
    stages = report["readiness"]["stages"]
    names = [_STAGE_LABELS.get(item["stage"], item["stage"]) for item in stages]
    ready = [1 if item["status"] == "READY" else 0 for item in stages]
    blockers = [len(item["blockers"]) for item in stages]
    colors = ["#1373e6" if value else "#dc5a5a" for value in ready]
    figure, axis = plt.subplots(figsize=(11, 6.4), constrained_layout=True)
    positions = np.arange(len(names), dtype=np.float64)
    axis.hlines(positions, 0, 1, color="#e2e8f0", linewidth=1.2, zorder=1)
    axis.scatter(ready, positions, color=colors, s=90, zorder=2)
    for position, status, count in zip(positions, ready, blockers):
        label = "就绪" if status else f"{count} 项阻塞"
        offset = 0.04 if status == 0 else -0.04
        alignment = "left" if status == 0 else "right"
        axis.text(status + offset, position, label, va="center", ha=alignment, fontsize=9)
    axis.set_yticks(positions, names)
    axis.invert_yaxis()
    axis.set_xlim(-0.12, 1.12)
    axis.set_xticks((0, 1), ("阻塞", "就绪"))
    axis.set_title("cabinet2 生命周期证据门禁")
    axis.grid(axis="x", alpha=0.16)
    for name in ("top", "right", "left"):
        axis.spines[name].set_visible(False)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _render_metrics(report: dict[str, Any], destination: Path) -> None:
    """Render only measured metrics; unavailable stages never receive synthetic values."""

    camera = report["camera_metrics"]
    depth = report["depth_quality"]
    labels = ["/".join(Path(item["path"]).parts[-4:-2]) for item in depth]
    coverage = [100.0 * float(item["coverage_mean"]) for item in depth]
    minimum = [100.0 * float(item["coverage_min"]) for item in depth]

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].bar(
        ("已注册图像", "未注册图像"),
        (
            float(camera["registered_images"]),
            float(camera["selected_images"] - camera["registered_images"]),
        ),
        color=("#1373e6", "#cbd5e1"),
    )
    axes[0].set_title(
        "COLMAP 相机恢复\n"
        f"注册率={100.0 * float(camera['registration_rate']):.2f}%, "
        f"平均重投影误差={float(camera['mean_reprojection_error_px']):.3f} px"
    )
    axes[0].set_ylabel("图像数")
    axes[0].grid(axis="y", alpha=0.2)

    positions = np.arange(len(labels), dtype=np.float64)
    width = 0.36
    axes[1].bar(positions - width / 2, coverage, width, label="抽样均值")
    axes[1].bar(positions + width / 2, minimum, width, label="抽样最小值")
    axes[1].set_xticks(positions, labels, rotation=20, ha="right")
    axes[1].set_ylim(0.0, 100.0)
    axes[1].set_ylabel("有效原始深度像素 (%)")
    axes[1].set_title("打包 uint16 深度有效覆盖率\n单位与米制尺度尚未解析")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(frameon=False)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def _probe_detail(value: str) -> dict[str, Any] | str:
    """Preserve a machine probe as JSON without accepting malformed lookalikes."""

    final_line = value.rsplit("\n", 1)[-1]
    try:
        document = json.loads(final_line)
    except json.JSONDecodeError:
        return value
    return document if isinstance(document, dict) else value


def _method_installations(platform: PhysRealX, plan: Any) -> list[dict[str, Any]]:
    installations: list[dict[str, Any]] = []
    for binding in plan.stages:
        try:
            installation = platform.methods.inspect(binding.capability, binding.method.name)
        except ResourceNotFoundError:
            installations.append(
                {
                    "stage": binding.stage.value,
                    "capability": binding.capability.value,
                    "method": binding.method.name,
                    "installed": False,
                    "executable": False,
                    "ready": False,
                    "detail": "method is not installed",
                }
            )
            continue
        record = installation.to_dict()
        record.update(
            {
                "stage": binding.stage.value,
                "method": binding.method.name,
                "installed": True,
                "detail": _probe_detail(installation.detail),
            }
        )
        installations.append(record)
    return installations


def _accelerator_status() -> dict[str, Any]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return {
            "cuda_driver_ready": False,
            "nvidia_smi": None,
            "detail": "nvidia-smi is not installed",
        }
    try:
        completed = subprocess.run(
            [binary, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "cuda_driver_ready": False,
            "nvidia_smi": binary,
            "detail": str(error),
        }
    detail = completed.stdout.strip()[-4096:]
    return {
        "cuda_driver_ready": completed.returncode == 0,
        "nvidia_smi": binary,
        "returncode": completed.returncode,
        "detail": detail,
    }


def _runtime_attestations(workspace: Path, output: Path) -> list[dict[str, Any]]:
    """Copy the exact deployment records used by the experiment into its evidence set."""

    source_root = (workspace / "workers.d").resolve(strict=True)
    destination_root = output / "runtime-attestations"
    records: list[dict[str, Any]] = []
    for source in sorted(source_root.rglob("*.json")):
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"worker attestation is not a regular file: {source}")
        resolved = source.resolve(strict=True)
        if source_root not in resolved.parents:
            raise ValueError(f"worker attestation escapes workers.d: {source}")
        relative = resolved.relative_to(source_root)
        payload = resolved.read_bytes()
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        records.append(
            {
                "path": relative.as_posix(),
                "sha256": f"sha256:{hashlib.sha256(payload).hexdigest()}",
                "size_bytes": len(payload),
            }
        )
    if not records:
        raise ValueError("experiment workspace contains no worker deployment attestations")
    return records


def main() -> int:
    arguments = _arguments()
    source = arguments.source.expanduser().resolve(strict=True)
    workspace = arguments.workspace.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or next(output.iterdir(), None) is not None):
        raise FileExistsError(
            f"experiment output must be a new or empty directory; refusing overwrite: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    platform = PhysRealX.open(workspace)

    inspection = platform.assets.inspect_source(source)
    observations = platform.assets.ingest(
        source,
        resource_id="cabinet2.observations",
        version="1.0.0",
        display_name="cabinet2 captured observations",
        config={
            "coordinate_frame": "camera_unregistered",
            "depth_unit": "native_u16_unscaled",
            "include_candidates": False,
        },
    )
    camera = platform.assets.recover_cameras(
        observations,
        method="colmap",
        config={
            "states": [arguments.camera_state],
            "frame_stride": arguments.camera_frame_stride,
            "max_frames_per_stream": arguments.camera_max_frames,
            "execution_device": "cpu",
            "camera_policy": "single",
            "sequential_overlap": 5,
            "mapper_min_model_size": 3,
            "num_threads": 1,
            "random_seed": 0,
        },
    )
    camera_outputs = {item.name: item.artifact for item in camera.outputs}
    for name, artifact in camera_outputs.items():
        suffix = ".zip" if artifact.media_type == "application/zip" else ".json"
        platform.artifacts.materialize(artifact, output / f"{name}{suffix}")

    depth_files = sorted(source.glob("real/**/depth/*_depth.npy"))
    depth_quality = [_depth_quality(path, arguments.depth_frame_stride) for path in depth_files]
    plan = platform.readiness.plan(_lifecycle_methods(platform))
    installations = _method_installations(platform, plan)
    runtime_attestations = _runtime_attestations(workspace, output)
    readiness = platform.readiness.preflight(
        plan,
        roots={
            LifecycleEvidenceKey.CAPTURE: source,
            LifecycleEvidenceKey.OBSERVATIONS: observations,
            LifecycleEvidenceKey.CAMERA_RECONSTRUCTION: camera_outputs["camera_rig_calibration"],
        },
    )
    report = {
        "schema": "physrealx.cabinet2-lifecycle-evidence/1",
        "source": {
            "path": str(source),
            "files": inspection.total_files,
            "bytes": inspection.total_bytes,
            "streams": [item.model_dump(mode="json") for item in inspection.streams],
            "issues": [item.model_dump(mode="json") for item in inspection.issues],
        },
        "observations": observations.model_dump(mode="json"),
        "depth_quality": depth_quality,
        "camera_operation": camera.model_dump(mode="json"),
        "camera_metrics": dict(camera.metrics),
        "camera_scale_status": "unresolved",
        "runtime_environment": _accelerator_status(),
        "runtime_attestations": runtime_attestations,
        "method_installations": installations,
        "readiness": {
            "status": readiness.status.value,
            "stages": [
                {
                    "stage": item.binding.stage.value,
                    "method": item.binding.method.name,
                    "status": item.status.value,
                    "satisfied_evidence": [value.value for value in item.satisfied_evidence],
                    "blockers": [
                        {"code": blocker.code.value, "message": blocker.message}
                        for blocker in item.blockers
                    ],
                }
                for item in readiness.stages
            ],
        },
        "scientific_boundary": {
            "metric_alignment": (
                "visible marker-like patterns are not metrology: marker family, marker size, "
                "metric layout and disjoint train/holdout 3D correspondences are absent"
            ),
            "ditto": "missing metric, common-frame before/after point clouds and checkpoint",
            "geometry": (
                "missing resolved metric cameras, validated enhanced depth and a registered "
                "geometry reconstruction worker; historical ArtGS meshes are not accepted"
            ),
            "collision": (
                "CoACD requires newly reconstructed per-link visual meshes and identified "
                "articulation; historical meshes are not upstream evidence"
            ),
            "physics_sysid": "missing measured q, qdot, applied effort/wrench and holdout captures",
            "asset_assembly": (
                "missing closed visual/collision, articulation, identified body physics and "
                "identified joint physics evidence"
            ),
            "policy_world_model": "missing task episodes, action trajectories and split protocol",
            "ros2": "missing robot interface, URDF, controller, safety limits and hardware runtime",
        },
    }
    _json(output / "cabinet2_lifecycle_evidence.json", report)
    _render_status(report, output / "cabinet2_lifecycle_readiness.png")
    _render_metrics(report, output / "cabinet2_measured_metrics.png")
    print(json.dumps(report["camera_metrics"], ensure_ascii=False, sort_keys=True))
    print(f"report={output / 'cabinet2_lifecycle_evidence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
