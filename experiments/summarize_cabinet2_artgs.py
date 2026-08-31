#!/usr/bin/env python3
"""Summarize an executed cabinet2 ArtGS run without overstating its evidence."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw

SCHEMA = "physrealx.cabinet2-artgs-run/1"
MODEL_RELATIVE = Path("artgs/realscan/cabinet2/artgs")
_METRIC = re.compile(r"^(PSNR|SSIM|LPIPS)\s+([0-9.eE+-]+)", re.MULTILINE)
_REVISION = re.compile(r'^revision\s*=\s*"([0-9a-f]{40})"\s*$', re.MULTILINE)
_GENERATED_SOURCE_PARTS = {"__pycache__", "build", "dist"}
_GENERATED_SOURCE_SUFFIXES = {".o", ".pyc", ".pyd", ".so"}


def _canonical_bytes(value: Any) -> bytes:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{serialized}\n".encode()


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
    }


def _source_identity(source: Path) -> dict[str, Any]:
    """Describe the immutable source snapshot embedded in this repository."""

    source = source.resolve(strict=True)
    manifest = source / "SOURCE.toml"
    if not manifest.is_file():
        raise FileNotFoundError(f"embedded ArtGS source manifest is missing: {manifest}")
    manifest_text = manifest.read_text(encoding="utf-8")
    revision_match = _REVISION.search(manifest_text)
    if revision_match is None:
        raise ValueError("embedded ArtGS source manifest has no pinned upstream revision")
    source_files: list[Path] = []
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if (
            not path.is_file()
            or path.is_symlink()
            or any(part in _GENERATED_SOURCE_PARTS for part in relative.parts)
            or path.suffix in _GENERATED_SOURCE_SUFFIXES
        ):
            continue
        source_files.append(path)
    return {
        "layout": "embedded",
        "path": "algorithms/artgs",
        "revision": revision_match.group(1),
        "source_file_count": len(source_files),
        "source_manifest": {
            "size_bytes": manifest.stat().st_size,
        },
    }


def _ply_elements(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    with path.open("rb") as stream:
        for raw_line in stream:
            line = raw_line.decode("ascii", errors="strict").strip()
            if line == "end_header":
                return values
            if line.startswith("element "):
                _, name, count = line.split()
                values[name] = int(count)
    raise ValueError(f"PLY header has no end_header: {path}")


def _image_psnr(render: Path, target: Path) -> float:
    prediction = np.asarray(Image.open(render).convert("RGB"), dtype=np.float32) / 255.0
    reference = np.asarray(Image.open(target).convert("RGB"), dtype=np.float32) / 255.0
    if prediction.shape != reference.shape:
        raise ValueError(f"render and target shapes differ: {render.name}")
    mse = float(np.mean(np.square(prediction - reference)))
    return -10.0 * math.log10(max(mse, 1e-12))


def _rank_views(model_root: Path) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for state in ("-1", "0"):
        render_root = model_root / "train" / "ours_best" / "renders" / state
        target_root = model_root / "train" / "ours_best" / "gt" / state
        for render in sorted(render_root.glob("*.png")):
            target = target_root / render.name
            if not target.is_file():
                continue
            ranked.append(
                {
                    "state": state,
                    "frame": render.name,
                    "psnr": _image_psnr(render, target),
                    "render": render,
                    "target": target,
                }
            )
    if not ranked:
        raise FileNotFoundError("ArtGS run contains no paired rendered and target frames")
    return sorted(ranked, key=lambda item: float(item["psnr"]))


def _render_comparison(path: Path, ranked: list[dict[str, Any]], metrics: dict[str, float]) -> None:
    selected = [ranked[-1], ranked[len(ranked) // 2], ranked[0]]
    labels = ("Best training view", "Median training view", "Worst training view")
    canvas = Image.new("RGB", (1600, 1050), "#f7f9fc")
    draw = ImageDraw.Draw(canvas)
    draw.text((50, 34), "PhysReal-X cabinet2 ArtGS executed run", fill="#071d39")
    draw.text(
        (50, 75),
        (
            f"35,000 optimization steps | PSNR {metrics['PSNR']:.3f} dB | "
            f"SSIM {metrics['SSIM']:.4f} | LPIPS {metrics['LPIPS']:.4f}"
        ),
        fill="#36556f",
    )
    for column, (label, sample) in enumerate(zip(labels, selected)):
        x = 50 + column * 515
        draw.text(
            (x, 125),
            f"{label} | state={sample['state']} | PSNR={sample['psnr']:.2f} dB",
            fill="#102a43",
        )
        for row, (kind, image_path) in enumerate(
            (("Target", sample["target"]), ("Rendered", sample["render"]))
        ):
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((470, 360), Image.Resampling.LANCZOS)
            y = 175 + row * 400
            draw.rounded_rectangle(
                (x, y, x + 480, y + 380),
                radius=12,
                fill="#ffffff",
                outline="#cad5e3",
            )
            canvas.paste(image, (x + (480 - image.width) // 2, y + 28))
            draw.text((x + 14, y + 8), kind, fill="#456078")
    draw.text(
        (50, 990),
        "Evaluation boundary: reconstructed training views; "
        "no held-out camera set or real articulation GT.",
        fill="#9a4d14",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def summarize(run_root: Path, output_root: Path, source: Path) -> dict[str, Any]:
    model_root = run_root / MODEL_RELATIVE
    render_log = run_root / "logs" / "04-render.log"
    predicted_joint_types = (
        run_root / "artgs" / "realscan" / "cabinet2" / "pred" / "predicted_joint_types.json"
    )
    gaussian_path = model_root / "point_cloud" / "iteration_best" / "point_cloud.ply"
    deform_path = model_root / "deform" / "iteration_best" / "deform.pth"
    mesh_root = model_root / "train" / "ours_best" / "meshes"
    required = (render_log, predicted_joint_types, gaussian_path, deform_path, mesh_root)
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"ArtGS run is incomplete: {missing}")

    log_text = render_log.read_text(encoding="utf-8", errors="replace")
    metrics = {name: float(value) for name, value in _METRIC.findall(log_text)}
    if set(metrics) != {"PSNR", "SSIM", "LPIPS"}:
        raise ValueError("ArtGS render log does not contain final PSNR/SSIM/LPIPS")
    joint_types = json.loads(predicted_joint_types.read_text(encoding="utf-8"))["joint_types"]
    ranked = _rank_views(model_root)
    meshes = []
    for mesh in sorted(mesh_root.glob("*.ply")):
        meshes.append({**_artifact(mesh, run_root), "elements": _ply_elements(mesh)})

    comparison_path = output_root / "artgs_render_comparison.png"
    _render_comparison(comparison_path, ranked, metrics)
    report = {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "method": {
            "name": "artgs",
            "source_uri": "https://github.com/bupt-yxp/PhysReal-X/tree/main/algorithms/artgs",
            "source": _source_identity(source),
        },
        "run_root": run_root.as_posix(),
        "training": {
            "coarse_iterations": 10_000,
            "joint_type_iterations": 5_000,
            "articulated_iterations": 20_000,
            "total_iterations": 35_000,
            "joint_types": joint_types,
            "train_views": len(ranked),
        },
        "render_metrics": {
            "split": "training_views",
            "psnr_db": metrics["PSNR"],
            "ssim": metrics["SSIM"],
            "lpips": metrics["LPIPS"],
            "per_view_psnr_min_db": float(ranked[0]["psnr"]),
            "per_view_psnr_median_db": float(ranked[len(ranked) // 2]["psnr"]),
            "per_view_psnr_max_db": float(ranked[-1]["psnr"]),
        },
        "outputs": {
            "gaussian": {
                **_artifact(gaussian_path, run_root),
                "elements": _ply_elements(gaussian_path),
            },
            "deformation_checkpoint": _artifact(deform_path, run_root),
            "joint_prediction": _artifact(predicted_joint_types, run_root),
            "meshes": meshes,
        },
        "evaluation_boundary": (
            "Metrics are reconstruction metrics on training views because cabinet2 has no held-out "
            "camera split and no gt/trans.json articulation ground truth. This run proves "
            "optimization, rendering and geometry export; it does not prove metric-scale "
            "joint limits or physical dynamics."
        ),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root / "artgs_training_report.json"
    report_path.write_bytes(_canonical_bytes(report))
    return report


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "algorithms" / "artgs",
        help="embedded ArtGS source snapshot (defaults to algorithms/artgs)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    report = summarize(args.run_root.resolve(), args.output.resolve(), args.source.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
