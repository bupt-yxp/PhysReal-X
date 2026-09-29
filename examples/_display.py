"""Shared notebook setup and presentation; business calls remain explicit in examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def setup_notebook(module: str) -> tuple[Path, Path]:
    """Validate the editable checkout/kernel and prepare common notebook resources."""
    import importlib.util
    import os
    import sys

    root = Path(__file__).resolve().parents[1]
    if not module or Path(module).name != module or module in {".", ".."}:
        raise ValueError("示例名称必须是 examples 下的单个目录名。")
    directory = root / "examples" / module
    if not (directory / "demo.ipynb").is_file():
        raise ValueError(f"未找到示例：{module}")
    if Path(sys.prefix).name != "PhysReal-X":
        raise RuntimeError(f"请选择 PhysReal-X Conda 内核；当前解释器：{sys.executable}")
    package = importlib.util.find_spec("physrealx")
    expected = root / "src/physrealx/__init__.py"
    if package is None or package.origin is None or Path(package.origin).resolve() != expected:
        raise RuntimeError(
            "SDK 未指向当前项目；请在 PhysReal-X 环境中执行 pip install -e . 并重启内核。"
        )
    for name, value in {
        "CUDA_VISIBLE_DEVICES": "0",
        "MUJOCO_GL": "egl",
        "OMP_NUM_THREADS": "4",
        "MAX_JOBS": "3",
    }.items():
        os.environ.setdefault(name, value)
    if Path("/usr/local/cuda").is_dir():
        os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")

    import matplotlib.pyplot as plt

    figure = plt.figure()
    plt.close(figure)
    output = directory / "output"
    output.mkdir(parents=True, exist_ok=True)
    show_metrics({"Python": sys.version.split()[0], "解释器": sys.executable})
    return root, output


def show_metrics(values: Mapping[str, Any]) -> None:
    """Display a compact, escaped table; detailed evidence stays in SDK artifacts."""
    from html import escape

    from IPython.display import HTML, display

    rows = []
    for name, value in values.items():
        formatted = f"{value:.4g}" if isinstance(value, float) else str(value)
        rows.append(f"<tr><th>{escape(str(name))}</th><td>{escape(formatted)}</td></tr>")
    display(HTML("<table>" + "".join(rows) + "</table>"))


def show_rgbd(rgb: Any, depth: Any, destination: Path, *, mask: Any | None = None) -> None:
    """Present captured RGB, metric depth and an optional published foreground mask."""
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import display

    rgb, depth = np.asarray(rgb), np.asarray(depth)
    if rgb.shape != (*depth.shape, 3):
        raise ValueError("RGB and depth dimensions differ")
    if mask is not None and np.asarray(mask).shape != depth.shape:
        raise ValueError("foreground mask and depth dimensions differ")
    figure, axes = plt.subplots(
        1, 3 if mask is not None else 2, figsize=(12 if mask is not None else 9, 3)
    )
    try:
        axes[0].imshow(rgb)
        axes[0].set_title("Captured RGB")
        valid_depth = np.ma.masked_where(~np.isfinite(depth) | (depth <= 0), depth)
        image = axes[1].imshow(valid_depth, cmap="viridis")
        axes[1].set_title("Depth (m)")
        figure.colorbar(image, ax=axes[1])
        if mask is not None:
            foreground = np.asarray(mask, dtype=bool)
            axes[2].imshow(rgb)
            overlay = np.zeros((*foreground.shape, 4), dtype=float)
            overlay[foreground] = (0.0, 0.86, 0.78, 0.58)
            interior = foreground.copy()
            interior[1:-1, 1:-1] &= (
                foreground[:-2, 1:-1]
                & foreground[2:, 1:-1]
                & foreground[1:-1, :-2]
                & foreground[1:-1, 2:]
            )
            overlay[foreground & ~interior] = (0.0, 0.48, 0.46, 0.95)
            axes[2].imshow(overlay)
            axes[2].set_title("Foreground mask")
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(destination, dpi=140)
        display(figure)
    finally:
        plt.close(figure)


def show_reconstruction_evaluation(
    report: Mapping[str, Any], *, labels: Mapping[str, str] | None = None
) -> None:
    """Present published metrics and missing-part statuses; perform no evaluation here."""
    from html import escape

    from IPython.display import HTML, display

    if report.get("schema") != "physrealx.reconstruction-evaluation-report/2":
        raise ValueError("评估报告版本已更新，请重新执行重建质量评估单元。")
    labels = labels or {}

    statuses = {
        "measured": "已评估",
        "merged": "未单独分离",
        "missing": "缺失",
        "unobserved_or_merged": "未观察到独立运动",
        "unsegmented": "未分离部件",
        "joint_type_mismatch": "关节类型不匹配",
        "parent_mismatch": "父部件不匹配",
        "reference_motion_unobserved": "参考状态无运动",
        "predicted_motion_degenerate": "未恢复有效运动",
    }

    def table(headers: tuple[str, ...], rows: list[list[Any]]) -> None:
        def text(value: Any) -> str:
            return (
                "—" if value is None else f"{value:.5g}" if isinstance(value, float) else str(value)
            )

        header = "".join(f"<th>{escape(name)}</th>" for name in headers)
        body = "".join(
            "<tr>" + "".join(f"<td>{escape(text(value))}</td>" for value in row) + "</tr>"
            for row in rows
        )
        display(HTML(f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"))

    start, end = report["geometry"]["start"], report["geometry"]["end"]
    rows = [
        [label, start[key]["cd"], end[key]["cd"], scope]
        for key, label, scope in (
            ("whole", "CD-w：整体", "协议过滤后的重建表面与全部参考表面"),
            (
                "static",
                "CD-s：静态组合",
                "、".join(
                    labels.get(name, name) for name in start["static"].get("reference_links", ())
                ),
            ),
        )
    ]
    rows.extend(
        [
            labels.get(name, name),
            values["cd"],
            end["parts"][name]["cd"],
            statuses.get(values["status"], values["status"]),
        ]
        for name, values in start["parts"].items()
    )
    table(("几何", "起始 CD (10⁻³ m²)", "运动后 CD (10⁻³ m²)", "范围 / 状态"), rows)
    rows = []
    for name, values in report["joints"].items():
        angular = values["joint_type"] == "hinge"
        rows.append(
            [
                f"{labels[values['reference_child']]}关节"
                if values["reference_child"] in labels
                else name,
                "旋转" if angular else "平移",
                values["axis_ang_deg"],
                values["axis_pos"] if angular else None,
                values["part_motion"],
                values["part_motion_unit"],
                statuses.get(values["status"], values["status"]),
            ]
        )
    table(
        (
            "关节",
            "类型",
            "Axis Ang (°)",
            "Axis Pos (10⁻¹ m)",
            "Part Motion",
            "运动单位",
            "状态",
        ),
        rows,
    )


def show_animation(payload: bytes) -> None:
    """Embed a published GIF so the saved Notebook can play it without local paths."""
    from IPython.display import Image, display

    display(Image(data=payload, format="gif"))


def show_image(payload: bytes) -> None:
    """Embed one published PNG without coupling the Notebook to its renderer."""
    from IPython.display import Image, display

    display(Image(data=payload, format="png"))


def show_image_strip(images: Mapping[str, Any], destination: Path) -> None:
    """Present already-decoded previews without knowing their reconstruction method."""
    import matplotlib.pyplot as plt
    from IPython.display import display

    if not 1 <= len(images) <= 8:
        raise ValueError("a preview strip requires between one and eight images")
    figure, axes = plt.subplots(1, len(images), figsize=(3 * len(images), 3), squeeze=False)
    try:
        for axis, (label, pixels) in zip(axes[0], images.items()):
            axis.imshow(pixels)
            axis.set_title(label)
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(destination, dpi=140)
        display(figure)
    finally:
        plt.close(figure)
