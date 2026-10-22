"""Validated inputs shared by the live workbench and the six notebooks."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from examples.catalog import DEMOS, MODULES, dependencies_for


@dataclass(frozen=True)
class InputField:
    name: str
    label: str
    kind: str
    default: str | int
    help: str
    options: tuple[str, ...] = ()
    minimum: int = 0
    maximum: int = 0


FIELDS: dict[str, tuple[InputField, ...]] = {
    "real_assets_and_digital_twin": (
        InputField(
            "capture_dir",
            "ArtModeling 数据目录",
            "directory",
            "examples/real_assets_and_digital_twin/input/ArtModeling/laptop1",
            "包含 can.start、can.end 的服务器目录；可使用已挂载的数据。",
        ),
    ),
    "physical_asset_building": (
        InputField(
            "video_dir",
            "实拍视频目录 / 可选",
            "directory_optional",
            "",
            "留空时使用模块一资产的 real/video；目录需包含 img 与已审核 Mask。",
        ),
    ),
    "scene_and_task_generation": (
        InputField(
            "scene_type",
            "场景",
            "choice",
            "all",
            "可运行全部四类场景，或只运行选定的一类。",
            ("all", "home", "tabletop", "shelf", "industrial"),
        ),
        InputField(
            "instruction",
            "自然语言任务 / 单场景可选",
            "text",
            "",
            "留空使用场景默认任务；自定义任务须与所选目标物体相符。",
        ),
    ),
    "simulation_and_data_generation": (
        InputField(
            "physics_backend",
            "物理后端",
            "choice",
            "mujoco",
            "Isaac Sim 需要服务器单独安装对应运行环境。",
            ("mujoco", "isaac_sim"),
        ),
    ),
    "data_management_and_training": (
        InputField(
            "act_steps", "ACT 训练步数", "integer", 300, "策略训练步数。", minimum=1, maximum=100000
        ),
        InputField(
            "iql_steps",
            "IQL 训练步数",
            "integer",
            300,
            "离线强化学习步数。",
            minimum=1,
            maximum=100000,
        ),
        InputField(
            "world_steps",
            "世界模型训练步数",
            "integer",
            600,
            "DINO-WM 训练步数。",
            minimum=200,
            maximum=100000,
        ),
    ),
    "evaluation_and_feedback": (
        InputField(
            "collection_count",
            "恢复示范采样数",
            "integer",
            6,
            "失败回流采样数量。",
            minimum=1,
            maximum=1000,
        ),
        InputField(
            "heldout_count",
            "独立评测回合数",
            "integer",
            3,
            "候选门控使用的留出回合数。",
            minimum=1,
            maximum=1000,
        ),
        InputField(
            "heldout_seed",
            "留出评测种子",
            "integer",
            131000,
            "与示范采样条件保持隔离。",
            minimum=1,
            maximum=2**31 - 1,
        ),
    ),
}


def module_catalog() -> list[dict[str, Any]]:
    return [
        {
            "slug": demo.slug,
            "title": demo.title,
            "dependencies": list(demo.dependencies),
            "chain": [*dependencies_for((demo.slug,)), demo.slug],
            "fields": [asdict(field) for field in FIELDS[demo.slug]],
        }
        for demo in DEMOS
    ]


def _directory(value: str, root: Path, allowed_roots: tuple[Path, ...]) -> str:
    path = Path(value).expanduser()
    try:
        path = (path if path.is_absolute() else root / path).resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"输入目录不存在：{value}") from error
    if not path.is_dir():
        raise ValueError(f"输入不是目录：{value}")
    if not any(path == base or base in path.parents for base in allowed_roots):
        raise ValueError(f"输入目录不在允许范围内：{value}")
    return str(path)


def validate_job_request(
    payload: Any, root: Path, allowed_roots: tuple[Path, ...]
) -> tuple[tuple[str, ...], dict[str, dict[str, str | int]]]:
    if not isinstance(payload, dict) or set(payload) != {"module", "inputs"}:
        raise ValueError("任务需要 module 和 inputs")
    target = payload["module"]
    if target not in MODULES:
        raise ValueError("未知模块")
    chain = (*dependencies_for((target,)), target)
    raw_inputs = payload["inputs"]
    if not isinstance(raw_inputs, dict) or set(raw_inputs) - set(chain):
        raise ValueError("inputs 只能包含当前模块和其上游模块")
    validated: dict[str, dict[str, str | int]] = {}
    for module in chain:
        raw = raw_inputs.get(module, {})
        if not isinstance(raw, dict) or set(raw) - {field.name for field in FIELDS[module]}:
            raise ValueError(f"{module} 存在未知或非法参数")
        values: dict[str, str | int] = {}
        for field in FIELDS[module]:
            value = raw.get(field.name, field.default)
            if field.kind == "integer":
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not field.minimum <= value <= field.maximum
                ):
                    raise ValueError(f"{field.label} 超出允许范围")
            elif not isinstance(value, str) or len(value) > 4096:
                raise ValueError(f"{field.label} 必须是文本")
            elif field.kind == "choice" and value not in field.options:
                raise ValueError(f"{field.label} 不在可选值中")
            elif field.kind == "text":
                value = value.strip()
                if len(value) > 300 or any(ord(char) < 32 for char in value):
                    raise ValueError(f"{field.label} 最多 300 字且不能包含控制字符")
            elif field.kind in {"directory", "directory_optional"} and value:
                value = _directory(value, root, allowed_roots)
            elif field.kind == "directory":
                raise ValueError(f"{field.label} 不能为空")
            values[field.name] = value
        if (
            module == "scene_and_task_generation"
            and values["scene_type"] == "all"
            and values["instruction"]
        ):
            raise ValueError("自定义任务须选择单个场景")
        validated[module] = values
    return chain, validated


def module_input(module: str) -> dict[str, str | int]:
    """Return operator-validated parameters; ordinary Jupyter runs keep defaults."""
    path = os.environ.get("PHYSREALX_WEB_INPUTS")
    if not path:
        return {field.name: field.default for field in FIELDS[module]}
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    values = document["inputs"][module]
    if set(values) != {field.name for field in FIELDS[module]}:
        raise ValueError(f"{module} 的网页输入不完整")
    return cast(dict[str, str | int], values)
