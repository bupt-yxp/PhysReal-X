"""The six demonstrations and their actual data dependencies."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DemoModule:
    slug: str
    title: str
    guide: str
    dependencies: tuple[str, ...] = ()


DEMOS = (
    DemoModule(
        "real_assets_and_digital_twin",
        "真实资产与数字孪生",
        "user-guide/real-assets-and-digital-twin.md",
    ),
    DemoModule(
        "physical_asset_building",
        "物理资产构建",
        "user-guide/physical-assets.md",
        ("real_assets_and_digital_twin",),
    ),
    DemoModule(
        "scene_and_task_generation",
        "场景与任务生成",
        "user-guide/scenes-and-tasks.md",
    ),
    DemoModule(
        "simulation_and_data_generation",
        "物理仿真与多模态数据",
        "user-guide/simulation-and-data-generation.md",
        ("scene_and_task_generation",),
    ),
    DemoModule(
        "data_management_and_training",
        "数据管理与训练",
        "user-guide/data-management-and-training.md",
        ("simulation_and_data_generation",),
    ),
    DemoModule(
        "evaluation_and_feedback",
        "评测与反馈",
        "user-guide/evaluation-and-feedback.md",
        ("data_management_and_training",),
    ),
)

MODULES = tuple(demo.slug for demo in DEMOS)


def dependencies_for(modules: tuple[str, ...]) -> tuple[str, ...]:
    """Return transitive inputs in execution order, excluding requested modules."""
    required = set(modules)
    for demo in reversed(DEMOS):
        if demo.slug in required:
            required.update(demo.dependencies)
    return tuple(name for name in MODULES if name in required and name not in modules)


def dependents_of(modules: tuple[str, ...]) -> tuple[str, ...]:
    """Return only outputs invalidated by rerunning the selected modules."""
    affected = set(modules)
    for demo in DEMOS:
        if affected.intersection(demo.dependencies):
            affected.add(demo.slug)
    return tuple(name for name in MODULES if name in affected)
