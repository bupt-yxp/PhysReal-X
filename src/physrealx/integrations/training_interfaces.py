"""Validated hand-off contracts for official external training runtimes."""

from __future__ import annotations

import io
import textwrap
from collections.abc import Mapping
from typing import Literal

from pydantic import Field, model_validator

from physrealx.contracts import ArtifactStore
from physrealx.domain import (
    ArtifactRef,
    DomainModel,
    Episode,
    JSONValue,
    TaskSpec,
    VersionedRef,
)
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import load_resource
from physrealx.integrations.datasets import load_snapshot

TrainingInterfaceMethod = Literal["openvla_oft", "td_mpc2"]


class ExternalSourcePin(DomainModel):
    uri: str = Field(min_length=1, max_length=1024)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    license_spdx: str = Field(min_length=1, max_length=64)


class ExternalTrainingConfig(DomainModel):
    gpu_count: int = Field(default=1, ge=1, le=64)
    observation: Literal["state", "rgb"] = "state"


class TrainingInterfaceManifest(DomainModel):
    """One immutable, executable hand-off boundary to an external project."""

    schema_name: Literal["physrealx.training-interface/1"] = Field(
        default="physrealx.training-interface/1", alias="schema", serialization_alias="schema"
    )
    status: Literal["PREPARED"] = "PREPARED"
    method: TrainingInterfaceMethod
    snapshot: ArtifactRef
    dataset: VersionedRef
    task: VersionedRef
    source: ExternalSourcePin
    adapter_contract: str = Field(min_length=1, max_length=255)
    input_mapping: dict[str, tuple[str, ...]]
    train_episodes: int = Field(gt=0)
    train_transitions: int = Field(gt=0)
    command: tuple[str, ...] = Field(min_length=2)
    environment: dict[str, str]
    resource_requirements: dict[str, JSONValue]
    expected_outputs: tuple[str, ...] = Field(min_length=1)
    limitations: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_interface(self) -> TrainingInterfaceManifest:
        if not self.dataset.resolved:
            raise ValueError("training interface requires a resolved dataset")
        if not self.task.resolved:
            raise ValueError("training interface requires a resolved task")
        if not self.input_mapping or any(not values for values in self.input_mapping.values()):
            raise ValueError("training interface mappings must be non-empty")
        if len(self.command) != len(tuple(item for item in self.command if item.strip())):
            raise ValueError("training interface command tokens must be non-empty")
        return self


_SOURCES: dict[TrainingInterfaceMethod, ExternalSourcePin] = {
    "openvla_oft": ExternalSourcePin(
        uri="https://github.com/moojink/openvla-oft",
        revision="e4287e94541f459edc4feabc4e181f537cd569a8",
        license_spdx="MIT",
    ),
    "td_mpc2": ExternalSourcePin(
        uri="https://github.com/nicklashansen/tdmpc2",
        revision="e9f59321933cbc8e11a002b842adc7d4ffae8ff1",
        license_spdx="MIT",
    ),
}


def _task_for_snapshot(
    store: ArtifactStore, partitions: Mapping[str, list[Episode]]
) -> tuple[VersionedRef, TaskSpec]:
    train = partitions.get("train", [])
    if not train:
        raise DomainValidationError("training interface requires a non-empty train split")
    episode = train[0]
    task_ref = episode.task
    if any(item.task != task_ref for item in train):
        raise DomainValidationError(
            "one external training interface supports one task per snapshot"
        )
    return task_ref, load_resource(store, task_ref, TaskSpec)


def prepare_training_interface(
    store: ArtifactStore,
    snapshot_ref: ArtifactRef,
    *,
    method: TrainingInterfaceMethod,
    config: Mapping[str, JSONValue] | None = None,
) -> ArtifactRef:
    """Validate a snapshot and publish an exact external training hand-off."""
    if method not in _SOURCES:
        raise DomainValidationError(f"unsupported external training method: {method}")
    snapshot, dataset, partitions = load_snapshot(store, snapshot_ref)
    task_ref, _ = _task_for_snapshot(store, partitions)
    signals = {signal.name for signal in dataset.observation_schema}
    train_episodes = partitions["train"]
    transitions = sum(len(episode.steps) - 1 for episode in train_episodes)
    supplied = ExternalTrainingConfig.model_validate(config or {})

    if method == "openvla_oft":
        required = {"camera.front.rgb", "joint.position"}
        if not required <= signals:
            raise DomainValidationError(
                f"OpenVLA-OFT interface is missing signals: {sorted(required - signals)!r}"
            )
        mapping: dict[str, tuple[str, ...]] = {
            "full_image": ("camera.front.rgb",),
            "state": ("joint.position",),
            "task_description": ("TaskSpec.instruction",),
            "action": (dataset.action_schema.name,),
        }
        image_count = 1
        if "camera.wrist.rgb" in signals:
            mapping["wrist_image"] = ("camera.wrist.rgb",)
            image_count = 2
        manifest = TrainingInterfaceManifest(
            method=method,
            snapshot=snapshot_ref,
            dataset=snapshot.dataset,
            task=task_ref,
            source=_SOURCES[method],
            adapter_contract="physrealx.snapshot-to-openx-rlds/1",
            input_mapping=mapping,
            train_episodes=len(train_episodes),
            train_transitions=transitions,
            command=(
                "torchrun",
                "--standalone",
                "--nproc-per-node=${GPU_COUNT}",
                "vla-scripts/finetune.py",
                "--vla_path=openvla/openvla-7b",
                "--data_root_dir=${DATASET_DIR}",
                "--dataset_name=${DATASET_NAME}",
                f"--num_images_in_input={image_count}",
            ),
            environment={
                "DATASET_DIR": "required",
                "DATASET_NAME": "required",
                "GPU_COUNT": str(supplied.gpu_count),
            },
            resource_requirements={
                "gpu_count": supplied.gpu_count,
                "training_vram_gb": "27-80",
                "inference_vram_gb": "16-18",
                "dataset_format": "Open-X RLDS",
            },
            expected_outputs=(
                "OpenVLA-OFT checkpoint",
                "continuous action head",
                "proprio projector",
                "closed-loop evaluation report",
            ),
            limitations=(
                "Official OpenVLA-OFT training and inference have not run in this workspace.",
                "The external worker must materialize this snapshot through the declared "
                "RLDS adapter.",
            ),
        )
    else:
        if supplied.observation == "rgb" and "camera.front.rgb" not in signals:
            raise DomainValidationError("TD-MPC2 RGB interface requires camera.front.rgb")
        state_signals = tuple(
            name
            for name in (
                "joint.position",
                "joint.velocity",
                *(
                    signal
                    for signal in sorted(signals)
                    if signal.startswith("joint.") and signal.endswith(".position")
                ),
                *(
                    signal
                    for signal in sorted(signals)
                    if signal.startswith("joint.") and signal.endswith(".velocity")
                ),
            )
            if name in signals
        )
        state_signals = tuple(dict.fromkeys(state_signals))
        if not state_signals:
            raise DomainValidationError("TD-MPC2 interface requires numeric state observations")
        mapping = {
            "observation": state_signals,
            "action": (dataset.action_schema.name,),
            "reward": ("transition.reward",),
            "discount": ("transition.discount",),
            "terminated": ("step.is_terminal",),
        }
        if "camera.front.rgb" in signals:
            mapping["pixels"] = ("camera.front.rgb",)
        manifest = TrainingInterfaceManifest(
            method=method,
            snapshot=snapshot_ref,
            dataset=snapshot.dataset,
            task=task_ref,
            source=_SOURCES[method],
            adapter_contract="physrealx.snapshot-to-tdmpc2-buffer/1",
            input_mapping=mapping,
            train_episodes=len(train_episodes),
            train_transitions=transitions,
            command=(
                "python",
                "train.py",
                "task=physrealx",
                "data_dir=${DATASET_DIR}",
                "episodic=true",
                f"obs={supplied.observation}",
            ),
            environment={"DATASET_DIR": "required"},
            resource_requirements={
                "gpu_count": supplied.gpu_count,
                "recommended_vram_gb": 8,
                "recommended_ram_gb": 12,
                "dataset_format": "TD-MPC2 replay buffer",
            },
            expected_outputs=(
                "TD-MPC2 checkpoint",
                "latent dynamics model",
                "policy and value heads",
                "closed-loop evaluation report",
            ),
            limitations=(
                "Official TD-MPC2 training and evaluation have not run in this workspace.",
                "The external worker must provide the PhysReal-X task wrapper and buffer adapter.",
            ),
        )
    return store.put_json(manifest.model_dump(mode="json", by_alias=True))


def preview_training_interfaces(
    store: ArtifactStore, interfaces: tuple[ArtifactRef, ...]
) -> ArtifactRef:
    """Render prepared external interfaces without implying completed training."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle

    if not 1 <= len(interfaces) <= 4:
        raise DomainValidationError("training interface preview accepts one to four interfaces")
    manifests = []
    for reference in interfaces:
        if not store.verify(reference):
            raise DomainValidationError("training interface artifact is unavailable")
        manifests.append(TrainingInterfaceManifest.model_validate_json(store.get_bytes(reference)))

    figure, axes = plt.subplots(1, len(manifests), figsize=(7 * len(manifests), 6.2), dpi=120)
    axes_list = [axes] if len(manifests) == 1 else list(axes)
    labels = {"openvla_oft": "OpenVLA-OFT · VLA", "td_mpc2": "TD-MPC2 · world model RL"}
    for axis, manifest in zip(axes_list, manifests):
        axis.axis("off")
        mapped = "\n".join(
            textwrap.fill(
                f"{target} <- {', '.join(source)}",
                width=58,
                initial_indent="  ",
                subsequent_indent="    ",
            )
            for target, source in manifest.input_mapping.items()
        )
        requirements = "\n".join(
            textwrap.fill(
                f"{name}: {value}",
                width=58,
                initial_indent="  ",
                subsequent_indent="    ",
            )
            for name, value in manifest.resource_requirements.items()
        )
        expected = "\n".join(
            textwrap.fill(
                item,
                width=58,
                initial_indent="  • ",
                subsequent_indent="    ",
            )
            for item in manifest.expected_outputs
        )
        axis.text(
            0.03,
            0.97,
            "\n".join(
                (
                    labels[manifest.method],
                    f"Status: {manifest.status}",
                    f"Train: {manifest.train_episodes} episodes / "
                    f"{manifest.train_transitions} transitions",
                    "",
                    "Validated mapping",
                    mapped,
                    "",
                    f"Adapter: {manifest.adapter_contract}",
                    "Resources",
                    requirements,
                    "",
                    "Expected output",
                    expected,
                )
            ),
            va="top",
            fontsize=9.5,
            linespacing=1.28,
        )
        axis.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                transform=axis.transAxes,
                fill=False,
                linewidth=1.5,
                edgecolor="#1565c0",
            )
        )
    figure.suptitle("Official training interfaces", fontsize=16)
    figure.text(
        0.5,
        0.012,
        "PREPARED = snapshot mapping and launch contract validated; external training not executed",
        ha="center",
        color="#b23a48",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    stream = io.BytesIO()
    try:
        figure.savefig(stream, format="png", dpi=120)
    finally:
        plt.close(figure)
    return store.put_bytes(stream.getvalue(), media_type="image/png")


__all__ = [
    "TrainingInterfaceManifest",
    "TrainingInterfaceMethod",
    "prepare_training_interface",
    "preview_training_interfaces",
]
