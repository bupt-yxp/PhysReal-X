"""Focused visual evidence for dataset management and model training."""

from __future__ import annotations

import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.domain import (
    ArtifactRef,
    Episode,
    EvaluationReport,
    ModelManifest,
    TaskSpec,
    VersionedRef,
)
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import load_resource, read_array, read_json
from physrealx.integrations.datasets import load_snapshot, validate_dataset
from physrealx.integrations.world_model import WorldModelRolloutEvaluation


def _configure_cjk_font() -> None:
    """Use an installed CJK font when task instructions contain Chinese text."""
    from matplotlib import font_manager, rcParams

    candidates = [
        path
        for path in font_manager.findSystemFonts()
        if any(
            name in Path(path).name.lower() for name in ("notosanscjk", "droidsansfallback", "gbsn")
        )
    ]
    candidates.sort(
        key=lambda path: (
            "regular" not in Path(path).name.lower(),
            "medium" not in Path(path).name.lower(),
            path,
        )
    )
    if candidates:
        font_manager.fontManager.addfont(candidates[0])
        rcParams["font.family"] = font_manager.FontProperties(fname=candidates[0]).get_name()
        rcParams["axes.unicode_minus"] = False


def _observation_array(store: ArtifactStore, episode: Episode, index: int, name: str) -> Any | None:
    import numpy as np

    value = episode.steps[index].observations.get(name)
    if value is None:
        return None
    if value.artifact is not None:
        return read_array(store, value.artifact)
    if value.inline is not None:
        return np.asarray(value.inline).reshape(value.shape)
    return None


def _position_signal(episode: Episode) -> str | None:
    return next(
        (
            name
            for name in episode.steps[0].observations
            if name.startswith("joint.") and name.endswith(".position") and name != "joint.position"
        ),
        None,
    )


def _representative_frame(episode: Episode, signal: str | None) -> int:
    import numpy as np

    if signal is None:
        return len(episode.steps) // 2
    values = [step.observations[signal].inline for step in episode.steps]
    if any(value is None or len(value) != 1 for value in values):
        return len(episode.steps) // 2
    scalar = np.asarray([value[0] for value in values if value is not None], dtype=float)
    return int(np.argmax(np.abs(scalar - scalar[0])))


def _normalise(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if not finite.size:
        return np.zeros_like(array)
    lower, upper = float(finite.min()), float(finite.max())
    if upper - lower <= 1e-12:
        return np.zeros_like(array)
    return (array - lower) / (upper - lower)


def preview_dataset_snapshot(
    store: ArtifactStore,
    snapshot_ref: ArtifactRef,
    *,
    camera: str = "front",
) -> ArtifactRef:
    """Render modalities, synchronized sequences, labels and split membership."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    _configure_cjk_font()

    if not camera or "." in camera:
        raise DomainValidationError("dataset preview camera must be a simple camera name")
    snapshot, dataset, partitions = load_snapshot(store, snapshot_ref)
    quality = validate_dataset(store, dataset)
    episodes = [episode for values in partitions.values() for episode in values]
    if not episodes:
        raise DomainValidationError("dataset preview requires at least one episode")
    episode = next((item for item in episodes if item.success), episodes[0])
    task = load_resource(store, episode.task, TaskSpec)
    position_name = _position_signal(episode)
    frame_index = _representative_frame(episode, position_name)
    prefix = f"camera.{camera}."
    modalities = {
        name: _observation_array(store, episode, frame_index, prefix + name)
        for name in ("rgb", "depth", "mask", "events")
    }
    times = np.asarray([step.timestamp_seconds for step in episode.steps])
    position_values = []
    if position_name is not None:
        for step in episode.steps:
            value = step.observations[position_name].inline
            if value is None or len(value) != 1:
                raise DomainValidationError("dataset preview position signal must be scalar inline")
            position_values.append(float(value[0]))
    position = np.asarray(position_values) if position_values else np.zeros(len(times))
    contact = np.asarray(
        [
            step.observations["contact.force"].inline[0]
            if step.observations.get("contact.force") is not None
            and step.observations["contact.force"].inline is not None
            else 0.0
            for step in episode.steps
        ]
    )
    action = np.asarray(
        [
            np.linalg.norm(step.action.inline)
            if step.action is not None and step.action.inline is not None
            else 0.0
            for step in episode.steps
        ]
    )

    figure = plt.figure(figsize=(16, 8.2), dpi=100, constrained_layout=True)
    grid = figure.add_gridspec(2, 4, width_ratios=(1, 1, 1.35, 1.05))
    axes = [figure.add_subplot(grid[row, column]) for row in range(2) for column in range(2)]

    rgb = modalities["rgb"]
    if rgb is not None:
        axes[0].imshow(rgb)
    else:
        axes[0].text(0.5, 0.5, "Unavailable", ha="center", va="center")
    axes[0].set_title(f"{camera} / RGB")

    depth = modalities["depth"]
    if depth is not None:
        valid = depth[np.isfinite(depth) & (depth > 0)]
        image = axes[1].imshow(
            depth,
            cmap="viridis",
            vmin=float(np.percentile(valid, 2)) if valid.size else None,
            vmax=float(np.percentile(valid, 98)) if valid.size else None,
        )
        figure.colorbar(image, ax=axes[1], fraction=0.046, pad=0.03)
    else:
        axes[1].text(0.5, 0.5, "Unavailable", ha="center", va="center")
    axes[1].set_title("Metric depth")

    mask = modalities["mask"]
    if mask is not None:
        axes[2].imshow(mask, cmap="tab20", interpolation="nearest")
    else:
        axes[2].text(0.5, 0.5, "Unavailable", ha="center", va="center")
    axes[2].set_title("Instance / link mask")

    events = modalities["events"]
    if events is not None and events.ndim == 3 and events.shape[-1] == 2:
        signed = events[..., 0].astype(float) - events[..., 1].astype(float)
        limit = max(1.0, float(np.abs(signed).max()))
        axes[3].imshow(signed, cmap="coolwarm", vmin=-limit, vmax=limit)
    else:
        axes[3].text(0.5, 0.5, "Unavailable", ha="center", va="center")
    axes[3].set_title("Events (+/-)")
    for axis in axes:
        axis.axis("off")

    sequence_axis = figure.add_subplot(grid[0, 2:])
    sequence_axis.plot(times, _normalise(position), label="object joint")
    sequence_axis.plot(times, _normalise(action), label="action norm")
    sequence_axis.plot(times, _normalise(contact), label="contact force")
    sequence_axis.axvline(times[frame_index], color="black", linestyle="--", linewidth=1)
    sequence_axis.set_title("Synchronized observation-action-state sequence")
    sequence_axis.set_xlabel("Time (s)")
    sequence_axis.set_ylabel("Normalized value")
    sequence_axis.grid(alpha=0.22)
    sequence_axis.legend(loc="upper right")

    split_axis = figure.add_subplot(grid[1, 2])
    names = ("train", "validation", "test")
    success = [sum(episode.success for episode in partitions[name]) for name in names]
    failure = [len(partitions[name]) - count for name, count in zip(names, success)]
    split_axis.bar(names, success, label="success", color="#2a9d8f")
    split_axis.bar(names, failure, bottom=success, label="failure", color="#e76f51")
    split_axis.set_title("Leakage-safe episode split")
    split_axis.set_ylabel("Episodes")
    split_axis.legend()
    split_axis.grid(axis="y", alpha=0.22)

    summary_axis = figure.add_subplot(grid[1, 3])
    summary_axis.axis("off")
    camera_count = len(
        {
            signal.name.split(".")[1]
            for signal in dataset.observation_schema
            if signal.name.startswith("camera.") and signal.name.count(".") >= 2
        }
    )
    summary_axis.text(
        0,
        1,
        "\n".join(
            (
                "Dataset contract",
                f"Task: {task.instruction}",
                f"Episodes: {quality['episodes']}",
                f"Transitions: {quality['transitions']}",
                f"Observation signals: {len(dataset.observation_schema)}",
                f"Cameras: {camera_count}",
                f"Checked arrays: {quality['checked_arrays']:,}",
                f"Snapshot: {snapshot.record_id[:18]}…",
                "Quality: PASSED",
            )
        ),
        va="top",
        linespacing=1.55,
    )
    figure.suptitle("PhysReal-X data management snapshot", fontsize=16)
    stream = io.BytesIO()
    try:
        figure.savefig(stream, format="png", dpi=120)
    finally:
        plt.close(figure)
    return store.put_bytes(stream.getvalue(), media_type="image/png")


def preview_policy_training(
    store: ArtifactStore,
    model_ref: VersionedRef,
    model: ModelManifest,
    evaluation: EvaluationReport,
) -> ArtifactRef:
    """Render policy optimization evidence and independent closed-loop results."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if evaluation.model != model_ref:
        raise DomainValidationError("policy preview evaluation references a different model")
    losses = np.asarray(model.training_config.get("loss_curve", ()), dtype=float)
    if losses.ndim != 1 or not len(losses) or not np.isfinite(losses).all():
        raise DomainValidationError("policy preview requires a finite training loss curve")
    window = min(21, max(1, len(losses) // 12 * 2 + 1))
    smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
    offset = window - 1

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.4), dpi=110)
    axes[0].plot(losses, color="#8ecae6", alpha=0.55, linewidth=1, label="step loss")
    axes[0].plot(
        np.arange(offset, offset + len(smoothed)),
        smoothed,
        color="#1565c0",
        linewidth=2,
        label=f"moving mean ({window})",
    )
    if np.all(losses > 0):
        axes[0].set_yscale("log")
    axes[0].set_title(f"{model.trainer.name.upper()} optimization")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.22)
    axes[0].legend()

    count = int(evaluation.metrics.get("episode_count", 0))
    successes = int(evaluation.metrics.get("success_count", round(evaluation.success_rate * count)))
    axes[1].bar(
        ("success", "failure"),
        (successes, max(0, count - successes)),
        color=("#2a9d8f", "#e76f51"),
    )
    axes[1].set_ylim(0, max(1, count))
    axes[1].set_title(f"Independent closed-loop evaluation ({evaluation.success_rate:.0%})")
    axes[1].set_ylabel("Episodes")
    axes[1].grid(axis="y", alpha=0.22)

    axes[2].axis("off")
    visual_inputs = [
        signal.name for signal in model.observation_signals if signal.name.endswith(".rgb")
    ]
    values: Mapping[str, Any] = {
        "Trainer": model.trainer.name,
        "Visual input": ", ".join(visual_inputs) if visual_inputs else "none (state-only)",
        "State inputs": len(model.observation_signals) - len(visual_inputs),
        "Transitions": int(model.metrics.get("transitions", 0)),
        "Initial loss": model.metrics.get("initial_loss", float("nan")),
        "Final loss": model.metrics.get("final_loss", float("nan")),
        "Training action MAE": model.metrics.get("training_action_mae", float("nan")),
        "Evaluation episodes": count,
    }
    axes[2].text(
        0,
        1,
        "\n".join(
            f"{name}: {value:.5g}" if isinstance(value, float) else f"{name}: {value}"
            for name, value in values.items()
        ),
        va="top",
        linespacing=1.65,
    )
    figure.suptitle("PhysReal-X policy training and validation", fontsize=15)
    figure.tight_layout()
    stream = io.BytesIO()
    try:
        figure.savefig(stream, format="png", dpi=120)
    finally:
        plt.close(figure)
    return store.put_bytes(stream.getvalue(), media_type="image/png")


def _truth_rgb(
    store: ArtifactStore,
    reference: Episode,
    evaluation: WorldModelRolloutEvaluation,
    camera: str,
) -> list[Any]:
    result = []
    name = f"camera.{camera}.rgb"
    for index in range(
        evaluation.start_index + 1,
        evaluation.start_index + evaluation.horizon + 1,
    ):
        array = _observation_array(store, reference, index, name)
        if array is None:
            raise DomainValidationError(f"world-model preview requires {name!r}")
        result.append(array)
    return result


def preview_world_model_rollout(
    store: ArtifactStore,
    rollout_ref: ArtifactRef,
    evaluation_ref: ArtifactRef,
    reference: Episode,
    *,
    camera: str = "front",
    frames: int = 8,
    fps: int = 4,
) -> ArtifactRef:
    """Animate predicted RGB and state trajectories against aligned MuJoCo truth."""
    import matplotlib
    import numpy as np
    from PIL import Image

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if type(frames) is not int or not 2 <= frames <= 20:
        raise DomainValidationError("world-model preview frames must be between 2 and 20")
    if type(fps) is not int or not 1 <= fps <= 20:
        raise DomainValidationError("world-model preview fps must be between 1 and 20")
    document = read_json(store, rollout_ref, "physrealx.world-model-rollout/1")
    evaluation = WorldModelRolloutEvaluation.model_validate(
        read_json(store, evaluation_ref, "physrealx.world-model-rollout-evaluation/1")
    )
    if evaluation.rollout != rollout_ref or evaluation.reference_episode_id != reference.episode_id:
        raise DomainValidationError("world-model preview evidence is not time-aligned")
    predicted_rgb = read_array(store, ArtifactRef.model_validate(document["rgb"]))
    predicted_state = read_array(store, ArtifactRef.model_validate(document["states"]))
    truth_state = read_array(store, evaluation.reference_states)
    if (
        predicted_rgb.ndim != 4
        or predicted_rgb.shape[-1] != 3
        or predicted_rgb.dtype != np.uint8
        or predicted_state.shape != truth_state.shape
        or len(predicted_rgb) != evaluation.horizon
    ):
        raise DomainValidationError("world-model preview arrays violate the rollout contract")
    truth_rgb = _truth_rgb(store, reference, evaluation, camera)
    height, width = predicted_rgb.shape[1:3]
    resized_truth = [
        np.asarray(
            Image.fromarray(array.astype(np.uint8)).resize(
                (width, height), Image.Resampling.BILINEAR
            )
        )
        for array in truth_rgb
    ]
    position_names = [
        name
        for name in evaluation.signals
        if name == "joint.position" or name.endswith(".position")
    ]
    if not position_names:
        raise DomainValidationError("world-model preview requires at least one position signal")
    position_names = position_names[:2]
    selected = np.rint(
        np.linspace(0, evaluation.horizon - 1, min(frames, evaluation.horizon))
    ).astype(int)
    timestamps = np.asarray(evaluation.timestamp_seconds)
    rendered = []
    for index in selected:
        figure = plt.figure(figsize=(14, 7), dpi=100, constrained_layout=True)
        grid = figure.add_gridspec(2, 3, height_ratios=(1.15, 1))
        truth_axis = figure.add_subplot(grid[0, 0])
        prediction_axis = figure.add_subplot(grid[0, 1])
        error_axis = figure.add_subplot(grid[0, 2])
        truth_axis.imshow(resized_truth[index])
        prediction_axis.imshow(predicted_rgb[index])
        delta = resized_truth[index].astype(float) - predicted_rgb[index].astype(float)
        error = np.abs(delta).mean(axis=-1)
        error_image = error_axis.imshow(error, cmap="magma", vmin=0, vmax=255)
        figure.colorbar(error_image, ax=error_axis, fraction=0.046, pad=0.03)
        truth_axis.set_title("MuJoCo RGB truth")
        prediction_axis.set_title("DINO-WM RGB prediction")
        error_axis.set_title("RGB absolute error")
        for axis in (truth_axis, prediction_axis, error_axis):
            axis.axis("off")

        for plot_index, name in enumerate(position_names):
            axis = figure.add_subplot(grid[1, plot_index])
            signal = evaluation.signals[name]
            column = signal.column_start
            axis.plot(timestamps, truth_state[:, column], label="MuJoCo truth", color="#1565c0")
            axis.plot(
                timestamps,
                predicted_state[:, column],
                label="DINO-WM",
                color="#e76f51",
            )
            initial = reference.steps[evaluation.start_index].observations[name]
            if initial.inline is not None:
                axis.axhline(initial.inline[0], color="#6c757d", linestyle=":", label="persistence")
            axis.axvline(timestamps[index], color="black", linestyle="--", linewidth=1)
            axis.set_title(name)
            axis.set_xlabel("Time (s)")
            axis.set_ylabel(f"Position ({signal.unit})")
            axis.grid(alpha=0.22)
            axis.legend(fontsize=8)
        if len(position_names) == 1:
            figure.add_subplot(grid[1, 1]).axis("off")
        summary_axis = figure.add_subplot(grid[1, 2])
        summary_axis.axis("off")
        summary_axis.text(
            0,
            1,
            "\n".join(
                (
                    f"Horizon: {evaluation.horizon} steps",
                    f"Time: {timestamps[index]:.2f} s",
                    f"Frame: {index + 1}/{evaluation.horizon}",
                    "Reference: held-out MuJoCo episode",
                    "Prediction: action-conditioned rollout",
                )
            ),
            va="top",
            linespacing=1.55,
        )
        figure.suptitle("PhysReal-X world-model rollout", fontsize=16)
        stream = io.BytesIO()
        try:
            figure.savefig(stream, format="png", dpi=100)
        finally:
            plt.close(figure)
        rendered.append(Image.open(io.BytesIO(stream.getvalue())).convert("RGB"))
    output = io.BytesIO()
    rendered[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=rendered[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=False,
    )
    return store.put_bytes(output.getvalue(), media_type="image/gif")


__all__ = [
    "preview_dataset_snapshot",
    "preview_policy_training",
    "preview_world_model_rollout",
]
