"""Visual evidence for evaluation, diagnosis, recovery, and policy update."""

from __future__ import annotations

import io
from collections import Counter
from typing import Any

from physrealx.contracts import ArtifactStore
from physrealx.domain import (
    ArtifactRef,
    DatasetManifest,
    DataValue,
    Episode,
    EvaluationReport,
    TaskSpec,
)
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import load_resource, read_array, read_json
from physrealx.integrations.feedback import episode_metrics
from physrealx.runtime import load_episode


def _episodes(store: ArtifactStore, report: EvaluationReport) -> list[Episode]:
    if report.dataset is None:
        raise DomainValidationError("feedback preview requires evaluation episode datasets")
    dataset = load_resource(store, report.dataset, DatasetManifest)
    return [load_episode(store, summary) for summary in dataset.episodes]


def _mean(rows: list[dict[str, Any]], name: str) -> float:
    import numpy as np

    return float(np.mean([float(row[name]) for row in rows]))


def preview_feedback_report(
    store: ArtifactStore,
    initial: EvaluationReport,
    updated: EvaluationReport,
    failure_report_ref: ArtifactRef,
    causal_report_ref: ArtifactRef,
    calibration_plan_ref: ArtifactRef,
    policy_update_ref: ArtifactRef | None = None,
) -> ArtifactRef:
    """Render the held-out comparison, failure evidence, and calibration decision."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    failure = read_json(store, failure_report_ref, "physrealx.failure-report/2")
    causal = read_json(store, causal_report_ref, "physrealx.causal-failure-report/1")
    calibration = read_json(store, calibration_plan_ref, "physrealx.physics-calibration-plan/1")
    policy_update = (
        read_json(store, policy_update_ref, "physrealx.policy-update-assessment/1")
        if policy_update_ref is not None
        else None
    )
    evaluation_rows = []
    for report in (initial, updated):
        rows = []
        for episode in _episodes(store, report):
            task = load_resource(store, episode.task, TaskSpec)
            rows.append(episode_metrics(store, episode, task))
        evaluation_rows.append(rows)

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=110, constrained_layout=True)
    names = ("Initial", "Updated")
    colors = ("#e76f51", "#2a9d8f")
    axes[0, 0].bar(names, (initial.success_rate, updated.success_rate), color=colors)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_ylabel("Success rate")
    axes[0, 0].set_title("Independent closed-loop evaluation")
    axes[0, 0].grid(axis="y", alpha=0.22)

    for axis, metric, title, unit in (
        (axes[0, 1], "control_target_error", "Final target error", "rad"),
        (axes[0, 2], "tracking_rmse", "Trajectory tracking RMSE", "rad"),
    ):
        values = [_mean(rows, metric) for rows in evaluation_rows]
        axis.bar(names, values, color=colors)
        axis.set_ylabel(unit)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.22)

    contact_names = ("Peak force", "Force variation", "Contact duration")
    before = np.asarray(
        [
            _mean(evaluation_rows[0], "peak_contact_force_n"),
            _mean(evaluation_rows[0], "contact_force_variation_rms_n"),
            _mean(evaluation_rows[0], "contact_duration_seconds"),
        ]
    )
    after = np.asarray(
        [
            _mean(evaluation_rows[1], "peak_contact_force_n"),
            _mean(evaluation_rows[1], "contact_force_variation_rms_n"),
            _mean(evaluation_rows[1], "contact_duration_seconds"),
        ]
    )
    scale = np.maximum(np.maximum(before, after), 1e-9)
    x = np.arange(len(contact_names))
    axes[1, 0].bar(x - 0.18, before / scale, width=0.36, label="Initial", color=colors[0])
    axes[1, 0].bar(x + 0.18, after / scale, width=0.36, label="Updated", color=colors[1])
    axes[1, 0].set_xticks(x, contact_names, rotation=12)
    axes[1, 0].set_ylabel("Relative to pair maximum")
    axes[1, 0].set_title("Contact stability")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.22)

    categories = Counter(row["category"] for row in failure["failures"])
    labels = sorted(categories) or ["none"]
    counts = [categories[name] for name in labels] if categories else [0]
    validated = Counter(
        row["causal_category"] for row in causal["paired_failures"] if row["causal_validation"]
    )
    axes[1, 1].bar(labels, counts, color="#457b9d", label="diagnosed")
    axes[1, 1].bar(
        labels,
        [validated.get(name, 0) for name in labels],
        color="#2a9d8f",
        label="causally validated",
    )
    axes[1, 1].set_title("Failure attribution")
    axes[1, 1].set_ylabel("Episodes")
    axes[1, 1].tick_params(axis="x", rotation=15)
    axes[1, 1].legend()
    axes[1, 1].grid(axis="y", alpha=0.22)

    axes[1, 2].axis("off")
    source = next(iter(failure.get("failures") or [{"source": "n/a"}]))["source"]
    axes[1, 2].text(
        0,
        1,
        "\n".join(
            (
                "Feedback evidence",
                f"Deployment source: {source}",
                f"Failed episodes: {len(failure['failures'])}",
                f"Paired interventions: {len(causal['paired_failures'])}",
                f"Validated policy causes: {causal['validated_count']}",
                f"Calibration: {calibration['status']}",
                f"Method: {calibration['method']}",
                f"Policy gate: {policy_update['status'] if policy_update else 'not assessed'}",
                "Holdout: independent trajectory",
            )
        ),
        va="top",
        linespacing=1.55,
    )
    figure.suptitle("PhysReal-X evaluation and feedback report", fontsize=16)
    stream = io.BytesIO()
    try:
        figure.savefig(stream, format="png", dpi=120)
    finally:
        plt.close(figure)
    return store.put_bytes(stream.getvalue(), media_type="image/png")


def _array(store: ArtifactStore, episode: Episode, index: int, name: str) -> Any:
    import numpy as np

    value = episode.steps[index].observations.get(name)
    if value is None:
        raise DomainValidationError(f"feedback replay requires {name!r}")
    if value.artifact is not None:
        return read_array(store, value.artifact)
    if value.inline is not None:
        return np.asarray(value.inline).reshape(value.shape)
    raise DomainValidationError(f"feedback replay signal {name!r} has no payload")


def _normalise(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=float)
    lower, upper = float(array.min()), float(array.max())
    if upper - lower <= 1e-12:
        return np.zeros_like(array)
    return (array - lower) / (upper - lower)


def _inline(value: DataValue | None, name: str) -> tuple[float, ...]:
    if value is None or value.inline is None:
        raise DomainValidationError(f"feedback replay requires inline signal {name!r}")
    return tuple(float(item) for item in value.inline)


def _position_series(episode: Episode) -> tuple[str, Any]:
    import numpy as np

    names = [
        name
        for name in episode.steps[0].observations
        if name.startswith("joint.") and name.endswith(".position") and name != "joint.position"
    ]
    name = sorted(names)[0] if names else "joint.position"
    values = np.asarray(
        [_inline(step.observations.get(name), name) for step in episode.steps], dtype=float
    )
    column = int(np.argmax(np.ptp(values, axis=0)))
    label = "object joint" if names else f"joint.position[{column}]"
    return label, values[:, column]


def preview_feedback_cycle(
    store: ArtifactStore,
    failed_ref: ArtifactRef,
    recovery_ref: ArtifactRef,
    updated_ref: ArtifactRef,
    *,
    frames: int = 8,
    fps: int = 3,
) -> ArtifactRef:
    """Animate failed policy, expert recovery, and updated policy from one condition."""
    import matplotlib
    import numpy as np
    from PIL import Image

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if type(frames) is not int or not 3 <= frames <= 20:
        raise DomainValidationError("feedback replay frames must be between 3 and 20")
    if type(fps) is not int or not 1 <= fps <= 12:
        raise DomainValidationError("feedback replay fps must be between 1 and 12")
    episodes = [
        Episode.model_validate_json(store.get_bytes(reference))
        for reference in (failed_ref, recovery_ref, updated_ref)
    ]
    baseline = episodes[0]
    if any(
        (row.seed, row.scene, row.task, row.backend, row.source)
        != (baseline.seed, baseline.scene, baseline.task, baseline.backend, baseline.source)
        for row in episodes[1:]
    ):
        raise DomainValidationError(
            "feedback replay episodes must share source, seed, scene, task, and backend"
        )
    if episodes[0].success or not episodes[1].success:
        raise DomainValidationError(
            "feedback replay requires a failed baseline and successful recovery"
        )
    if any(
        episodes[left].policy == episodes[right].policy for left, right in ((0, 1), (0, 2), (1, 2))
    ):
        raise DomainValidationError("feedback replay requires three distinct policies")
    labels = ("Failed policy", "Expert recovery", "Updated policy")
    rendered = []
    for frame in range(frames):
        progress = frame / (frames - 1)
        figure, axes = plt.subplots(2, 3, figsize=(15, 7.5), dpi=100, constrained_layout=True)
        for column, (episode, label) in enumerate(zip(episodes, labels, strict=True)):
            index = round(progress * (len(episode.steps) - 1))
            rgb = _array(store, episode, index, "camera.front.rgb")
            axes[0, column].imshow(rgb)
            status = "SUCCESS" if episode.success else "FAILURE"
            axes[0, column].set_title(f"{label} · {status}")
            axes[0, column].axis("off")

            position_label, position = _position_series(episode)
            times = np.asarray([step.timestamp_seconds for step in episode.steps])
            force = np.asarray(
                [
                    np.linalg.norm(_inline(step.observations.get("contact.force"), "contact.force"))
                    for step in episode.steps
                ],
                dtype=float,
            )
            action_error = np.asarray(
                [
                    np.linalg.norm(
                        np.asarray(_inline(step.action, "action"))
                        - np.asarray(
                            _inline(
                                episode.steps[i + 1].observations.get("joint.position"),
                                "joint.position",
                            )
                        )
                    )
                    for i, step in enumerate(episode.steps[:-1])
                ]
                + [0.0]
            )
            axes[1, column].plot(times, _normalise(position), label=position_label)
            axes[1, column].plot(times, _normalise(force), label="contact force")
            axes[1, column].plot(times, _normalise(action_error), label="tracking error")
            axes[1, column].axvline(times[index], color="black", linestyle="--", linewidth=1)
            axes[1, column].set_xlabel("Time (s)")
            axes[1, column].set_ylabel("Normalized value")
            axes[1, column].set_ylim(-0.05, 1.05)
            axes[1, column].grid(alpha=0.22)
            axes[1, column].legend(fontsize=8, loc="upper right")
        figure.suptitle(
            f"PhysReal-X feedback replay · same scene/task/seed · {progress:.0%}", fontsize=16
        )
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


__all__ = ["preview_feedback_cycle", "preview_feedback_report"]
