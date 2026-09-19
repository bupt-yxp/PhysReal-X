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
    VersionedRef,
)
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import load_resource, read_array, read_json
from physrealx.integrations.feedback import episode_metrics
from physrealx.runtime import load_episode


def preview_demonstrations(store: ArtifactStore, datasets: tuple[VersionedRef, ...]) -> ArtifactRef:
    """Show actual collection outcomes, initial states and commanded joint motion."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    if not datasets:
        raise DomainValidationError("demonstration preview requires collected datasets")
    rows = [
        load_episode(store, summary)
        for reference in datasets
        for summary in load_resource(store, reference, DatasetManifest).episodes
    ]
    if len({row.episode_id for row in rows}) != len(rows):
        raise DomainValidationError("demonstration preview contains duplicate episodes")
    signals = [row.steps[0].observations["joint.position"] for row in rows]
    if len({(signal.shape, signal.unit) for signal in signals}) != 1 or any(
        signal.inline is None for signal in signals
    ):
        raise DomainValidationError("demonstration initial joint schemas must match")
    positions = np.asarray([signal.inline for signal in signals], dtype=float)
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5), constrained_layout=True)
    success = sum(row.success for row in rows)
    bars = axes[0].bar(
        ("Accepted", "Failed / retained"),
        (success, len(rows) - success),
        color=("#2a9d8f", "#e76f51"),
    )
    axes[0].bar_label(bars)
    axes[0].set_title("Expert collection outcomes")
    axes[0].set_ylabel("Episodes")
    center = positions.mean(axis=0)
    for row, position in zip(rows, positions, strict=True):
        axes[1].scatter(
            np.arange(len(position)) + 1,
            position - center,
            color="#2a9d8f" if row.success else "#e76f51",
            alpha=0.7,
            s=18,
        )
        times = [step.timestamp_seconds for step in row.steps]
        motion = np.asarray([step.observations["joint.position"].inline for step in row.steps])
        axes[2].plot(
            times,
            np.linalg.norm(motion - motion[0], axis=1),
            color="#2a9d8f" if row.success else "#e76f51",
            alpha=0.6,
        )
    axes[1].set_title(f"Recorded initial states ({len(np.unique(positions, axis=0))} distinct)")
    axes[1].set_xlabel("Robot joint")
    axes[1].set_ylabel(f"Offset from collection mean ({signals[0].unit})")
    axes[2].set_title("Robot joint displacement")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel(signals[0].unit)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    stream = io.BytesIO()
    try:
        figure.savefig(stream, format="png", dpi=120)
    finally:
        plt.close(figure)
    return store.put_bytes(stream.getvalue(), media_type="image/png")


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
        read_json(store, policy_update_ref, "physrealx.policy-update-assessment/2")
        if policy_update_ref is not None
        else None
    )
    if causal["failure_report"] != failure_report_ref.model_dump(mode="json") or calibration[
        "failure_report"
    ] != failure_report_ref.model_dump(mode="json"):
        raise DomainValidationError(
            "feedback preview reports do not share the same failure evidence"
        )
    if policy_update is not None:
        from physrealx.domain import VersionedRef

        for name, report in (("initial", initial), ("updated", updated)):
            linked = load_resource(
                store,
                VersionedRef.model_validate(policy_update[f"{name}_evaluation"]),
                EvaluationReport,
            )
            if linked != report:
                raise DomainValidationError(
                    "feedback preview evaluation differs from policy assessment"
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
    bars = axes[0, 0].bar(names, (initial.success_rate, updated.success_rate), color=colors)
    axes[0, 0].bar_label(
        bars,
        labels=[f"{sum(row['success'] for row in rows)}/{len(rows)}" for rows in evaluation_rows],
    )
    if policy_update:
        minimum = policy_update["guardrails"]["min_success_rate"]
        axes[0, 0].axhline(
            minimum,
            color="#666",
            linestyle=":",
        )
        axes[0, 0].text(
            0.98,
            minimum,
            f"min {minimum:.0%}",
            transform=axes[0, 0].get_yaxis_transform(),
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    axes[0, 0].set_ylim(0, 1.12)
    axes[0, 0].set_yticks(np.linspace(0, 1, 6))
    axes[0, 0].set_ylabel("Success rate")
    axes[0, 0].set_title("Independent closed-loop evaluation")
    axes[0, 0].grid(axis="y", alpha=0.22)

    task_units = {
        load_resource(store, episode.task, TaskSpec).action.signal.unit
        for report in (initial, updated)
        for episode in _episodes(store, report)
    }
    if len(task_units) != 1:
        plt.close(figure)
        raise DomainValidationError("feedback metrics cannot combine different joint units")
    unit = task_units.pop()
    for axis, metric, title in (
        (axes[0, 1], "control_target_error", "Final actuator target error"),
        (axes[0, 2], "tracking_rmse", "Actuator tracking RMSE"),
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
    for offset, contact_values, label, color in (
        (-0.18, before, "Initial", colors[0]),
        (0.18, after, "Updated", colors[1]),
    ):
        bars = axes[1, 0].bar(
            x + offset, contact_values / scale, width=0.36, label=label, color=color
        )
        axes[1, 0].bar_label(
            bars,
            labels=[
                f"{value:.2g}{unit}"
                for value, unit in zip(contact_values, ("N", "N", "s"), strict=True)
            ],
            fontsize=8,
        )
    axes[1, 0].set_ylim(0, 1.2)
    axes[1, 0].set_xticks(x, contact_names, rotation=12)
    axes[1, 0].set_ylabel("Relative to pair maximum")
    axes[1, 0].set_title("Contact metrics")
    axes[1, 0].legend()
    axes[1, 0].grid(axis="y", alpha=0.22)

    categories = Counter(row["category"] for row in failure["failures"])
    validated = Counter(
        row["causal_category"] for row in causal["paired_failures"] if row["causal_validation"]
    )
    labels = sorted(set(categories) | set(validated)) or ["none"]
    positions = np.arange(len(labels))
    axes[1, 1].bar(
        positions - 0.18,
        [categories[name] for name in labels],
        width=0.36,
        color="#457b9d",
        label="symptom",
    )
    axes[1, 1].bar(
        positions + 0.18,
        [validated.get(name, 0) for name in labels],
        width=0.36,
        color="#2a9d8f",
        label="causally validated",
    )
    axes[1, 1].set_xticks(positions, labels)
    axes[1, 1].set_title("Symptoms and policy intervention")
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
                f"Evaluation episodes: {len(evaluation_rows[0])} / {len(evaluation_rows[1])}",
                *(
                    (
                        f"Required episodes: {policy_update['guardrails']['min_episodes']}",
                        "Successful quality pairs: "
                        f"{policy_update['metrics']['quality_pair_count']}",
                        "Quality reference: "
                        + ("expert" if policy_update["reference_dataset"] else "initial policy")
                        + " (successful pairs)",
                    )
                    if policy_update
                    else ()
                ),
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


def _inline(value: DataValue | None, name: str) -> tuple[float, ...]:
    if value is None or value.inline is None:
        raise DomainValidationError(f"feedback replay requires inline signal {name!r}")
    return tuple(float(item) for item in value.inline)


def _replay_series(
    store: ArtifactStore, episodes: list[Episode]
) -> tuple[str, str, list[dict[str, Any]]]:
    import numpy as np

    task = load_resource(store, episodes[0].task, TaskSpec)
    names = [
        predicate.signal
        for predicate in task.success_conditions
        if predicate.signal.startswith("joint.") and predicate.signal.endswith(".position")
    ]
    name = names[0] if names else "joint.position"
    positions = [
        np.asarray([_inline(step.observations.get(name), name) for step in episode.steps])
        for episode in episodes
    ]
    column = int(np.argmax(np.ptp(np.concatenate(positions), axis=0)))
    unit = episodes[0].steps[0].observations[name].unit
    result = []
    for episode, position in zip(episodes, positions, strict=True):
        if any(step.observations[name].unit != unit for step in episode.steps):
            raise DomainValidationError("feedback replay cannot combine different joint units")
        time = np.asarray([step.timestamp_seconds for step in episode.steps], dtype=float)
        time -= time[0]
        force = np.asarray(
            [
                np.linalg.norm(_inline(step.observations.get("contact.force"), "contact.force"))
                for step in episode.steps
            ]
        )
        tracking = np.asarray(
            [
                np.sqrt(
                    np.mean(
                        (
                            np.asarray(_inline(step.action, "action"))
                            - np.asarray(
                                _inline(
                                    episode.steps[i + 1].observations.get("joint.position"),
                                    "joint.position",
                                )
                            )
                        )
                        ** 2
                    )
                )
                for i, step in enumerate(episode.steps[:-1])
            ]
        )
        result.append(
            {"time": time, "position": position[:, column], "force": force, "tracking": tracking}
        )
    label = "Object joint" if name != "joint.position" else f"Actuator {column}"
    return label, unit, result


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
    joint_label, unit, series = _replay_series(store, episodes)
    action_unit = load_resource(store, baseline.task, TaskSpec).action.signal.unit
    duration = max(float(row["time"][-1]) for row in series)
    row_specs = (
        ("position", joint_label, unit),
        ("force", "Contact force", "N"),
        ("tracking", "Tracking RMSE", action_unit),
    )
    limits = {}
    for key, _, _ in row_specs:
        values = np.concatenate([row[key] for row in series])
        lower, upper = float(values.min()), float(values.max())
        padding = max((upper - lower) * 0.08, 0.01)
        limits[key] = (lower - padding, upper + padding)
    rendered = []
    for time in np.linspace(0, duration, frames):
        figure, axes = plt.subplots(
            4,
            3,
            figsize=(15, 9),
            dpi=100,
            gridspec_kw={"height_ratios": [3.2, 1, 1, 1]},
            constrained_layout=True,
        )
        for column, (episode, label, data) in enumerate(zip(episodes, labels, series, strict=True)):
            index = int(
                np.clip(
                    np.searchsorted(data["time"], time, side="right") - 1, 0, len(episode.steps) - 1
                )
            )
            rgb = _array(store, episode, index, "camera.front.rgb")
            axes[0, column].imshow(rgb)
            status = "SUCCESS" if episode.success else "FAILURE"
            held = " · ended" if time > data["time"][-1] else ""
            axes[0, column].set_title(
                f"{label} · {status}\nRecorded front RGB · t={data['time'][index]:.2f}s{held}"
            )
            axes[0, column].axis("off")
            for row_index, (key, title, row_unit) in enumerate(row_specs, start=1):
                axis = axes[row_index, column]
                times = data["time"][1:] if key == "tracking" else data["time"]
                axis.plot(times, data[key], color=("#e76f51", "#2a9d8f", "#457b9d")[column])
                axis.axvline(time, color="black", linestyle="--", linewidth=1)
                axis.set_xlim(0, duration)
                axis.set_ylim(*limits[key])
                axis.grid(alpha=0.22)
                if column == 0:
                    axis.set_ylabel(f"{title} ({row_unit})")
                if row_index == 3:
                    axis.set_xlabel("Elapsed time (s)")
                else:
                    axis.tick_params(labelbottom=False)
        figure.suptitle(f"PhysReal-X feedback replay · t = {time:.2f} s", fontsize=16)
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
