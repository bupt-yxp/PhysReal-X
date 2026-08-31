#!/usr/bin/env python3
"""Run an actual cabinet2 MuJoCo -> LeRobot ACT -> world-model experiment.

This experiment deliberately distinguishes *executed software* from *measured
physics*.  The drawer geometry and dynamics are engineering priors derived from
the supplied cabinet2 package; they are randomized during simulation and are
never promoted to identified/validated physical parameters.

The script must run inside an environment containing LeRobot, PyTorch, MuJoCo,
NumPy, Pillow and safetensors.  A compatible invocation for the current server
is documented in ``docs/user-guide/data-management-and-training.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy
from PIL import Image, ImageDraw
from safetensors.torch import load_file, save_file
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SCHEMA = "physrealx.cabinet2-training-run/1"
FPS = 50
DT = 1.0 / FPS
EPISODE_STEPS = 80
STATE_NAMES = ("drawer_q", "drawer_qdot", "target_q", "mass", "damping", "friction")
# LeRobot 0.3.3 squeezes a scalar action before temporal chunking.  The explicit
# no-op channel keeps the action tensor shaped (batch, horizon, action_dim); only
# drawer_force is applied to MuJoCo and the invariant channel is checked at load.
ACTION_NAMES = ("drawer_force", "reserved_noop")
MUJOCO_XML = """
<mujoco model="physrealx_cabinet2_drawer_proxy">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.01" integrator="RK4" gravity="0 0 -9.81"/>
  <default>
    <joint limited="true" damping="0.5" frictionloss="0.2"/>
    <geom condim="4" friction="0.5 0.01 0.001"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.05" rgba="0.12 0.14 0.18 1"/>
    <body name="cabinet" pos="0 0 0.30">
      <geom name="cabinet_shell" type="box" size="0.25 0.21 0.28"
            rgba="0.32 0.24 0.16 1" contype="0" conaffinity="0"/>
      <body name="drawer" pos="0 -0.01 0">
        <joint name="drawer_slide" type="slide" axis="1 0 0" range="0 0.28"/>
        <geom name="drawer_box" type="box" size="0.16 0.18 0.08"
              pos="0.0 0 0" mass="2.0" rgba="0.70 0.48 0.24 1"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="drawer_force" joint="drawer_slide" gear="1" ctrlrange="-45 45"/>
  </actuator>
</mujoco>
"""


@dataclass(frozen=True)
class Dynamics:
    mass: float
    damping: float
    friction: float
    target: float
    q0: float


class StateWorldModel(nn.Module):
    """Action-conditioned delta model for drawer position and velocity."""

    def __init__(self, input_dim: int = 7, hidden_dim: int = 128) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


def _canonical_bytes(value: Any) -> bytes:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"{serialized}\n".encode()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(value))


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": path.stat().st_size,
    }


def _external_artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().as_posix(),
        "size_bytes": path.stat().st_size,
    }


def _tree_identity(root: Path) -> dict[str, Any]:
    files = sorted(item for item in root.rglob("*") if item.is_file())
    return {
        "root": root.as_posix(),
        "file_count": len(files),
        "size_bytes": sum(path.stat().st_size for path in files),
    }


def _sample_dynamics(rng: np.random.Generator, split: str) -> Dynamics:
    if split == "test_ood":
        mass = rng.uniform(4.0, 5.0)
        damping = rng.uniform(1.2, 1.6)
        friction = rng.uniform(0.5, 0.75)
    else:
        mass = rng.uniform(1.2, 4.0)
        damping = rng.uniform(0.2, 1.2)
        friction = rng.uniform(0.05, 0.5)
    return Dynamics(
        mass=float(mass),
        damping=float(damping),
        friction=float(friction),
        target=float(rng.uniform(0.22, 0.27)),
        q0=float(rng.uniform(0.0, 0.035)),
    )


def _make_model(dynamics: Dynamics) -> tuple[mujoco.MjModel, mujoco.MjData]:
    model = mujoco.MjModel.from_xml_string(MUJOCO_XML)
    data = mujoco.MjData(model)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drawer")
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "drawer_slide")
    dof_id = int(model.jnt_dofadr[joint_id])
    qpos_id = int(model.jnt_qposadr[joint_id])
    model.body_mass[body_id] = dynamics.mass
    model.dof_damping[dof_id] = dynamics.damping
    model.dof_frictionloss[dof_id] = dynamics.friction
    data.qpos[qpos_id] = dynamics.q0
    mujoco.mj_forward(model, data)
    return model, data


def _state(data: mujoco.MjData, dynamics: Dynamics) -> np.ndarray:
    return np.asarray(
        [
            data.qpos[0],
            data.qvel[0],
            dynamics.target,
            dynamics.mass,
            dynamics.damping,
            dynamics.friction,
        ],
        dtype=np.float32,
    )


def _expert_action(state: np.ndarray, rng: np.random.Generator) -> float:
    q, qdot, target, mass, damping, friction = (float(item) for item in state)
    desired_acceleration = 90.0 * (target - q) - 16.0 * qdot
    friction_compensation = friction * math.tanh((target - q) / 0.01)
    force = mass * desired_acceleration + damping * qdot + friction_compensation
    force += float(rng.normal(0.0, 0.12))
    return float(np.clip(force, -45.0, 45.0))


def _rollout_expert(dynamics: Dynamics, rng: np.random.Generator) -> dict[str, np.ndarray]:
    model, data = _make_model(dynamics)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[np.ndarray] = []
    dones: list[np.ndarray] = []
    for step in range(EPISODE_STEPS):
        state = _state(data, dynamics)
        force = _expert_action(state, rng)
        states.append(state)
        actions.append(np.asarray([force, 0.0], dtype=np.float32))
        for _ in range(2):
            data.ctrl[0] = force
            mujoco.mj_step(model, data)
        error = abs(float(data.qpos[0]) - dynamics.target)
        reward = math.exp(-30.0 * error) - 0.0002 * force * force
        done = step == EPISODE_STEPS - 1
        rewards.append(np.asarray([reward], dtype=np.float32))
        dones.append(np.asarray([done], dtype=np.bool_))
    return {
        "state": np.stack(states),
        "action": np.stack(actions),
        "reward": np.stack(rewards),
        "done": np.stack(dones),
    }


def generate_episodes(output_root: Path, seed: int, counts: dict[str, int]) -> dict[str, Any]:
    raw_dir = output_root / "raw_episodes"
    raw_dir.mkdir(parents=True, exist_ok=False)
    split_records: dict[str, list[dict[str, Any]]] = {}
    for split_index, (split, count) in enumerate(counts.items()):
        rng = np.random.default_rng(seed + split_index * 10_000)
        records: list[dict[str, Any]] = []
        for episode_index in range(count):
            dynamics = _sample_dynamics(rng, split)
            rollout = _rollout_expert(dynamics, rng)
            path = raw_dir / f"{split}-{episode_index:04d}.npz"
            np.savez_compressed(path, **rollout)
            final_state = rollout["state"][-1]
            records.append(
                {
                    "episode_id": f"{split}-{episode_index:04d}",
                    "artifact": _artifact(path, output_root),
                    "dynamics": dynamics.__dict__,
                    "expert_success": bool(
                        abs(float(final_state[0]) - dynamics.target) < 0.012
                        and abs(float(final_state[1])) < 0.12
                    ),
                }
            )
        split_records[split] = records
    manifest = {
        "schema": "physrealx.mujoco-episode-source/1",
        "status": "COMPLETED",
        "seed": seed,
        "fps": FPS,
        "episode_steps": EPISODE_STEPS,
        "state_names": list(STATE_NAMES),
        "action_names": list(ACTION_NAMES),
        "physics_status": "engineering_prior",
        "physics_warning": (
            "The cabinet2 package contains no measured control/wrench trajectories; "
            "sampled dynamics are "
            "simulation priors and must not be reported as identified real-object physics."
        ),
        "splits": split_records,
    }
    _write_json(output_root / "simulation_manifest.json", manifest)
    return manifest


def package_lerobot(output_root: Path, simulation_manifest: dict[str, Any]) -> Path:
    dataset_root = output_root / "lerobot_dataset"
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (2,),
            "names": list(STATE_NAMES[:2]),
        },
        "observation.environment_state": {
            "dtype": "float32",
            "shape": (len(STATE_NAMES) - 2,),
            "names": list(STATE_NAMES[2:]),
        },
        "action": {"dtype": "float32", "shape": (2,), "names": list(ACTION_NAMES)},
        "next.reward": {"dtype": "float32", "shape": (1,), "names": ["reward"]},
        "next.done": {"dtype": "bool", "shape": (1,), "names": ["done"]},
    }
    dataset = LeRobotDataset.create(
        repo_id="physrealx/cabinet2-drawer-engineering-v1",
        fps=FPS,
        root=dataset_root,
        robot_type="cabinet2_drawer_actuator_proxy",
        features=features,
        use_videos=False,
    )
    for record in simulation_manifest["splits"]["train"]:
        episode_path = output_root / record["artifact"]["path"]
        with np.load(episode_path) as episode:
            for frame_index in range(len(episode["state"])):
                dataset.add_frame(
                    {
                        "observation.state": episode["state"][frame_index, :2],
                        "observation.environment_state": episode["state"][frame_index, 2:],
                        "action": episode["action"][frame_index],
                        "next.reward": episode["reward"][frame_index],
                        "next.done": episode["done"][frame_index],
                    },
                    task="open the cabinet drawer to the requested position",
                )
        dataset.save_episode()
    info_path = dataset_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    _write_json(
        output_root / "dataset_manifest.json",
        {
            "schema": "physrealx.lerobot-dataset-run/1",
            "status": "COMPLETED",
            "repo_id": dataset.repo_id,
            "codebase_version": info["codebase_version"],
            "episodes": info["total_episodes"],
            "frames": info["total_frames"],
            "fps": info["fps"],
            "root": dataset_root.as_posix(),
            "info_artifact": _artifact(info_path, output_root),
        },
    )
    return dataset_root


def train_act(output_root: Path, dataset_root: Path, steps: int) -> Path:
    policy_root = output_root / "policy_act"
    log_path = output_root / "logs" / "policy_act.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "lerobot.scripts.train",
        "--dataset.repo_id=physrealx/cabinet2-drawer-engineering-v1",
        f"--dataset.root={dataset_root}",
        "--policy.type=act",
        "--policy.device=cpu",
        "--policy.push_to_hub=false",
        "--policy.chunk_size=16",
        "--policy.n_action_steps=1",
        "--policy.dim_model=64",
        "--policy.n_heads=4",
        "--policy.dim_feedforward=256",
        "--policy.n_encoder_layers=2",
        "--policy.n_decoder_layers=1",
        "--policy.latent_dim=8",
        "--policy.n_vae_encoder_layers=1",
        "--policy.dropout=0.05",
        "--policy.optimizer_lr=0.0003",
        f"--output_dir={policy_root}",
        f"--steps={steps}",
        "--batch_size=64",
        "--num_workers=0",
        "--eval_freq=0",
        "--log_freq=20",
        f"--save_freq={steps}",
        "--save_checkpoint=true",
        "--wandb.enable=false",
        "--seed=42",
    ]
    started = time.monotonic()
    with log_path.open("wb") as log_stream:
        completed = subprocess.run(
            command,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError(f"LeRobot ACT training failed with code {completed.returncode}:\n{tail}")
    pretrained = policy_root / "checkpoints" / "last" / "pretrained_model"
    model_path = pretrained / "model.safetensors"
    if not model_path.is_file():
        raise RuntimeError(f"LeRobot did not publish the expected checkpoint: {model_path}")
    losses: list[dict[str, float]] = []
    pattern = re.compile(r"step:(\d+)\s+.*?loss:([0-9.eE+-]+)")
    for match in pattern.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
        losses.append({"step": float(match.group(1)), "loss": float(match.group(2))})
    _write_json(
        output_root / "policy_training_report.json",
        {
            "schema": "physrealx.policy-training-report/1",
            "status": "COMPLETED",
            "method": "lerobot_act",
            "lerobot_version": _package_version("lerobot"),
            "steps": steps,
            "duration_seconds": time.monotonic() - started,
            "loss_samples": losses,
            "checkpoint": _artifact(model_path, output_root),
            "log": _artifact(log_path, output_root),
            "command": command,
        },
    )
    return pretrained


def _load_transitions(
    output_root: Path,
    records: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, ...]:
    inputs: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for record in records:
        with np.load(output_root / record["artifact"]["path"]) as episode:
            states = episode["state"].astype(np.float32)
            actions = episode["action"].astype(np.float32)
            if not np.allclose(actions[:, 1], 0.0):
                raise ValueError("reserved_noop action channel must remain zero")
            inputs.append(np.concatenate([states[:-1], actions[:-1, :1]], axis=1))
            targets.append(states[1:, :2] - states[:-1, :2])
    return np.concatenate(inputs), np.concatenate(targets)


def train_world_model(
    output_root: Path,
    manifest: dict[str, Any],
    max_epochs: int = 180,
) -> Path:
    torch.manual_seed(43)
    np.random.seed(43)
    train_x, train_y = _load_transitions(output_root, manifest["splits"]["train"])
    val_x, val_y = _load_transitions(output_root, manifest["splits"]["validation"])
    x_mean, x_std = train_x.mean(0), np.maximum(train_x.std(0), 1e-6)
    y_mean, y_std = train_y.mean(0), np.maximum(train_y.std(0), 1e-6)
    train_inputs = torch.from_numpy((train_x - x_mean) / x_std).float()
    train_targets = torch.from_numpy((train_y - y_mean) / y_std).float()
    val_inputs = torch.from_numpy((val_x - x_mean) / x_std).float()
    val_targets = torch.from_numpy((val_y - y_mean) / y_std).float()
    loader = DataLoader(TensorDataset(train_inputs, train_targets), batch_size=256, shuffle=True)
    model = StateWorldModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = math.inf
    patience = 0
    metrics: list[dict[str, float]] = []
    started = time.monotonic()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        sample_count = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch_x)
            loss = torch.nn.functional.smooth_l1_loss(prediction, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_loss += float(loss.item()) * len(batch_x)
            sample_count += len(batch_x)
        model.eval()
        with torch.no_grad():
            val_prediction = model(val_inputs)
            val_loss = float(torch.nn.functional.mse_loss(val_prediction, val_targets).item())
        train_loss /= sample_count
        metrics.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": val_loss,
            }
        )
        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {
                name: value.detach().clone() for name, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if patience >= 20 and epoch >= 50:
            break
    if best_state is None:
        raise RuntimeError("world model training did not produce a finite checkpoint")
    model.load_state_dict(best_state)
    model_path = output_root / "world_model" / "model.safetensors"
    model_path.parent.mkdir(parents=True, exist_ok=False)
    save_file(model.state_dict(), model_path)
    stats = {
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
    }
    stats_path = model_path.parent / "normalization.json"
    _write_json(stats_path, stats)
    metrics_path = output_root / "world_model_training_metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "validation_loss"])
        writer.writeheader()
        writer.writerows(metrics)
    _write_json(
        output_root / "world_model_training_report.json",
        {
            "schema": "physrealx.world-model-training-report/1",
            "status": "COMPLETED",
            "method": "action_conditioned_state_delta_mlp",
            "epochs": len(metrics),
            "best_validation_loss_normalized": best_val,
            "duration_seconds": time.monotonic() - started,
            "checkpoint": _artifact(model_path, output_root),
            "normalization": _artifact(stats_path, output_root),
            "metrics": _artifact(metrics_path, output_root),
        },
    )
    return model_path.parent


def _load_world_model(model_root: Path) -> tuple[StateWorldModel, dict[str, np.ndarray]]:
    model = StateWorldModel()
    model.load_state_dict(load_file(model_root / "model.safetensors"))
    model.eval()
    raw_stats = json.loads((model_root / "normalization.json").read_text(encoding="utf-8"))
    return model, {name: np.asarray(value, dtype=np.float32) for name, value in raw_stats.items()}


def _world_predict(
    model: StateWorldModel,
    stats: dict[str, np.ndarray],
    state: np.ndarray,
    action: float,
) -> np.ndarray:
    model_input = np.concatenate([state, np.asarray([action], dtype=np.float32)])
    normalized = (model_input - stats["x_mean"]) / stats["x_std"]
    with torch.no_grad():
        delta_norm = model(torch.from_numpy(normalized).float().unsqueeze(0)).squeeze(0).numpy()
    delta = delta_norm * stats["y_std"] + stats["y_mean"]
    prediction = state.copy()
    prediction[:2] += delta
    return prediction


def _evaluate_policy_episode(
    policy: ACTPolicy,
    dynamics: Dynamics,
    world_model: StateWorldModel,
    world_stats: dict[str, np.ndarray],
) -> dict[str, float | bool]:
    model, data = _make_model(dynamics)
    policy.reset()
    errors: list[float] = []
    world_errors: list[float] = []
    forces: list[float] = []
    success_step: int | None = None
    for step in range(EPISODE_STEPS):
        state = _state(data, dynamics)
        with torch.no_grad():
            action_tensor = policy.select_action(
                {
                    "observation.state": torch.from_numpy(state[:2]).unsqueeze(0),
                    "observation.environment_state": torch.from_numpy(state[2:]).unsqueeze(0),
                }
            )
        action = action_tensor.squeeze(0).cpu().numpy()
        force = float(np.clip(action[0], -45.0, 45.0))
        predicted_next = _world_predict(world_model, world_stats, state, force)
        for _ in range(2):
            data.ctrl[0] = force
            mujoco.mj_step(model, data)
        next_state = _state(data, dynamics)
        error = abs(float(next_state[0]) - dynamics.target)
        errors.append(error)
        world_errors.append(float(np.linalg.norm(predicted_next[:2] - next_state[:2])))
        forces.append(abs(force))
        if success_step is None and error < 0.012 and abs(float(next_state[1])) < 0.12:
            success_step = step
    final_state = _state(data, dynamics)
    success = (
        abs(float(final_state[0]) - dynamics.target) < 0.012 and abs(float(final_state[1])) < 0.12
    )
    return {
        "success": success,
        "final_position_error_m": abs(float(final_state[0]) - dynamics.target),
        "trajectory_position_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
        "world_model_state_rmse": float(np.sqrt(np.mean(np.square(world_errors)))),
        "completion_time_s": (
            float(success_step * DT) if success_step is not None else EPISODE_STEPS * DT
        ),
        "max_abs_force_n": max(forces),
        "force_limit_violations": float(sum(force > 45.0 + 1e-6 for force in forces)),
    }


def evaluate(
    output_root: Path,
    policy_root: Path,
    world_model_root: Path,
    seed: int,
    episodes_per_split: int,
) -> dict[str, Any]:
    policy = ACTPolicy.from_pretrained(policy_root)
    policy.eval()
    world_model, world_stats = _load_world_model(world_model_root)
    split_results: dict[str, Any] = {}
    for split_index, split in enumerate(("test_in_distribution", "test_ood")):
        sampler_name = "test_ood" if split == "test_ood" else "test"
        rng = np.random.default_rng(seed + 90_000 + split_index * 10_000)
        episodes: list[dict[str, Any]] = []
        for episode_index in range(episodes_per_split):
            dynamics = _sample_dynamics(rng, sampler_name)
            metrics = _evaluate_policy_episode(policy, dynamics, world_model, world_stats)
            episodes.append(
                {
                    "episode_id": f"{split}-{episode_index:04d}",
                    "seed": seed + 90_000 + split_index * 10_000,
                    "dynamics": dynamics.__dict__,
                    **metrics,
                }
            )
        success_count = sum(bool(item["success"]) for item in episodes)
        split_results[split] = {
            "episodes": episodes,
            "success_rate": success_count / len(episodes),
            "success_count": success_count,
            "mean_final_position_error_m": float(
                np.mean([float(item["final_position_error_m"]) for item in episodes])
            ),
            "mean_world_model_state_rmse": float(
                np.mean([float(item["world_model_state_rmse"]) for item in episodes])
            ),
            "mean_completion_time_s": float(
                np.mean([float(item["completion_time_s"]) for item in episodes])
            ),
            "force_limit_violations": int(
                sum(int(item["force_limit_violations"]) for item in episodes)
            ),
        }
    report = {
        "schema": "physrealx.cabinet2-fixed-benchmark/1",
        "status": "COMPLETED",
        "protocol": {
            "seed": seed,
            "episodes_per_split": episodes_per_split,
            "episode_steps": EPISODE_STEPS,
            "fps": FPS,
            "success_predicate": "abs(q-target)<0.012m and abs(qdot)<0.12m/s at final step",
            "force_limit_n": 45.0,
        },
        "policy": "lerobot_act",
        "world_model": "action_conditioned_state_delta_mlp",
        "physics_status": "engineering_prior",
        "splits": split_results,
    }
    _write_json(output_root / "benchmark_report.json", report)
    return report


def _package_version(name: str) -> str:
    from importlib.metadata import version

    return version(name)


def _read_world_metrics(path: Path) -> list[tuple[float, float]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [
            (float(row["epoch"]), float(row["validation_loss"])) for row in csv.DictReader(stream)
        ]


def _read_policy_metrics(report_path: Path) -> list[tuple[float, float]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    return [(float(item["step"]), float(item["loss"])) for item in report["loss_samples"]]


def _plot_line(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    points: Sequence[tuple[float, float]],
    color: str,
    title: str,
) -> None:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline="#cad5e3", width=2)
    draw.text((x0 + 16, y0 + 12), title, fill="#102a43")
    if len(points) < 2:
        draw.text((x0 + 16, y0 + 50), "no logged samples", fill="#64748b")
        return
    xs = np.asarray([item[0] for item in points], dtype=np.float64)
    ys = np.asarray([item[1] for item in points], dtype=np.float64)
    y_min, y_max = float(ys.min()), float(ys.max())
    if y_max <= y_min:
        y_max = y_min + 1.0
    mapped = []
    for x, y in zip(xs, ys):
        px = x0 + 30 + (x - xs.min()) / max(xs.max() - xs.min(), 1e-9) * (x1 - x0 - 60)
        py = y1 - 30 - (y - y_min) / (y_max - y_min) * (y1 - y0 - 80)
        mapped.append((float(px), float(py)))
    draw.line(mapped, fill=color, width=4, joint="curve")
    draw.text((x0 + 16, y1 - 24), f"last={ys[-1]:.5g}", fill=color)


def render_dashboard(output_root: Path, benchmark: dict[str, Any]) -> Path:
    canvas = Image.new("RGB", (1800, 980), "#f8fafc")
    draw = ImageDraw.Draw(canvas)
    draw.text((70, 45), "PhysReal-X cabinet2 executed training run", fill="#071d39")
    draw.text(
        (70, 88),
        "MuJoCo Episode  ->  LeRobot ACT  ->  Action-conditioned World Model  ->  Fixed Benchmark",
        fill="#456078",
    )
    _plot_line(
        draw,
        (70, 150, 850, 570),
        _read_policy_metrics(output_root / "policy_training_report.json"),
        "#1463ff",
        "LeRobot ACT training loss",
    )
    _plot_line(
        draw,
        (950, 150, 1730, 570),
        _read_world_metrics(output_root / "world_model_training_metrics.csv"),
        "#0f9f73",
        "World model validation loss",
    )
    draw.text((70, 630), "Fixed-seed closed-loop benchmark", fill="#102a43")
    y = 690
    for name, result in benchmark["splits"].items():
        draw.rounded_rectangle((70, y, 1730, y + 100), radius=16, fill="#ffffff", outline="#d5deea")
        draw.text((100, y + 18), name, fill="#102a43")
        summary = (
            f"success={result['success_rate'] * 100:.1f}%   "
            f"final error={result['mean_final_position_error_m'] * 1000:.2f} mm   "
            f"world RMSE={result['mean_world_model_state_rmse']:.5f}   "
            f"force violations={result['force_limit_violations']}"
        )
        draw.text((100, y + 56), summary, fill="#33506e")
        y += 125
    draw.text(
        (70, 940),
        "Physics status: engineering_prior (no measured force/torque evidence)",
        fill="#a14f16",
    )
    path = output_root / "training_dashboard.png"
    canvas.save(path)
    return path


def build_run_manifest(output_root: Path, source: Path, artgs_run: Path | None) -> dict[str, Any]:
    artifact_paths = [
        output_root / "simulation_manifest.json",
        output_root / "dataset_manifest.json",
        output_root / "policy_training_report.json",
        output_root / "world_model_training_report.json",
        output_root / "benchmark_report.json",
        output_root / "training_dashboard.png",
    ]
    source_files = [
        source / "sim" / name for name in ("info.json", "kinematics.json", "Material.json")
    ]
    artgs_report = artgs_run / "summary" / "artgs_training_report.json" if artgs_run else None
    implementation = Path(__file__).resolve()
    manifest = {
        "schema": SCHEMA,
        "status": "COMPLETED",
        "source": {
            "path": source.resolve().as_posix(),
            "engineering_prior_files": [
                {
                    "path": path.as_posix(),
                    "size_bytes": path.stat().st_size,
                }
                for path in source_files
                if path.is_file()
            ],
            "artgs_run": artgs_run.resolve().as_posix() if artgs_run is not None else None,
            "artgs_report": (
                _external_artifact(artgs_report)
                if artgs_report is not None and artgs_report.is_file()
                else None
            ),
        },
        "implementation": _external_artifact(implementation),
        "data_identity": {
            "raw_episodes": _tree_identity(output_root / "raw_episodes"),
            "lerobot_dataset": _tree_identity(output_root / "lerobot_dataset"),
        },
        "environments": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "mujoco": mujoco.__version__,
            "lerobot": _package_version("lerobot"),
        },
        "artifacts": [_artifact(path, output_root) for path in artifact_paths],
        "scientific_boundary": (
            "This is a fully executed simulation/training benchmark. "
            "It is not a real-robot validation and "
            "does not promote cabinet2 engineering priors to identified physical parameters."
        ),
    }
    _write_json(output_root / "run_manifest.json", manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artgs-run", type=Path)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--train-episodes", type=int, default=96)
    parser.add_argument("--validation-episodes", type=int, default=24)
    parser.add_argument("--test-episodes", type=int, default=24)
    parser.add_argument("--policy-steps", type=int, default=600)
    parser.add_argument("--benchmark-episodes", type=int, default=25)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    source = args.source.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    required = [source / "sim" / name for name in ("info.json", "kinematics.json", "Material.json")]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cabinet2 engineering-prior files are missing: {missing}")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite an existing training run: {output_root}")
    output_root.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    simulation = generate_episodes(
        output_root,
        args.seed,
        {
            "train": args.train_episodes,
            "validation": args.validation_episodes,
            "test": args.test_episodes,
            "test_ood": args.test_episodes,
        },
    )
    dataset_root = package_lerobot(output_root, simulation)
    policy_root = train_act(output_root, dataset_root, args.policy_steps)
    world_root = train_world_model(output_root, simulation)
    benchmark = evaluate(
        output_root,
        policy_root,
        world_root,
        args.seed,
        args.benchmark_episodes,
    )
    render_dashboard(output_root, benchmark)
    build_run_manifest(output_root, source, args.artgs_run)
    print(json.dumps({"status": "COMPLETED", "output_root": output_root.as_posix()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
