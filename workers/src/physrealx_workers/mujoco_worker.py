"""MuJoCo articulated-joint simulation worker."""

from __future__ import annotations

import json
from pathlib import Path

from physrealx.external_workers import WorkerProducedFile, WorkerRequest, WorkerResponse, run_worker
from physrealx_workers.common import config, input_path, is_probe, report_probe


def _handle(request: WorkerRequest, output_directory: Path) -> WorkerResponse:
    import mujoco
    import numpy as np

    input_path(request, "scene")
    input_path(request, "task")
    options = config(request)
    joint_type = str(options.get("joint_type", "slide"))
    if joint_type not in {"slide", "hinge"}:
        raise ValueError("joint_type must be 'slide' or 'hinge'")
    lower = float(options.get("lower", 0.0))
    upper = float(options.get("upper", 0.4 if joint_type == "slide" else 1.57))
    target = float(options.get("target", upper * 0.8))
    seconds = float(options.get("duration", 2.0))
    timestep = float(options.get("timestep", 0.002))
    kp = float(options.get("kp", 80.0))
    kv = float(options.get("kv", 8.0))
    if not lower < upper or not lower <= target <= upper:
        raise ValueError("joint limits and target are inconsistent")
    axis = "1 0 0" if joint_type == "slide" else "0 0 1"
    size = "0.22 0.18 0.04" if joint_type == "slide" else "0.22 0.04 0.28"
    xml = f"""
<mujoco model="physrealx_cabinet_joint">
  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast"/>
  <worldbody>
    <geom type="plane" size="1 1 0.1" rgba="0.2 0.25 0.3 1"/>
    <body name="cabinet_part" pos="0 0 0.35">
      <joint name="joint" type="{joint_type}" axis="{axis}" range="{lower} {upper}"
             damping="2" frictionloss="0.1" limited="true"/>
      <geom name="moving_part" type="box" size="{size}" mass="2.0" rgba="0.2 0.55 0.9 1"/>
    </body>
  </worldbody>
  <actuator><motor name="drive" joint="joint" gear="1"/></actuator>
</mujoco>
"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    steps = max(1, int(seconds / timestep))
    sample_stride = max(1, round(0.01 / timestep))
    samples: list[dict[str, float]] = []
    peak_force = 0.0
    energy = 0.0
    for step in range(steps + 1):
        error = target - float(data.qpos[0])
        command = float(np.clip(kp * error - kv * float(data.qvel[0]), -100.0, 100.0))
        data.ctrl[0] = command
        peak_force = max(peak_force, abs(command))
        energy += abs(command * float(data.qvel[0])) * timestep
        if step % sample_stride == 0 or step == steps:
            samples.append(
                {
                    "time": float(data.time),
                    "position": float(data.qpos[0]),
                    "velocity": float(data.qvel[0]),
                    "control": command,
                    "target": target,
                }
            )
        mujoco.mj_step(model, data)
    final_error = abs(target - float(data.qpos[0]))
    trajectory = {
        "schema": "physrealx.state-trajectory/1",
        "engine": f"mujoco-{mujoco.__version__}",
        "joint": {"type": joint_type, "lower": lower, "upper": upper, "target": target},
        "samples": samples,
    }
    report = {
        "schema": "physrealx.simulation-report/1",
        "stable": bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
        "within_limits": bool(lower - 1e-8 <= data.qpos[0] <= upper + 1e-8),
        "final_error": final_error,
        "peak_control": peak_force,
        "control_energy": energy,
        "sample_count": len(samples),
    }
    trajectory_path = output_directory / "state_trajectory.json"
    report_path = output_directory / "simulation_report.json"
    trajectory_path.write_text(json.dumps(trajectory, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    produced = {
        "state_trajectory": WorkerProducedFile(
            name="state_trajectory",
            contract="StateTrajectory",
            path=trajectory_path.name,
            media_type="application/json",
        ),
        "simulation_report": WorkerProducedFile(
            name="simulation_report",
            contract="SimulationReport",
            path=report_path.name,
            media_type="application/json",
        ),
    }
    return WorkerResponse(
        outputs=tuple(produced[item.name] for item in request.operation.outputs),
        metrics={
            "final_joint_error": final_error,
            "peak_control": peak_force,
            "control_energy": energy,
            "simulated_seconds": float(data.time),
        },
    )


def main() -> int:
    if is_probe():
        import mujoco
        import numpy

        if numpy.__version__ != "1.26.4":
            raise RuntimeError("MuJoCo worker requires NumPy 1.26.4")
        return report_probe(
            "mujoco",
            {
                "mujoco": mujoco.__version__,
                "numpy": numpy.__version__,
            },
        )
    return run_worker(_handle)


if __name__ == "__main__":
    raise SystemExit(main())
