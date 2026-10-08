"""Simulation feasibility of prior assets, separate from independent physical validation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

from pydantic import Field

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, DomainModel, SceneEntityRole, SceneEntitySpec, SceneSpec
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import read_json
from physrealx.integrations.assembly import assembled_asset
from physrealx.integrations.mujoco import MujocoConfig, compile_scene


class FeasibilityConfig(DomainModel):
    sweep_samples: int = Field(default=361, ge=3, le=2001)
    sweep_workers: int = Field(default=0, ge=0, le=64)
    adaptive_sweep_levels: int = Field(default=2, ge=0, le=8)
    adaptive_top_intervals: int = Field(default=8, ge=1, le=64)
    penetration_tolerance_m: float = Field(default=0.001, ge=0, le=0.01)
    minimum_clearance_m: float = Field(default=0.0002, gt=0, le=0.01)
    rollout_seconds: float = Field(default=1.0, gt=0, le=10)


def evaluate_asset_feasibility(
    store: ArtifactStore,
    asset: Any,
    *,
    config: FeasibilityConfig,
    retain_contact_constraints: bool = False,
    contact_constraint_tolerance_m: float | None = None,
    run_rollouts: bool = True,
) -> dict[str, Any]:
    """Evaluate articulation clearance and, when requested, dynamic stability."""
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import mujoco
    import numpy as np

    if len(asset.joints) != 1:
        raise DomainValidationError("feasibility sweep currently requires one articulated joint")
    constraint_tolerance = (
        config.penetration_tolerance_m
        if contact_constraint_tolerance_m is None
        else contact_constraint_tolerance_m
    )
    if not 0 <= constraint_tolerance <= config.penetration_tolerance_m:
        raise DomainValidationError(
            "contact constraint tolerance must be between zero and the acceptance tolerance"
        )
    joint = asset.joints[0]
    scene = SceneSpec(
        resource_id="physical.feasibility",
        version="1.0.0",
        display_name="Feasibility",
        entities=(
            SceneEntitySpec(name="asset", asset=asset.reference(), role=SceneEntityRole.OBJECT),
        ),
    )
    simulation = MujocoConfig(
        controlled_joints=(f"asset/{joint.name}",),
        initial_positions=(joint.lower_limit,),
        # Collision-only audits do not apply actuator effort.  Keep assets with
        # an unspecified optional effort limit auditable without mutating them.
        effort_limits=(joint.effort_limit if joint.effort_limit is not None else 1.0,),
        substeps=1,
    )
    compiled = compile_scene(scene, {"asset": asset}, store, simulation)
    root = ET.fromstring(compiled.xml)
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    # Audit every cross-link collision pair including parent/child. MuJoCo normally
    # filters adjacent bodies; exclusions must never hide geometry defects in this check.
    for child in list(contact):
        contact.remove(child)
    geom_links: dict[str, str] = {}
    for body in root.iter("body"):
        body_name = body.attrib["name"]
        for geom in body.findall("geom"):
            if geom.get("contype") != "1":
                continue
            name = geom.attrib["name"]
            geom_links[name] = body_name.split("/", 1)[-1]
    # Use MuJoCo's dynamic broad/mid phase at every position, including soft-limit
    # excursions. Disable only adjacent-body filtering; retain same-body filtering
    # and each geometry's original contact parameters. Explicit all-pairs lists
    # bypass native pruning and replace geometry contact parameters with pair defaults.
    # https://mujoco.readthedocs.io/en/stable/computation/#collision-detection
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    flags = option.find("flag")
    if flags is None:
        flags = ET.SubElement(option, "flag")
    flags.set("filterparent", "disable")
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(root, encoding="unicode"), assets=compiled.files
    )
    # Expand collision detection only. Equal margin/gap increments preserve the
    # original force activation distance; near contacts remain inactive.
    padding = np.maximum(0.0, config.minimum_clearance_m - model.geom_margin)
    model.geom_margin[:] += padding
    model.geom_gap[:] += padding
    pair_padding = np.maximum(0.0, config.minimum_clearance_m - model.pair_margin)
    model.pair_margin[:] += pair_padding
    model.pair_gap[:] += pair_padding
    address = int(model.joint(f"asset/{joint.name}").qposadr[0])
    geom_names = [
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, index))
        for index in range(model.ngeom)
    ]

    worker_state = threading.local()

    def evaluate_position(q: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Evaluate one immutable model position using thread-local mutable state."""
        data = getattr(worker_state, "data", None)
        if data is None:
            data = mujoco.MjData(model)
            worker_state.data = data
        mujoco.mj_resetData(model, data)
        data.qpos[address] = q
        mujoco.mj_forward(model, data)
        worst = 0.0
        clearance_lower_bound = config.minimum_clearance_m
        deepest_contact: dict[str, Any] | None = None
        contacts: list[dict[str, Any]] = []
        for item in data.contact:
            distance = float(item.dist)
            clearance_lower_bound = min(clearance_lower_bound, distance)
            worst = min(worst, distance)
            penetration = max(0.0, -distance)
            if penetration <= 0:
                continue
            names = (geom_names[int(item.geom1)], geom_names[int(item.geom2)])
            left, right = sorted(names)
            normal = item.frame[:3] if left == names[0] else -item.frame[:3]
            contact = {
                "geom_a": left,
                "link_a": geom_links[left],
                "geom_b": right,
                "link_b": geom_links[right],
                "penetration_m": penetration,
                "q": float(q),
                "position_m": item.pos.tolist(),
                "normal_a_to_b": normal.tolist(),
            }
            contacts.append(contact)
            if deepest_contact is None or penetration > deepest_contact["penetration_m"]:
                deepest_contact = dict(contact)
        return (
            {
                "q": float(q),
                "penetration_m": max(0.0, -worst),
                "clearance_lower_bound_m": clearance_lower_bound,
                "contacts": int(data.ncon),
                "deepest_contact": deepest_contact,
            },
            contacts,
        )

    worker_count = config.sweep_workers or min(8, max(1, (os.cpu_count() or 1) // 2))
    pool = (
        ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="mujoco-sweep")
        if worker_count > 1
        else None
    )

    def evaluate_positions(positions: Any) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
        values = [float(value) for value in positions]
        if pool is None or len(values) == 1:
            return [evaluate_position(value) for value in values]
        return list(pool.map(evaluate_position, values))

    positions = np.linspace(joint.lower_limit, joint.upper_limit, config.sweep_samples)
    evaluated = evaluate_positions(positions)
    # Refine only around observed high-contact intervals. The fixed base grid remains
    # the independent acceptance set; adaptive points protect narrow local maxima.
    for _ in range(config.adaptive_sweep_levels):
        ordered = sorted((value[0] for value in evaluated), key=lambda value: value["q"])
        intervals = sorted(
            zip(ordered[:-1], ordered[1:], strict=True),
            key=lambda pair: (
                max(pair[0]["penetration_m"], pair[1]["penetration_m"]),
                abs(pair[0]["contacts"] - pair[1]["contacts"]),
            ),
            reverse=True,
        )
        active = [
            pair
            for pair in intervals
            if max(pair[0]["penetration_m"], pair[1]["penetration_m"]) > 0
            or min(pair[0]["clearance_lower_bound_m"], pair[1]["clearance_lower_bound_m"])
            < config.minimum_clearance_m
            or pair[0]["contacts"] != pair[1]["contacts"]
        ][: config.adaptive_top_intervals]
        if not active:
            break
        known = {float(value[0]["q"]) for value in evaluated}
        midpoints = [
            (float(left["q"]) + float(right["q"])) / 2
            for left, right in active
            if (float(left["q"]) + float(right["q"])) / 2 not in known
        ]
        if not midpoints:
            break
        evaluated.extend(evaluate_positions(midpoints))

    rows = sorted((value[0] for value in evaluated), key=lambda value: value["q"])
    pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
    contact_constraints: list[dict[str, Any]] = []
    for _, contacts in evaluated:
        for item in contacts:
            left, right = item["geom_a"], item["geom_b"]
            penetration = float(item["penetration_m"])
            stats = pair_stats.setdefault(
                (left, right),
                {
                    "geom_a": left,
                    "link_a": geom_links[left],
                    "geom_b": right,
                    "link_b": geom_links[right],
                    "samples": 0,
                    "max_penetration_m": 0.0,
                },
            )
            stats["samples"] += 1
            if retain_contact_constraints and penetration > constraint_tolerance:
                contact_constraints.append(
                    {
                        **{key: value for key, value in item.items() if key != "position_m"},
                        "contact_position_m": item["position_m"],
                    }
                )
            if penetration > stats["max_penetration_m"]:
                stats["max_penetration_m"] = penetration
                stats["peak_q"] = item["q"]
                stats["contact_position_m"] = item["position_m"]
                stats["normal_a_to_b"] = item["normal_a_to_b"]
    rollouts = []
    if run_rollouts:
        import types

        rollout_positions = (
            joint.lower_limit,
            (joint.lower_limit + joint.upper_limit) / 2,
            joint.upper_limit,
        )
        native_bindings = isinstance(mujoco.MjModel, type) and isinstance(
            mujoco.mj_step, types.BuiltinFunctionType
        )
        if native_bindings:
            from mujoco import rollout as mujoco_rollout

            state_spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
            state_size = mujoco.mj_stateSize(model, state_spec)
            initial_state = np.empty((len(rollout_positions), state_size))
            for index, q in enumerate(rollout_positions):
                data = mujoco.MjData(model)
                mujoco.mj_resetData(model, data)
                data.qpos[address] = q
                mujoco.mj_forward(model, data)
                mujoco.mj_getState(model, data, initial_state[index], state_spec)
            rollout_data = [
                mujoco.MjData(model) for _ in range(min(worker_count, len(rollout_positions)))
            ]
            steps = int(config.rollout_seconds / simulation.timestep_seconds)
            with mujoco_rollout.Rollout(nthread=len(rollout_data)) as runner:
                states, _ = runner.rollout(
                    model,
                    rollout_data,
                    initial_state,
                    nstep=steps,
                    control_spec=mujoco.mjtState.mjSTATE_CTRL,
                )
            velocity_start = 1 + model.nq
            velocities = states[:, :, velocity_start : velocity_start + model.nv]
            for index, q in enumerate(rollout_positions):
                times = states[index, :, 0]
                stable = bool(
                    np.isfinite(states[index]).all()
                    and (len(times) < 2 or np.all(np.diff(times) > 0))
                )
                # Re-evaluate contacts at every actual dynamic pose, including
                # soft-limit excursions outside the nominal sweep interval.
                dynamic_rows = evaluate_positions(states[index, :, 1 + address]) if stable else []
                dynamic_clearance = min(
                    (r[0]["clearance_lower_bound_m"] for r in dynamic_rows), default=None
                )
                rollouts.append(
                    {
                        "initial_q": q,
                        "stable": stable,
                        "minimum_clearance_lower_bound_m": dynamic_clearance,
                        "clearance_passed": dynamic_clearance is not None
                        and dynamic_clearance >= config.minimum_clearance_m,
                        "contact_poses_evaluated": len(dynamic_rows),
                        "peak_speed": float(
                            np.linalg.norm(velocities[index], axis=1).max(initial=0.0)
                        ),
                    }
                )
        else:
            # Keep instrumented/custom bindings observable while production uses
            # MuJoCo's native thread-pool implementation above.
            def evaluate_rollout(q: float) -> dict[str, Any]:
                data = mujoco.MjData(model)
                mujoco.mj_resetData(model, data)
                data.qpos[address] = q
                stable = True
                peak_speed = 0.0
                dynamic_clearance = config.minimum_clearance_m
                contact_poses = 0
                for _ in range(int(config.rollout_seconds / simulation.timestep_seconds)):
                    mujoco.mj_step(model, data)
                    if (
                        any(w.number for w in data.warning)
                        or not np.isfinite(data.qpos).all()
                        or not np.isfinite(data.qvel).all()
                    ):
                        stable = False
                        break
                    peak_speed = max(peak_speed, float(np.linalg.norm(data.qvel)))
                    contact_row, _ = evaluate_position(float(data.qpos[address]))
                    dynamic_clearance = min(
                        dynamic_clearance, contact_row["clearance_lower_bound_m"]
                    )
                    contact_poses += 1
                return {
                    "initial_q": q,
                    "stable": stable,
                    "peak_speed": peak_speed,
                    "minimum_clearance_lower_bound_m": dynamic_clearance,
                    "clearance_passed": contact_poses > 0
                    and dynamic_clearance >= config.minimum_clearance_m,
                    "contact_poses_evaluated": contact_poses,
                }

            if pool is None:
                rollouts = [evaluate_rollout(q) for q in rollout_positions]
            else:
                rollouts = list(pool.map(evaluate_rollout, rollout_positions))
    if pool is not None:
        pool.shutdown(wait=True)
    maximum = max(row["penetration_m"] for row in rows)
    minimum_clearance = min(row["clearance_lower_bound_m"] for row in rows)
    passed = minimum_clearance >= config.minimum_clearance_m and (
        not run_rollouts or all(r["stable"] and r["clearance_passed"] for r in rollouts)
    )
    result = {
        "status": "PASSED" if passed else "FAILED",
        "locked_range": [joint.lower_limit, joint.upper_limit],
        "max_penetration_m": maximum,
        "minimum_clearance_lower_bound_m": minimum_clearance,
        "required_clearance_m": config.minimum_clearance_m,
        "acceptance_scope": "sampled_positive_clearance_and_dynamic_contact_stability",
        "continuous_collision_certified": False,
        "sweep": rows,
        "rollouts": rollouts,
        "rollouts_evaluated": run_rollouts,
        "sweep_strategy": {
            "base_samples": config.sweep_samples,
            "evaluated_samples": len(rows),
            "adaptive_levels": config.adaptive_sweep_levels,
            "adaptive_top_intervals": config.adaptive_top_intervals,
            "workers": worker_count,
        },
        "pair_broadphase": {
            "method": "mujoco_native",
            "parent_child_filter_disabled": True,
            "explicit_pairs": int(model.npair),
        },
        "collision_pairs": sorted(
            pair_stats.values(), key=lambda value: value["max_penetration_m"], reverse=True
        ),
    }
    if retain_contact_constraints:
        result["contact_constraints"] = contact_constraints
    return result


def check_feasibility(
    store: ArtifactStore,
    assembly: ArtifactRef,
    *,
    config: FeasibilityConfig,
) -> ArtifactRef:
    import mujoco

    asset = assembled_asset(read_json(store, assembly, "physrealx.asset-assembly-bundle/1"), store)
    evaluation = evaluate_asset_feasibility(store, asset, config=config)
    return store.put_json(
        {
            "schema": "physrealx.physical-feasibility-report/1",
            "source_assembly": assembly.model_dump(mode="json"),
            "status": evaluation["status"],
            "backend": "mujoco",
            "backend_version": mujoco.__version__,
            "config": config.model_dump(mode="json"),
            "scope": "fixed_root_collision_sweep_and_zero_effort_rollouts",
            "physical_validation": False,
            "evidence_state": asset.evidence_state.value,
            **evaluation,
            "adjacent_contact_filter_disabled_for_audit": True,
            "limitations": [
                "adaptive sampled sweep, not formal continuous collision certification",
                "does not validate real mass, friction or restitution",
                "no observed force replay; not a match to hand-driven video",
            ],
        }
    )
