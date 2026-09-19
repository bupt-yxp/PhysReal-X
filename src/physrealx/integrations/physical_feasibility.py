"""Simulation feasibility of prior assets, separate from independent physical validation."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from itertools import combinations

from pydantic import Field

from physrealx.contracts import ArtifactStore
from physrealx.domain import ArtifactRef, DomainModel, SceneEntityRole, SceneEntitySpec, SceneSpec
from physrealx.errors import DomainValidationError
from physrealx.integrations._artifacts import read_json
from physrealx.integrations.assembly import assembled_asset
from physrealx.integrations.mujoco import MujocoConfig, compile_scene


class FeasibilityConfig(DomainModel):
    sweep_samples: int = Field(default=181, ge=3, le=2001)
    penetration_tolerance_m: float = Field(default=0.001, ge=0, le=0.01)
    rollout_seconds: float = Field(default=1.0, gt=0, le=10)


def check_feasibility(
    store: ArtifactStore,
    assembly: ArtifactRef,
    *,
    config: FeasibilityConfig,
) -> ArtifactRef:
    import mujoco
    import numpy as np

    asset = assembled_asset(read_json(store, assembly, "physrealx.asset-assembly-bundle/1"), store)
    if len(asset.joints) != 1:
        raise DomainValidationError("feasibility sweep currently requires one articulated joint")
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
        effort_limits=(joint.effort_limit,),
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
    geoms = [
        (body.attrib["name"], geom.attrib["name"])
        for body in root.iter("body")
        for geom in body.findall("geom")
        if geom.get("contype") == "1"
    ]
    for (body_a, geom_a), (body_b, geom_b) in combinations(geoms, 2):
        if body_a != body_b:
            ET.SubElement(contact, "pair", geom1=geom_a, geom2=geom_b)
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(root, encoding="unicode"), assets=compiled.files
    )
    data = mujoco.MjData(model)
    address = int(model.joint(f"asset/{joint.name}").qposadr[0])
    positions = np.linspace(joint.lower_limit, joint.upper_limit, config.sweep_samples)
    rows = []
    for q in positions:
        mujoco.mj_resetData(model, data)
        data.qpos[address] = q
        mujoco.mj_forward(model, data)
        worst = min((float(c.dist) for c in data.contact), default=0.0)
        rows.append({"q": float(q), "penetration_m": max(0.0, -worst), "contacts": int(data.ncon)})
    rollouts = []
    for q in (joint.lower_limit, (joint.lower_limit + joint.upper_limit) / 2, joint.upper_limit):
        mujoco.mj_resetData(model, data)
        data.qpos[address] = q
        stable = True
        peak_speed = 0.0
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
        rollouts.append({"initial_q": q, "stable": stable, "peak_speed": peak_speed})
    maximum = max(row["penetration_m"] for row in rows)
    passed = maximum <= config.penetration_tolerance_m and all(r["stable"] for r in rollouts)
    return store.put_json(
        {
            "schema": "physrealx.physical-feasibility-report/1",
            "source_assembly": assembly.model_dump(mode="json"),
            "status": "PASSED" if passed else "FAILED",
            "backend": "mujoco",
            "backend_version": mujoco.__version__,
            "config": config.model_dump(mode="json"),
            "scope": "fixed_root_collision_sweep_and_zero_effort_rollouts",
            "physical_validation": False,
            "evidence_state": asset.evidence_state.value,
            "locked_range": [joint.lower_limit, joint.upper_limit],
            "max_penetration_m": maximum,
            "sweep": rows,
            "rollouts": rollouts,
            "adjacent_contact_filter_disabled_for_audit": True,
            "limitations": [
                "discrete sweep, not continuous collision certification",
                "does not validate real mass, friction or restitution",
                "no observed force replay; not a match to hand-driven video",
            ],
        }
    )
