"""Persistent multi-pose contact half-spaces and coupled sparse feasibility projection.

Feasibility restoration first minimizes the worst gap deficit, then the
area-weighted per-pair violations at that attained bound. Physical clearance is
hard for separated pairs; optional reserve can be consumed, never physical gap.
Separation branches persist across iterations;
an independent all-pair sweep remains the publication authority.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import sparse

from physrealx.reconstruction import joint_transform

from ._triangle_contacts import separating_axes


def _quadratic_projection(
    matrix: Any,
    metric: Any,
    target: Any,
    lo: Any,
    hi: Any,
    lower: Any,
    upper: Any,
    *,
    require_feasible: bool = True,
) -> tuple[Any, Any]:
    """Whiten a diagonal metric; verify the sparse QP in original mm units."""
    from time import monotonic

    import clarabel  # type: ignore[import-untyped]

    started = monotonic()
    scale = 1 / np.sqrt(metric)
    transformed = matrix.multiply(scale).tocsc()
    constraints = sparse.vstack([transformed, sparse.eye(len(target))], format="csc")
    lhs, rhs = np.r_[lower, lo / scale], np.r_[upper, hi / scale]
    row_scale = np.maximum(np.asarray(abs(constraints).max(axis=1).toarray()).ravel(), 1e-12)
    constraints = constraints.multiply((1 / row_scale)[:, None]).tocsc()
    lhs, rhs = lhs / row_scale, rhs / row_scale
    equal = np.isfinite(lhs) & np.isfinite(rhs) & (lhs == rhs)
    above, below = np.isfinite(rhs) & ~equal, np.isfinite(lhs) & ~equal
    # Equalities have their own cone; duplicating them as tight inequalities
    # makes the nearly active observation/area constraints poorly conditioned.
    cone_matrix = sparse.vstack(
        [constraints[equal], constraints[above], -constraints[below]], format="csc"
    )
    cone_rhs = np.r_[rhs[equal], rhs[above], -lhs[below]]
    cones = []
    if equal.any():
        cones.append(clarabel.ZeroConeT(int(equal.sum())))
    if above.any() or below.any():
        cones.append(clarabel.NonnegativeConeT(int(above.sum() + below.sum())))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.time_limit = 10.0
    settings.tol_feas = 1e-10
    settings.tol_gap_abs = settings.tol_gap_rel = 1e-9
    solver = clarabel.DefaultSolver(
        sparse.eye(len(target), format="csc"),
        -target / scale,
        cone_matrix,
        cone_rhs,
        cones,
        settings,
    )
    result = solver.solve()
    x = np.asarray(result.x) * scale if result.x is not None else np.full(len(target), np.nan)
    residual = (
        max(
            np.max(lower - matrix @ x, initial=0),
            np.max(matrix @ x - upper, initial=0),
            np.max(lo - x, initial=0),
            np.max(x - hi, initial=0),
        )
        if np.isfinite(x).all()
        else np.inf
    )
    audit = dict(
        status=str(result.status),
        solver="clarabel",
        iterations=result.iterations,
        solve_seconds=monotonic() - started,
        optimality_proven=str(result.status) == "Solved",
        linear_constraints_satisfied=bool(residual <= 1e-6),
        maximum_linear_constraint_violation_m=max(0.0, float(residual)) / 1000,
        restoration_slack_m=0.0,
        area_weighted_restoration_sum_m=0.0,
    )
    # Solver labels never replace the original unscaled constraint check.
    usable = str(result.status) in {"Solved", "AlmostSolved", "MaxIterations", "MaxTime"}
    # A stalled interior-point iterate can still propose a useful nonlinear step.
    # Only callers that independently certify the entire update may request it.
    usable |= not require_feasible and str(result.status) == "InsufficientProgress"
    return (
        x
        if usable and np.isfinite(x).all() and (not require_feasible or residual <= 1e-6)
        else None
    ), audit


class ContactProjection:
    def __init__(self, motion: Any, source: Any, graphs: Any, config: Any):
        self.motion, self.graphs, self.config = motion, graphs, config
        self.centers = {n: np.asarray(m.centroid) for n, m in source.items()}
        self.rows: dict[Any, Any] = {}
        self.branch_directions: dict[Any, Any] = {}
        self.coherent_keys: set[Any] = set()
        self.hard_safe_keys: set[Any] = set()
        self.self_planes: dict[Any, Any] = {}
        self.local_radii = {
            name: np.full(len(mesh.vertices), 0.0005) for name, mesh in source.items()
        }

    def learn_local_limits(self, name: str, requested: Any, realized: Any) -> Any:
        """Feed local validity restrictions back into the next coupled solve.

        These are incremental trust radii, not frozen geometry or a relaxation
        of observation/contact constraints. Ignore export noise and zero moves.
        """
        magnitude = np.abs(requested).max(axis=1)
        limited = (
            self.graphs[name]["editable"]
            & (magnitude > 1e-9)
            & (np.abs(realized - requested).max(axis=1) > np.maximum(1e-10, magnitude * 1e-6))
        )
        radii = self.local_radii[name]
        radii[limited] = np.maximum(
            1e-9, np.minimum(radii[limited] * 0.5, np.abs(realized[limited]).max(axis=1))
        )
        return limited

    def release_local_limits(self, limited: Any) -> None:
        """Allow safe nodes to recover motion rather than permanently locking them."""
        for name, radii in self.local_radii.items():
            safe = ~limited[name]
            radii[safe] = np.minimum(0.0005, radii[safe] * 1.5)

    def add_self_contacts(self, name: str, planes: Any) -> None:
        for a, b, normal, gap in planes:
            key = (name, tuple(a), tuple(b))
            if key not in self.self_planes:
                self.self_planes[key] = (name, a.copy(), b.copy(), normal.copy(), float(gap))

    def add(self, constraints: Any, meshes: Any, motion: Any) -> int:
        parent, child = motion["parent"], motion["child"]
        added = 0
        for q in np.unique(constraints["positions"]):
            select = constraints["positions"] == q
            af, bf, normals, raw_gaps = (
                constraints[k][select] for k in ("parent_faces", "child_faces", "normals", "gaps")
            )
            matrix = joint_transform("hinge", motion["axis"], motion["origin_m"], q)
            left = meshes[parent].vertices[af]
            right = meshes[child].vertices[bf] @ matrix[:3, :3].T + matrix[:3, 3]
            # Same canonical bulk-side decision at every pose; never choose a
            # fresh shortest push-through direction independently for each pair.
            reference = joint_transform("hinge", self.motion["axis"], self.motion["origin_m"], q)
            preferred = self.centers[parent] - (
                reference[:3, :3] @ self.centers[child] + reference[:3, 3]
            )
            preferred /= max(np.linalg.norm(preferred), 1e-15)
            for a, b, normal, raw_gap in zip(af, bf, normals, raw_gaps, strict=True):
                key = (float(q), *a.tolist(), *b.tolist())
                if key not in self.rows:
                    self.rows[key] = (a.copy(), b.copy(), normal.copy(), float(q))
                    self.branch_directions[key] = normal.copy()
                    added += 1
                if raw_gap >= self.config.minimum_clearance_m:
                    self.hard_safe_keys.add(key)
            # A separated neighbour of a crossing face may be on the wrong
            # side of the same surface. Independent nearest-side choices then
            # impose opposite inequalities on their shared vertices. Reconcile
            # the entire connected contact region, including persistent pairs.
            keys = [key for key in self.rows if key[0] == float(q)]
            region_a = np.array([self.rows[key][0] for key in keys])
            region_b = np.array([self.rows[key][1] for key in keys])
            nodes = np.column_stack([region_a, region_b + len(meshes[parent].vertices)])
            edges = np.column_stack([np.repeat(nodes[:, 0], 5), nodes[:, 1:].ravel()])
            size = len(meshes[parent].vertices) + len(meshes[child].vertices)
            graph = sparse.coo_matrix(
                (np.ones(len(edges)), (edges[:, 0], edges[:, 1])), shape=(size, size)
            )
            _, labels = sparse.csgraph.connected_components(graph, directed=False)
            regions = labels[nodes[:, 0]]
            left = meshes[parent].vertices[region_a]
            right = meshes[child].vertices[region_b] @ matrix[:3, :3].T + matrix[:3, 3]
            gaps, _ = separating_axes(left, right)
            needs_coherence = (gaps <= 0) | np.array([key in self.coherent_keys for key in keys])
            for region in np.unique(regions[needs_coherence]):
                ids = np.flatnonzero(regions == region)
                persistent = [
                    self.branch_directions[keys[i]] for i in ids if keys[i] in self.coherent_keys
                ]
                if persistent:
                    anchor = np.mean(persistent, axis=0)
                else:
                    _, initial = separating_axes(left[ids], right[ids], preferred)
                    anchor = initial.mean(axis=0)
                anchor /= max(np.linalg.norm(anchor), 1e-15)
                _, normals = separating_axes(left[ids], right[ids], anchor, minimum_alignment=0.9)
                for i, normal in zip(ids, normals, strict=True):
                    key = keys[i]
                    self.rows[key] = region_a[i], region_b[i], normal, float(q)
                    self.branch_directions[key] = anchor.copy()
                    self.coherent_keys.add(key)
        self.refit_supports(meshes, motion)
        return added

    def refit_supports(self, meshes: Any, motion: Any) -> None:
        parent, child = motion["parent"], motion["child"]
        # Refit supporting planes as the triangles bend. Preserve the initial
        # oriented hemisphere, not a narrow angular cone: an unfolding surface
        # may rotate substantially without changing its separation branch.
        keys = list(self.rows)
        af, bf, old_normals, poses = self.arrays()
        branches = np.asarray([self.branch_directions[key] for key in keys])
        for q in np.unique(poses):
            ids = np.flatnonzero(poses == q)
            matrix = joint_transform("hinge", motion["axis"], motion["origin_m"], q)
            left = meshes[parent].vertices[af[ids]]
            right = meshes[child].vertices[bf[ids]] @ matrix[:3, :3].T + matrix[:3, 3]
            new_gap, normals = separating_axes(
                left,
                right,
                branches[ids],
                minimum_alignment=1e-6,
            )
            # An actual separating plane with the required physical gap is
            # a certificate even if an obsolete branch points the other way.
            # Crossing pairs retain the coherent oriented hemisphere while
            # their local support direction is allowed to follow the surface.
            exact_gap, exact_normal = separating_axes(left, right)
            certified = exact_gap >= self.config.minimum_clearance_m
            new_gap[certified], normals[certified] = exact_gap[certified], exact_normal[certified]
            self.hard_safe_keys.update(keys[i] for i in ids[certified])
            old_gap = (left * old_normals[ids, None]).sum(2).min(1) - (
                right * old_normals[ids, None]
            ).sum(2).max(1)
            for i, normal in zip(
                ids[new_gap > old_gap + 1e-12], normals[new_gap > old_gap + 1e-12], strict=True
            ):
                self.rows[keys[i]] = af[i], bf[i], normal, float(q)

    def arrays(self) -> tuple[Any, ...]:
        if not self.rows:
            return np.empty((0, 3), int), np.empty((0, 3), int), np.empty((0, 3)), np.empty(0)
        rows = list(self.rows.values())
        return tuple(np.asarray([r[i] for r in rows]) for i in range(4))

    def gaps(self, meshes: Any, motion: Any) -> Any:
        af, bf, normals, poses = self.arrays()
        result = np.empty(len(af))
        for q in np.unique(poses):
            select = poses == q
            m = joint_transform("hinge", motion["axis"], motion["origin_m"], q)
            a = meshes[motion["parent"]].vertices[af[select]]
            b = meshes[motion["child"]].vertices[bf[select]] @ m[:3, :3].T + m[:3, 3]
            result[select] = (a * normals[select, None]).sum(2).min(1) - (
                b * normals[select, None]
            ).sum(2).max(1)
        return result

    def refresh_self_contacts(self, meshes: Any) -> None:
        """Refresh local feature planes on the CCD-validated current surface.

        A plane from an earlier deformation need not separate the same features
        after they rotate. It is a local linearization, not a physical boundary.
        Full deformation CCD remains the authority for every proposed update.
        """
        if not self.self_planes:
            return
        keys, records = list(self.self_planes), list(self.self_planes.values())
        left = np.array([meshes[n].vertices[np.resize(a, 3)] for n, a, b, normal, gap in records])
        right = np.array([meshes[n].vertices[np.resize(b, 3)] for n, a, b, normal, gap in records])
        sat_gap, sat_normal = separating_axes(left, right)
        center = left.mean(1) - right.mean(1)
        edges = np.concatenate(
            [np.roll(left, -1, axis=1) - left, np.roll(right, -1, axis=1) - right], axis=1
        )
        direction = edges / np.maximum(np.linalg.norm(edges, axis=2, keepdims=True), 1e-30)
        perpendicular = (
            center[:, None] - (center[:, None] * direction).sum(2)[..., None] * direction
        )
        axes = np.concatenate(
            [np.array([r[3] for r in records])[:, None], center[:, None], perpendicular], axis=1
        )
        lengths = np.linalg.norm(axes, axis=2)
        axes /= np.maximum(lengths[..., None], 1e-30)
        a = np.einsum("nvc,nac->nav", left, axes)
        b = np.einsum("nvc,nac->nav", right, axes)
        forward, backward = a.min(2) - b.max(2), b.min(2) - a.max(2)
        gaps = np.maximum(forward, backward)
        gaps[lengths < 1e-18] = -np.inf
        ids, rows = np.argmax(gaps, axis=1), np.arange(len(keys))
        best = gaps[rows, ids]
        normals = (
            axes[rows, ids]
            * np.where(forward[rows, ids] >= backward[rows, ids], 1.0, -1.0)[:, None]
        )
        use_sat = sat_gap > best
        best[use_sat], normals[use_sat] = sat_gap[use_sat], sat_normal[use_sat]
        for i, (name, af, bf, _normal, gap) in enumerate(records):
            if best[i] > 0:
                self.self_planes[keys[i]] = (
                    name,
                    af,
                    bf,
                    normals[i],
                    min(gap, float(best[i]) * 0.5),
                )

    def area_weights(self, meshes: Any) -> Any:
        """Normalize triangle area so refinement does not amplify a tiny patch."""
        af, bf, _normals, _poses = self.arrays()
        areas = []
        for name, faces in ((self.motion["parent"], af), (self.motion["child"], bf)):
            v = meshes[name].vertices[faces]
            areas.append(np.linalg.norm(np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0]), axis=1))
        weights = np.minimum(*areas)
        return weights / max(float(weights.mean()) if len(weights) else 0.0, 1e-20)

    def project(
        self,
        meshes: Any,
        motion: Any,
        proposed: Any,
        *,
        quadratic: bool = False,
        quadratic_metric: Any = None,
        reserve_m: float = 0.0,
        allow_inexact_proposal: bool = False,
    ) -> tuple[Any, Any]:
        """Project joint and vertex increments together, in millimetre units.

        Every pair has nine vertex inequalities, rather than one frozen support
        vertex. The joint differential is relinearized at the current hinge.
        """
        from time import monotonic

        import highspy  # type: ignore[import-untyped]

        if not np.isfinite(reserve_m) or reserve_m < 0:
            raise ValueError("contact reserve must be finite and nonnegative")
        self.refresh_self_contacts(meshes)
        af, bf, normals, poses = self.arrays()
        parent, child = motion["parent"], motion["child"]
        names = list(meshes)
        maps, offset = {}, 6
        observation_corrections = {}
        boundary_expansion = {}
        proposals = {n: np.asarray(p).copy() for n, p in zip(names, proposed[2:], strict=True)}
        for name in names:
            graph = self.graphs[name]
            involved = np.zeros(len(meshes[name].vertices), dtype=bool)
            involved[np.unique(af if name == parent else bf)] = True
            for n, a, b, _normal, _gap in self.self_planes.values():
                if n == name:
                    involved[np.r_[a, b]] = True
            edges = np.asarray(meshes[name].edges_unique)
            # Only the topological contact neighbourhood participates in the
            # sparse solve. Unrelated fit increments remain explicit
            # boundary values, not thousands of unnecessary projection unknowns.
            for _ in range(2):
                involved[np.unique(edges[involved[edges].any(1)])] = True
            proposals[name][~graph["editable"]] = 0
            seen = np.flatnonzero(graph["observed"] & graph["editable"])
            n = graph["observation_normal"][seen]
            current = meshes[name].vertices[seen] - graph["base"].cpu().numpy()[seen]
            distance = ((current + proposals[name][seen]) * n).sum(1)
            interval = graph["observation_interval"][seen]
            correction = np.clip(distance, interval[:, 0], interval[:, 1]) - distance
            proposals[name][seen] += correction[:, None] * n
            # A prior numerical residual must remain correctable after local
            # CCD narrows the trust region. This allowance restores the exact
            # observation slab; it never enlarges the slab or bypasses CCD.
            signed = (current * n).sum(1)
            repair = np.clip(signed, interval[:, 0], interval[:, 1]) - signed
            observation_corrections[name] = np.zeros_like(proposals[name])
            observation_corrections[name][seen] = repair[:, None] * n
            # A fixed two-ring domain can make a valid surface's projection
            # infeasible: an outside fit update collapses a boundary face while
            # its inside vertices have tight CCD trust radii. Expand precisely
            # the offending faces until a zero interior step has admissible
            # boundary values. Unrelated safe fit updates remain unchanged.
            initial_size = int((involved & graph["editable"]).sum())
            faces = np.asarray(meshes[name].faces)
            triangles = np.asarray(meshes[name].vertices)[faces]
            cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            area = np.linalg.norm(cross, axis=1)
            normal = cross / np.maximum(area[:, None], 1e-30)
            base = graph["base"].cpu().numpy()[faces]
            base_area = np.linalg.norm(
                np.cross(base[:, 1] - base[:, 0], base[:, 2] - base[:, 0]), axis=1
            )
            gradients = np.cross(
                np.roll(triangles, -1, axis=1) - np.roll(triangles, -2, axis=1), normal[:, None]
            )
            floor = np.maximum(0.1 * base_area, 0.5 * area)
            while True:
                frozen = proposals[name].copy()
                frozen[involved & graph["editable"]] = 0
                linear_area = area + (gradients * frozen[faces]).sum(axis=(1, 2))
                bad = linear_area < floor - 1e-15
                nodes = np.unique(faces[bad])
                nodes = nodes[graph["editable"][nodes] & ~involved[nodes]]
                if not len(nodes):
                    break
                involved[nodes] = True
            active_ids = np.flatnonzero(involved & graph["editable"])
            boundary_expansion[name] = len(active_ids) - initial_size
            ids = np.full(len(meshes[name].vertices), -1, dtype=int)
            ids[active_ids] = np.arange(len(active_ids)) * 3 + offset
            maps[name] = ids
            offset += 3 * len(active_ids)
        count = offset
        pair_count = len(af)
        physical_and_slack = count + pair_count
        target = np.zeros(count)
        target[:6] = np.concatenate(proposed[:2])
        for name, p in proposals.items():
            active = maps[name] >= 0
            start = maps[name][active]
            target[(start[:, None] + np.arange(3)).ravel()] = p[active].ravel()
        # Proposed values and joint translations use mm; axis tangents use
        # milliradians, preventing the six hinge variables dominating scaling.
        target *= 1000
        row_ids, columns, values = [], [], []
        lower = []
        current_gaps = self.gaps(meshes, motion)
        required = self.config.minimum_clearance_m
        violation = max(0.0, required - current_gaps.min()) if len(current_gaps) else 0.0
        row = 0
        for q in np.unique(poses):
            selection = np.flatnonzero(poses == q)
            n = normals[selection]
            left, right = af[selection], bf[selection]
            m = joint_transform("hinge", motion["axis"], motion["origin_m"], q)
            points = meshes[child].vertices[right]
            moved = points @ m[:3, :3].T + m[:3, 3]
            joint = []
            for d in range(6):
                axis, origin = np.array(motion["axis"]), np.array(motion["origin_m"])
                # The optimized variable is the unnormalized source-axis
                # tangent, not the currently normalized axis. Recover its
                # length before differencing, matching state() exactly.
                source_axis = np.asarray(self.motion["axis"], dtype=float)
                source_axis /= np.linalg.norm(source_axis)
                axis /= float(axis @ source_axis)
                (axis if d < 3 else origin)[d % 3] += 1e-6
                axis /= np.linalg.norm(axis)
                shifted = joint_transform("hinge", axis, origin, q)
                derivative = (points @ shifted[:3, :3].T + shifted[:3, 3] - moved) / 1e-6
                joint.append(-(derivative * n[:, None]).sum(2))
            joint = np.stack(joint, axis=2)
            for a in range(3):
                for b in range(3):
                    r = row + np.arange(len(selection))
                    row += len(selection)
                    gap = ((meshes[parent].vertices[left[:, a]] - moved[:, b]) * n).sum(1)
                    # A 2 µm solve margin covers the linear solver numerical residual;
                    # the actual nonlinear and independent gates stay exact.
                    floor = self.config.minimum_clearance_m
                    solve_margin = np.where(
                        current_gaps[selection] < self.config.minimum_clearance_m,
                        2e-6,
                        np.clip((current_gaps[selection] - floor) * 0.5, 0.0, 2e-6),
                    )
                    lower.extend((required + reserve_m + solve_margin - gap) * 1000)
                    for name, ids, coefficient in (
                        (parent, left[:, a], n),
                        (child, right[:, b], -n @ m[:3, :3]),
                    ):
                        index = maps[name][ids]
                        valid = index >= 0
                        for d in range(3):
                            row_ids.extend(r[valid])
                            columns.extend(index[valid] + d)
                            values.extend(coefficient[valid, d])
                    for d in range(6):
                        row_ids.extend(r)
                        columns.extend(np.full(len(r), d))
                        values.extend(joint[:, b, d])
                    # Restoration slack exists only for infeasible pairs.
                    # Every already-safe pair keeps the physical hard floor.
                    row_ids.extend(r)
                    columns.extend(count + selection)
                    values.extend(np.ones(len(selection)))
        for name, a, b, normal, required_gap in self.self_planes.values():
            for i in a:
                for j in b:
                    gap = float((meshes[name].vertices[i] - meshes[name].vertices[j]) @ normal)
                    lo = (required_gap - gap) * 1000
                    for vertex, sign in ((i, 1.0), (j, -1.0)):
                        index = maps[name][vertex]
                        if index >= 0:
                            for d in range(3):
                                row_ids.append(row)
                                columns.append(index + d)
                                values.append(sign * normal[d])
                        else:
                            lo -= sign * float(proposals[name][vertex] @ normal) * 1000
                    lower.append(lo)
                    row += 1
        area_rows = 0
        for name in names:
            mesh, graph = meshes[name], self.graphs[name]
            faces = np.asarray(mesh.faces)
            faces = faces[(maps[name][faces] >= 0).any(axis=1)]
            v = np.asarray(mesh.vertices)[faces]
            cross = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
            area = np.linalg.norm(cross, axis=1)
            normal = cross / np.maximum(area[:, None], 1e-30)
            base = graph["base"].cpu().numpy()[faces]
            original_area = np.linalg.norm(
                np.cross(base[:, 1] - base[:, 0], base[:, 2] - base[:, 0]), axis=1
            )
            # Linearize about the CURRENT normal, so a folded sheet may rotate.
            # Exact area-path and CCD checks still certify the nonlinear update.
            gradients = np.cross(np.roll(v, -1, axis=1) - np.roll(v, -2, axis=1), normal[:, None])
            scale = np.maximum(np.linalg.norm(gradients, axis=2).max(axis=1), 1e-30)
            gradients /= scale[:, None, None]
            lo_area = (np.maximum(0.1 * original_area, 0.5 * area) - area) * 1000 / scale
            r = row + np.arange(len(faces))
            for corner in range(3):
                vertices = faces[:, corner]
                index = maps[name][vertices]
                valid = index >= 0
                for d in range(3):
                    row_ids.extend(r[valid])
                    columns.extend(index[valid] + d)
                    values.extend(gradients[valid, corner, d])
                lo_area[~valid] -= (
                    gradients[~valid, corner] * proposals[name][vertices[~valid]]
                ).sum(1) * 1000
            lower.extend(lo_area)
            row += len(faces)
            area_rows += len(faces)
        upper = [np.inf] * row
        for name in names:
            graph = self.graphs[name]
            ids = np.flatnonzero(graph["observed"] & (maps[name] >= 0))
            normal = graph["observation_normal"][ids]
            displacement = meshes[name].vertices[ids] - graph["base"].cpu().numpy()[ids]
            offset_normal = (normal * displacement).sum(1)
            r = row + np.arange(len(ids))
            row += len(ids)
            for d in range(3):
                row_ids.extend(r)
                columns.extend(maps[name][ids] + d)
                values.extend(normal[:, d])
            interval = graph["observation_interval"][ids]
            lower.extend((interval[:, 0] - offset_normal) * 1000)
            upper.extend((interval[:, 1] - offset_normal) * 1000)
        # Use the same fixed gauge as the actual parameterization. Projecting
        # an unconstrained joint increment afterwards can invalidate its contacts.
        axis0 = np.asarray(self.motion["axis"], dtype=float)
        axis0 /= np.linalg.norm(axis0)
        for gauge_start in (0, 3):
            row_ids.extend([row] * 3)
            columns.extend(np.arange(gauge_start, gauge_start + 3))
            values.extend(axis0)
            lower.append(0.0)
            upper.append(0.0)
            row += 1
        # Both joint variables live in the fixed two-dimensional gauge plane.
        # Inscribed disks constrain the ACTUAL accumulated parameters inside
        # the coupled problem; clipping a joint afterwards breaks its contacts.
        # Include the current radial direction as a polygon vertex, so even a
        # state exactly on the spherical boundary retains a feasible zero step.
        tangent = np.eye(3)[np.argmin(np.abs(axis0))]
        tangent -= axis0 * (tangent @ axis0)
        tangent /= np.linalg.norm(tangent)
        basis = np.stack([tangent, np.cross(axis0, tangent)])
        fitted_axis = np.asarray(motion["axis"])
        joint_states = (
            (
                0,
                fitted_axis / (fitted_axis @ axis0) - axis0,
                np.tan(np.radians(self.config.max_axis_change_deg)),
            ),
            (
                3,
                np.asarray(motion["origin_m"]) - np.asarray(self.motion["origin_m"]),
                self.config.max_origin_displacement_m,
            ),
        )
        sides = 256
        for joint_start, accumulated, radius in joint_states:
            planar = basis @ accumulated
            angles = np.arctan2(planar[1], planar[0]) + (np.arange(sides) + 0.5) * (
                2 * np.pi / sides
            )
            directions = np.column_stack([np.cos(angles), np.sin(angles)]) @ basis
            r = row + np.arange(sides)
            for dimension in range(3):
                row_ids.extend(r)
                columns.extend(np.full(sides, joint_start + dimension))
                values.extend(directions[:, dimension])
            lower.extend(np.full(sides, -np.inf))
            upper.extend((radius * np.cos(np.pi / sides) - directions @ accumulated) * 1000)
            row += sides
        matrix = sparse.coo_matrix(
            (values, (row_ids, columns)), shape=(row, physical_and_slack)
        ).tocsc()
        # Phase I reduces every local deficit, rather than only the worst one.
        # Phase II projects the fit proposal into the feasible half-spaces.
        # A shared worst-deficit upper bound prevents trading one deep crossing
        # for many smaller improvements. No artificial contraction is prescribed.
        upper_bounds, lower_bounds = np.asarray(upper), np.asarray(lower)
        bounds = np.full(count, 0.5)
        bounds[:3] = 2.0
        for name, ids in maps.items():
            active = ids >= 0
            bounds[ids[active, None] + np.arange(3)] = (
                np.maximum(
                    self.local_radii[name][active, None],
                    np.abs(observation_corrections[name][active]) + 1e-9,
                )
                * 1000
            )
        slack_upper = np.full(pair_count, violation * 1000 + 0.002)
        protected = (current_gaps >= self.config.minimum_clearance_m) | np.asarray(
            [key in self.hard_safe_keys for key in self.rows], dtype=bool
        )
        slack_upper[protected] = 0.0
        lo = np.r_[-bounds, np.zeros(pair_count)]
        hi = np.r_[bounds, slack_upper]

        def motion_problem(upper: Any, desired: Any) -> tuple[Any, ...]:
            selector = sparse.hstack(
                [sparse.eye(count, format="csc"), sparse.csc_matrix((count, pair_count))],
                format="csc",
            )
            identity = sparse.eye(count, format="csc")
            A = sparse.vstack(
                [
                    sparse.hstack([matrix, sparse.csc_matrix((row, count))]),
                    sparse.hstack([selector, -identity]),
                    sparse.hstack([-selector, -identity]),
                ],
                format="csc",
            )
            A.sort_indices()
            c = np.r_[np.zeros(physical_and_slack), np.ones(count)]
            c[physical_and_slack : physical_and_slack + 3] = 0.04
            return (
                A,
                c,
                np.r_[lo, np.zeros(count)],
                np.r_[upper, np.full(count, np.inf)],
                np.r_[lower_bounds, np.full(2 * count, -np.inf)],
                np.r_[upper_bounds, desired, -desired],
            )

        restoring = bool(np.any(current_gaps < self.config.minimum_clearance_m))
        if restoring:
            # Direct signed increments avoid doubling the physical unknowns.
            # An interior solution handles the many nearly parallel contact
            # planes without thousands of degenerate simplex pivots.
            local_maximum = sparse.hstack(
                [
                    sparse.csc_matrix((pair_count, count)),
                    -sparse.eye(pair_count),
                    np.ones((pair_count, 1)),
                ]
            )
            ranged = sparse.vstack(
                [sparse.hstack([matrix, sparse.csc_matrix((row, 1))]), local_maximum],
                format="csc",
            )
            cost = np.r_[np.zeros(physical_and_slack), 1.0]
            col_lower = np.r_[lo, 0.0]
            col_upper = np.r_[hi, violation * 1000 + 0.002]
            row_lower = np.r_[lower_bounds, np.zeros(pair_count)]
            row_upper = np.r_[upper_bounds, np.full(pair_count, np.inf)]
        else:
            if quadratic:
                # Euclidean projection in physical displacement units preserves
                # a preconditioned gradient's descent direction. An L1 tie-break
                # can instead choose an ascending corner of the feasible set.
                metric = np.ones(physical_and_slack)
                lever = max(
                    np.linalg.norm(m.vertices - motion["origin_m"], axis=1).max()
                    for m in meshes.values()
                )
                metric[:3] = max(lever, 0.05) ** 2
                if quadratic_metric is not None:
                    metric[:6] = np.concatenate(quadratic_metric[:2])
                    for name, diagonal in zip(names, quadratic_metric[2:], strict=True):
                        active = maps[name] >= 0
                        index = maps[name][active, None] + np.arange(3)
                        metric[index.ravel()] = np.asarray(diagonal)[active].ravel()
                    if not np.isfinite(metric).all() or np.any(metric <= 0):
                        raise ValueError("projection metric must be finite and positive")
                    metric /= np.median(metric[:count])
                # Safe-pair slacks are fixed at zero. Eliminate them instead of
                # feeding fixed columns into a degenerate active-set Hessian.
                x, audit = _quadratic_projection(
                    matrix[:, :count],
                    metric[:count],
                    target,
                    lo[:count],
                    hi[:count],
                    lower_bounds,
                    upper_bounds,
                    require_feasible=not allow_inexact_proposal,
                )
                audit.update(
                    boundary_expanded_nodes=boundary_expansion,
                    area_constraints=area_rows,
                    constraints=row,
                    variables=count,
                    self_contact_planes=len(self.self_planes),
                    restoration_pairs=0,
                    prior_violation_m=violation,
                )
                if x is None:
                    return None, audit
                delta = [x[:3] / 1000, x[3:6] / 1000]
                for name in names:
                    vertex_delta = proposals[name].copy()
                    active = maps[name] >= 0
                    vertex_delta[active] = x[maps[name][active, None] + np.arange(3)] / 1000
                    delta.append(vertex_delta)
                return delta, audit
            else:
                ranged, cost, col_lower, col_upper, row_lower, row_upper = motion_problem(
                    hi, target
                )

        def pass_problem(A: Any, c: Any, lower: Any, upper: Any, lhs: Any, rhs: Any) -> None:
            lp = highspy.HighsLp()
            lp.num_col_, lp.num_row_ = A.shape[1], A.shape[0]
            lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, lower, upper
            lp.row_lower_, lp.row_upper_ = lhs, rhs
            lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = (
                A.indptr,
                A.indices,
                A.data,
            )
            solver.passModel(lp)

        solver = highspy.Highs()
        for name, value in dict(
            output_flag=False,
            solver="ipm",
            run_crossover="off",
            presolve="on",
            # HiGHS owns a process-wide scheduler. A stable budget avoids
            # rejecting later solves when callers use different query pools.
            threads=1,
            primal_feasibility_tolerance=1e-8,
            dual_feasibility_tolerance=1e-8,
        ).items():
            solver.setOptionValue(name, value)
        pass_problem(ranged, cost, col_lower, col_upper, row_lower, row_upper)
        started = monotonic()
        logging.getLogger(__name__).info(
            "Contact projection: %s variables, %s inequalities, %.3f mm worst deficit, "
            "%s local restoration slacks",
            count,
            row,
            violation * 1000,
            pair_count,
        )
        solver.setOptionValue("time_limit", 120.0)
        presolve_retries = 0

        def run_problem() -> Any:
            nonlocal presolve_retries
            solver.run()
            result = solver.getModelStatus()
            if result == highspy.HighsModelStatus.kUnknown and not presolve_retries:
                # Presolve can produce an ill-conditioned reduced problem for
                # nearly coplanar contact systems. Retry the original matrix;
                # one nanometre LP tolerances remain below the geometric guard.
                solver.clearSolver()
                for option, value in dict(
                    presolve="off",
                    primal_feasibility_tolerance=1e-6,
                    dual_feasibility_tolerance=1e-6,
                    ipm_optimality_tolerance=1e-6,
                ).items():
                    solver.setOptionValue(option, value)
                pass_problem(ranged, cost, col_lower, col_upper, row_lower, row_upper)
                presolve_retries += 1
                solver.run()
                result = solver.getModelStatus()
            return result

        primary_status = run_problem()
        stage_status = primary_status
        primary = solver.getSolution()
        primary_iterations = solver.getInfo().ipm_iteration_count
        target_slack = None
        if (
            restoring
            and primary_status == highspy.HighsModelStatus.kOptimal
            and primary.value_valid
        ):
            primary_values = np.asarray(primary.col_value)
            primary_residual = max(
                np.max(row_lower - ranged @ primary_values, initial=0),
                np.max(ranged @ primary_values - row_upper, initial=0),
                np.max(col_lower - primary_values),
                np.max(primary_values - col_upper),
            )
            if np.isfinite(primary_values).all() and primary_residual <= 1e-6:
                # Lexicographic Phase II cannot buy area-weighted improvement
                # by leaving the hardest crossing at its old deficit ceiling.
                target_slack = max(0.0, float(primary_values[-1])) + 1e-6
                # Eliminate the max epigraph after Phase I. Fixed slack bounds
                # express the same hierarchy without nearly singular active rows.
                ranged, row_lower, row_upper = matrix, lower_bounds, upper_bounds
                cost = np.r_[np.zeros(count), self.area_weights(meshes)]
                col_lower, col_upper = lo.copy(), hi.copy()
                col_upper[count:] = np.minimum(col_upper[count:], target_slack)
                pass_problem(ranged, cost, col_lower, col_upper, row_lower, row_upper)
                stage_status = run_problem()
        status = stage_status
        solution = solver.getSolution()
        secondary_status = solver.modelStatusToString(status) if target_slack is not None else None
        selected_phase = "local_deficits" if target_slack is not None else "primary"
        allowed_status = (
            highspy.HighsModelStatus.kOptimal,
            highspy.HighsModelStatus.kTimeLimit,
            highspy.HighsModelStatus.kIterationLimit,
            highspy.HighsModelStatus.kUnknown,
        )
        if target_slack is not None:
            secondary_values = np.asarray(solution.col_value)
            secondary_residual = (
                max(
                    np.max(row_lower - ranged @ secondary_values, initial=0),
                    np.max(ranged @ secondary_values - row_upper, initial=0),
                    np.max(col_lower - secondary_values),
                    np.max(secondary_values - col_upper),
                )
                if solution.value_valid and np.isfinite(secondary_values).all()
                else np.inf
            )
            if status not in allowed_status or secondary_residual > 1e-6:
                # The checked primary optimum is still a valid restoration step;
                # a failed secondary refinement must not discard that progress.
                status, solution = primary_status, primary
                selected_phase = "worst_gap"
        motion_status = None
        iterations = primary_iterations + (
            solver.getInfo().ipm_iteration_count if target_slack is not None else 0
        )
        if restoring and status in allowed_status and solution.value_valid:
            attained = np.asarray(solution.col_value)[:physical_and_slack]
            feasible = (
                np.isfinite(attained).all()
                and max(
                    np.max(lower_bounds - matrix @ attained, initial=0),
                    np.max(matrix @ attained - upper_bounds, initial=0),
                    np.max(lo - attained),
                    np.max(attained - hi),
                )
                <= 1e-6
            )
            if feasible:
                # A feasibility LP has a large null space. Arbitrary vertex
                # motion in that space folds neighbouring faces and forces CCD
                # to undo otherwise useful progress. Freeze every achieved
                # contact deficit before minimizing movement; motion can never
                # purchase a worse separation result.
                saved = (
                    ranged,
                    cost,
                    col_lower,
                    col_upper,
                    row_lower,
                    row_upper,
                    status,
                    solution,
                    selected_phase,
                )
                attained_upper = hi.copy()
                attained_upper[count:] = np.minimum(
                    attained_upper[count:],
                    np.maximum(0, attained[count:]) + 1e-6,
                )
                ranged, cost, col_lower, col_upper, row_lower, row_upper = motion_problem(
                    attained_upper, np.zeros(count)
                )
                pass_problem(ranged, cost, col_lower, col_upper, row_lower, row_upper)
                # This optional tie-break must not consume the entire remaining
                # restoration budget repeatedly on a degenerate contact system.
                # Keep the verified separation optimum if ten further seconds
                # cannot produce a feasible minimum-motion proposal. HiGHS'
                # runtime is cumulative across the three objectives.
                solver.setOptionValue("time_limit", min(120.0, solver.getRunTime() + 10.0))
                trial_status, trial = run_problem(), solver.getSolution()
                motion_status = solver.modelStatusToString(trial_status)
                iterations += solver.getInfo().ipm_iteration_count
                trial_values = np.asarray(trial.col_value)
                residual = (
                    max(
                        np.max(row_lower - ranged @ trial_values, initial=0),
                        np.max(ranged @ trial_values - row_upper, initial=0),
                        np.max(col_lower - trial_values),
                        np.max(trial_values - col_upper),
                    )
                    if trial.value_valid and np.isfinite(trial_values).all()
                    else np.inf
                )
                if trial_status in allowed_status and residual <= 1e-6:
                    status, solution, selected_phase = trial_status, trial, "minimum_motion"
                else:
                    (
                        ranged,
                        cost,
                        col_lower,
                        col_upper,
                        row_lower,
                        row_upper,
                        status,
                        solution,
                        selected_phase,
                    ) = saved
        audit = dict(
            status=solver.modelStatusToString(status),
            boundary_expanded_nodes=boundary_expansion,
            secondary_status=secondary_status,
            minimum_motion_status=motion_status,
            selected_phase=selected_phase,
            optimality_proven=status == highspy.HighsModelStatus.kOptimal,
            presolve_retries=presolve_retries,
            area_constraints=area_rows,
            worst_gap_phase_status=solver.modelStatusToString(primary_status),
            worst_gap_target_m=target_slack / 1000 if target_slack is not None else None,
            constraints=row,
            prior_violation_m=violation,
            iterations=iterations,
            variables=count,
            solve_seconds=monotonic() - started,
            self_contact_planes=len(self.self_planes),
            restoration_pairs=int(np.count_nonzero(current_gaps < self.config.minimum_clearance_m)),
        )
        if status not in allowed_status or not solution.value_valid:
            return None, audit
        solution_values = np.asarray(solution.col_value)[: ranged.shape[1]]
        x = solution_values[:physical_and_slack]
        actual = matrix @ x
        residual = max(
            np.max(row_lower - ranged @ solution_values, initial=0),
            np.max(ranged @ solution_values - row_upper, initial=0),
            np.max(col_lower - solution_values),
            np.max(solution_values - col_upper),
            np.max(lower_bounds - actual, initial=0),
            np.max(actual - upper_bounds, initial=0),
            np.max(lo - x),
            np.max(x - hi),
        )
        audit["maximum_linear_constraint_violation_m"] = float(max(0, residual) / 1000)
        if residual > 1e-6 or not np.isfinite(x).all():
            return None, audit
        delta = [x[:3] / 1000, x[3:6] / 1000]
        for name in names:
            value = proposals[name].copy()
            active = maps[name] >= 0
            value[active] = x[maps[name][active, None] + np.arange(3)] / 1000
            delta.append(value)
        audit["restoration_slack_m"] = float(np.max(x[count:], initial=0) / 1000)
        audit["area_weighted_restoration_sum_m"] = float(
            self.area_weights(meshes) @ x[count:] / 1000
        )
        return delta, audit
