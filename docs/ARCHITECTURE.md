# PhysReal-X architecture

## Platform responsibility

PhysReal-X is the control plane and data plane that connects digital-twin assets, simulator
workers, trajectory generation, multimodal datasets, policy training, evaluation, and real-robot
feedback. It is not another physics engine and does not copy Isaac Sim, Genesis, or MuJoCo APIs
into the domain model.

The platform closes three loops:

```text
real capture -> reconstruction/calibration -> versioned trainable asset
asset + task -> seeded simulation -> validated episodes -> immutable dataset
dataset -> training -> held-out/real evaluation -> failure cases -> directed regeneration
```

The current `0.1` release deliberately proves that architecture with one executable drawer
vertical slice. Production integrations remain external adapters until they pass the shared
contracts; the repository does not advertise placeholder Isaac, robot, or reconstruction support.

## Dependency direction

```text
domain <- contracts <- application services <- adapters/interfaces
```

- `domain.py` owns frozen, versioned specifications and manifests. It cannot import storage,
  Torch, a simulator, or a service framework.
- `contracts.py` owns narrow simulator, policy, trainer, repository, and artifact ports.
- `runtime.py` and `platform.py` resolve lineage, enforce task semantics, execute workflows, and
  publish only validated terminal results.
- `reference.py`, `training.py`, and `storage.py` are replaceable adapters.
- `cli.py` is a thin interface over the same application facade; a future HTTP interface must be
  equally thin.

This implements dependency inversion without creating interfaces that have no consumer. A new
adapter is introduced only when an actual method is ready to integrate.

## Core execution contract

1. Register an `AssetSpec` and `TaskSpec`; bytes are canonicalized and addressed by SHA-256.
2. Compile a `RunSpec` into resolved asset/task digests plus exact backend/policy versions.
3. Create one expensive simulator session and policy per run, then reset both for each seed.
4. Validate initial, success, and termination predicates centrally from declared observations.
5. Validate every action and frame against dtype, shape, unit, frame, bounds, and timestamps.
6. Atomically publish an Episode artifact and summary for every completed rollout.
7. Publish a Dataset manifest only after every requested Episode exists and passes validation.
8. Stream Episodes to a trainer; verify returned weights and lineage before publishing a model.
9. Evaluate on deterministic held-out seeds and persist categorized `FailureCase` records.

SQLite transactions protect metadata state transitions. The local artifact store writes a temporary
object, syncs it, atomically renames it, and verifies its content digest before returning. These
ports can later be implemented by PostgreSQL and S3-compatible storage without changing domain
contracts.

## PPT capability mapping and integration order

The research methods in the project presentation should enter in this order:

### 1. Asset intelligence worker

Input: calibrated RGB/RGB-D/event/force streams and robot/object state.

Initial adapters:

- calibration, synchronization, depth completion, and quality/confidence estimation;
- category-level pose and shape reconstruction;
- VGGT/mesh/3DGS reconstruction behind one worker boundary;
- part decomposition and articulated joint estimation such as PD²GS-style methods;
- mass, friction, stiffness, damping, and uncertainty calibration.

Output: geometry artifacts plus one validated `AssetSpec`. Mesh/USD/Gaussian bytes live in the
artifact store; manifests carry checksums, link/joint structure, physics provenance, confidence,
and semantic version. Research-native outputs are converted at the adapter boundary rather than
leaking model-specific tensors into the platform API.

### 2. Simulation workers

- Isaac Sim/Isaac Lab is the primary high-fidelity USD, RTX, articulation, and RL worker.
- Genesis is added for throughput or deformable-material workloads after it passes the same
  backend contract suite.
- MuJoCo is useful for lightweight control baselines, not as the visual asset source of truth.

Each simulator runs in its own worker environment. Isaac Kit's Python/Torch runtime must not be
installed into the core `PhysReal-X` conda environment. The worker reports capabilities before a
run, fails fast on unsupported observations/actions, and returns standard `DataValue` frames.

### 3. Generation and orchestration

Add concrete generators incrementally: task templates, scene/layout randomization, motion-planner
or teleoperation demonstrations, learned VLA/diffusion policies, and targeted long-tail/failure
generation. Every generator is a versioned Policy adapter and therefore produces identical Episode
semantics regardless of simulator.

RGB, depth, masks, point clouds, force, contacts, joint state, action, reward, and events share the
same timeline. Large arrays are immutable `ArtifactRef` objects; small numeric state is inlined.
Dataset manifests retain asset/task/backend/policy digests and every episode seed.

### 4. Learning and evaluation

The reference Torch behavior-cloning adapter proves Dataset-to-Policy wiring on cu121. Next
production adapters should be added in evidence order: scalable imitation learning, Isaac Lab RL,
VLA fine-tuning, then action-conditioned world/dynamics models. Trainers consume lazy episode
iterables and own batching/sharding; the application layer must never materialize a large dataset
or GPU tensor set for them.

Evaluation always produces a versioned report, aggregate metrics, and episode-level failures
(perception, planning, control, contact, timeout, data quality, or unknown). A ROS 2 deployment
adapter can record real Episodes using the same contracts. Real failures then select missing asset,
physics, scene, or task regions for regeneration.

## Service evolution

The Python facade is the authoritative application API. Once multiple machines or users require
remote submission, add a thin FastAPI/gRPC control interface and durable worker queue around it:

```text
SDK/CLI/HTTP -> application facade -> job repository
                                      |
                              capability-matched worker
                                      |
                         artifact store + manifest commit
```

Remote services must preserve idempotency keys, immutable resource versions, atomic manifest
publication, cancellation, retry state, worker heartbeats, and failure isolation. Kubernetes,
Kafka, PostgreSQL, and object storage are deployment choices, not domain dependencies, and should
not be introduced before scale measurements justify them.

## Adapter acceptance gates

An adapter is supported only when it demonstrates:

- API and plugin identity compatibility, capability discovery, and fail-fast preflight;
- `create -> reset -> step -> observe -> close` lifecycle and idempotent cleanup;
- deterministic reset for a fixed seed where the backend claims determinism;
- exact dtype/shape/unit/frame/device semantics and monotonic per-signal timestamps;
- checksum and version lineage for every input and output artifact;
- cancellation/error cleanup without publishing a partial Dataset manifest;
- shared contract tests plus an isolated integration test in its real runtime.

This keeps the core small while allowing the presentation's reconstruction, simulation, world
model, VLA/RL, and real-robot results to become callable platform capabilities one at a time.
