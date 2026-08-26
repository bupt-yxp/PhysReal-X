# PhysReal-X

[中文文档](https://bupt-yxp.github.io/PhysReal-X/) ·
[平台架构](https://bupt-yxp.github.io/PhysReal-X/ARCHITECTURE/)

PhysReal-X is a contract-first platform for reproducible embodied-AI data generation and
robot-learning experiments. It is an orchestration and data system above simulators—not a new
physics engine.

The detailed platform boundaries and research-method integration order are documented in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

The first release intentionally implements one complete vertical slice:

```text
versioned asset + task
        ↓
resolved, seeded run
        ↓
deterministic reference drawer simulation
        ↓
validated Episode artifacts and Dataset manifest
        ↓
PyTorch behavior cloning (CUDA 12.1 when requested)
        ↓
held-out evaluation and structured failures
```

Isaac Sim, Genesis, robot hardware, and research models join through narrow adapters. They are not
core dependencies and importing `physrealx` never initializes Torch, CUDA, a simulator, the network,
or a workspace.

## Design principles

- **One public name:** distribution `physreal-x`, import and CLI `physrealx`.
- **Immutable contracts:** Pydantic v2 models reject unknown fields and non-finite values.
- **Exact lineage:** every run resolves asset, task, backend, policy, version, digest, and seed.
- **Atomic publication:** artifacts are SHA-256 addressed; manifests publish only after validation.
- **Narrow plugins:** simulation, policy, and training adapters expose separate protocols.
- **No compatibility debt:** this repository began empty; there are no legacy aliases or v0 APIs.
- **Local-first, replaceable infrastructure:** SQLite/WAL and a local artifact store provide the same
  boundaries that future PostgreSQL/S3 adapters can implement.

## Installation

Python 3.11 is the recommended development runtime; the package and CPU CI also support Python
3.12. GPU development uses PyTorch 2.5.1 compiled for CUDA 12.1.
The PyTorch wheel contains its CUDA user-space runtime; the NVIDIA driver only needs to be new
enough to run it.

```bash
conda env create -f environment.yml
conda activate PhysReal-X
physrealx doctor
```

On this workstation CUDA 12.1 compilation tools live at `/usr/local/cuda-12.1`. Do not change the
global CUDA symlink. Only custom extension builds should use command-local settings:

```bash
CUDA_HOME=/usr/local/cuda-12.1 \
PATH=/usr/local/cuda-12.1/bin:$PATH \
python -m build
```

Core-only development does not require Torch:

```bash
python -m pip install -e '.[dev]'
```

## Quick start

Create a local workspace and run the deterministic end-to-end demonstration:

```bash
physrealx init .physrealx
physrealx demo --workspace .physrealx --episodes 8
```

Inspect the environment, plugins, resources, and a persisted run:

```bash
physrealx doctor --json
physrealx plugins
physrealx list asset --workspace .physrealx
physrealx status RUN_ID --workspace .physrealx
```

All CLI commands return stable non-zero exit codes for expected operational errors and suppress
tracebacks unless `--debug` is explicitly requested.

The same vertical slice is available as a typed Python API:

```python
from pathlib import Path

from physrealx import PhysRealX, TrainingSpec
from physrealx.reference import reference_asset, reference_run, reference_task

platform = PhysRealX.open(Path(".physrealx"), discover_plugins=False)
asset = reference_asset()
task = reference_task(asset)
platform.register_asset(asset)
platform.register_task(task)

run = platform.execute(reference_run(asset, task, episode_count=8, base_seed=7))
assert run.result is not None and run.result.dataset is not None
model = platform.train(
    TrainingSpec(
        resource_id="drawer.bc",
        version="1.0.0",
        dataset=run.result.dataset,
        trainer="torch_bc",
        config={"device": "cuda", "epochs": 120, "seed": 17},
    )
)
```

`PhysRealX.open` owns the application composition root. Production simulator, policy, and trainer
packages register narrow adapters through the `physrealx.plugins` entry-point group; core domain
models remain independent of those packages.

## Package boundaries

```text
domain.py       frozen specifications and manifests
contracts.py    simulation, policy, trainer, artifact ports
plugins.py      explicit registry and entry-point discovery
storage.py      atomic local artifacts and transactional SQLite metadata
runtime.py      run compilation, execution, Episode validation, datasets
reference.py    deterministic drawer backend and scripted policy
training.py     optional, lazy Torch behavior-cloning adapter
platform.py     application facade
cli.py          thin command-line interface
```

Execution depends inward on contracts. Domain code never imports storage, Torch, CUDA, CLI, or a
simulator. A backend session and policy are created once per run and reset for each seeded episode,
so expensive scenes and GPU models are not reconstructed for every rollout. Trainers receive a
lazy episode iterable; production adapters can batch and stream rather than loading the entire
dataset in the application facade.

## Data semantics

An Episode step follows RLDS-style semantics: it contains the current observation and the action
taken from that observation. The final step contains the final observation and has no action or
reward. Control timestamps are monotonic; each sensor value may also carry its own timestamp.

High-bandwidth observations are represented by immutable `ArtifactRef` values instead of embedding
images or point clouds in JSON. Numeric state may be inlined when its declared dtype and shape match.

## Quality gates

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest -m 'not gpu' --cov=physrealx --cov-branch
python -m build
```

The CUDA smoke test is deliberately separate because it requires the `PhysReal-X` cu121 environment:

```bash
python -m pytest -m gpu tests/integration/test_cuda.py
```

It performs only a 16×16 matrix multiplication, which is safe on a busy development GPU.

## Backend roadmap

1. Keep the reference backend as the executable contract suite.
2. Add an Isaac Sim/Isaac Lab worker for USD, RTX rendering, and articulated manipulation.
3. Add Genesis only where high-throughput or deformable simulation is required.
4. Add ROS 2 capture/deployment without changing the Episode contract.

An adapter is considered supported only after it passes the shared lifecycle, determinism, schema,
capability, and failure-isolation contract tests.

## License

MIT. See [LICENSE](LICENSE).
