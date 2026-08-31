# 命令行

命令行用于本地工作区初始化、运行时诊断、领域规格校验和可复现演示。数字孪生构建的阶段配方使用 Python API，不维护第二套 CLI 语义。

## 命令树

```text
physrealx [--debug] COMMAND
├── init [WORKSPACE]
├── doctor [--json] [--no-cuda] [--strict]
├── methods [--workspace PATH]
├── validate {asset,scene,task,run,training} FILE
├── add {asset,scene,task} FILE [--workspace PATH]
├── list RESOURCE_TYPE [--workspace PATH]
├── run FILE [--workspace PATH]
├── status RUN_ID [--workspace PATH]
├── train FILE [--workspace PATH]
├── evaluate MODEL_ID MODEL_VERSION RUN_FILE --id ID --version VERSION
└── demo [--episodes N] [--base-seed N] [--train] [--device DEVICE]
```

`RESOURCE_TYPE` 包含 `observation_set`、`asset`、`scene`、`task`、`dataset`、`model`、
`evaluation`、`operation` 和 `recipe_execution`。

## 工作区与诊断

```bash
physrealx init .physrealx
physrealx doctor --strict
physrealx methods --workspace .physrealx
```

`doctor --strict` 在发现 Python、Torch 或 CUDA 环境问题时返回非零退出码；`--no-cuda` 用于不应初始化 CUDA 的 CI。
`methods` 逐项输出 `execution_kind`、`executable`、`ready` 和 `detail`；只有描述符或本机缺少可执行文件/镜像时 `ready=false`。

## 规格校验与发布

```bash
physrealx validate asset assets/drawer.asset.yaml
physrealx add asset assets/drawer.asset.yaml --workspace .physrealx
physrealx list asset --workspace .physrealx
```

`validate` 只解析和校验文件；`add` 将规范化清单原子写入制品存储，并发布带制品标识的不可变引用。

## 运行、训练与评测

```bash
physrealx run runs/generate.run.yaml --workspace .physrealx
physrealx status RUN_ID --workspace .physrealx
physrealx train training/bc.training.yaml --workspace .physrealx
physrealx evaluate drawer.bc 1.0.0 runs/evaluate.run.yaml \
  --id drawer.heldout --version 1.0.0 --workspace .physrealx
```

命令成功时输出结构化 JSON。预期的 `PhysRealXError`、Pydantic 校验错误和文件错误默认仅输出稳定错误码与消息；使用 `--debug` 时保留回溯。

## 可执行契约演示

```bash
physrealx demo --workspace .physrealx-demo --episodes 8 --train --device cpu
```

演示使用内置抽屉资产、确定性参考后端和脚本策略，验证资产→场景→任务→运行→数据集，
可选继续验证参考训练和评测契约。它不执行外部物理仿真器或真机。
