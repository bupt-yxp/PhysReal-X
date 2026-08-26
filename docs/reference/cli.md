# 命令行参考

命令行入口为 `physrealx`。所有业务命令都调用同一个 `PhysRealX` 应用门面。

## 全局选项

```bash
physrealx [--debug] COMMAND
```

默认情况下，预期错误以稳定错误码输出且不显示回溯。只有调试问题时才使用 `--debug`。

## 命令总览

| 命令 | 用途 |
| --- | --- |
| `init` | 初始化本地工作区 |
| `doctor` | 检查 Python、Conda、Torch 与 CUDA |
| `plugins` | 列出已注册插件 |
| `validate` | 校验资产、任务、运行或训练规格 |
| `add` | 注册不可变资产或任务 |
| `list` | 列出版本化资源 |
| `run` | 执行运行规格 |
| `status` | 查询持久化运行状态 |
| `train` | 从数据集训练模型 |
| `evaluate` | 使用独立种子评测模型 |
| `demo` | 运行内置抽屉参考闭环 |

## 工作区

除 `doctor` 和 `validate` 外，资源命令使用 `--workspace` 指定工作区；默认值为当前目录下的 `.physrealx`。

```bash
physrealx init .physrealx
physrealx list dataset --workspace .physrealx
```

## 环境检查

```bash
physrealx doctor [--json] [--no-cuda] [--strict]
```

- `--json`：输出结构化结果；
- `--no-cuda`：跳过 CUDA 探测；
- `--strict`：存在诊断问题时返回非零退出码。

## 规格校验与注册

```bash
physrealx validate {asset,task,run,training} PATH
physrealx add {asset,task} PATH --workspace WORKSPACE
```

## 资源查询

```bash
physrealx list {asset,task,dataset,model,evaluation} --workspace WORKSPACE
physrealx status RUN_ID --workspace WORKSPACE
```

## 运行与训练

```bash
physrealx run RUN_FILE --workspace WORKSPACE
physrealx train TRAINING_FILE --workspace WORKSPACE
```

## 模型评测

```bash
physrealx evaluate MODEL_ID MODEL_VERSION RUN_FILE \
  --id EVALUATION_ID \
  --version EVALUATION_VERSION \
  [--episodes COUNT] \
  [--base-seed SEED] \
  [--workspace WORKSPACE]
```

## 参考闭环

```bash
physrealx demo \
  [--episodes COUNT] \
  [--base-seed SEED] \
  [--train] \
  [--device {auto,cpu,cuda}] \
  [--workspace WORKSPACE]
```

各命令的实时参数说明可以通过 `physrealx COMMAND --help` 查看。
