# 快速开始

本页使用内置抽屉任务验证平台从仿真到评测的完整连接。参考后端是确定性小型动力学模型，用于执行契约测试，不用于替代生产物理引擎。

## 初始化工作区

```bash
physrealx init .physrealx
```

工作区包含两类持久数据：

- `registry.sqlite3`：资源版本、运行状态和 Episode 摘要；
- `artifacts/`：按 SHA-256 内容寻址的规格、轨迹、数据集、权重和报告。

## 生成参考数据集

```bash
physrealx demo --workspace .physrealx --episodes 8
```

该命令会注册抽屉资产与开抽屉任务，使用固定种子生成 8 条轨迹，并在全部轨迹通过校验后发布数据集。

## 完成 CUDA 训练与评测

```bash
physrealx demo \
  --workspace .physrealx \
  --episodes 8 \
  --train \
  --device cuda
```

完整过程如下：

```text
抽屉资产与任务
      ↓
脚本策略生成 8 条 Episode
      ↓
发布不可变数据集
      ↓
CUDA 行为克隆训练
      ↓
新种子仿真评测
      ↓
评测报告与失败案例
```

没有可用 GPU 时，将设备改为 `cpu`；使用 `auto` 时，训练器会优先选择符合要求的 CUDA 环境。

## 检查结果

```bash
physrealx plugins --workspace .physrealx
physrealx list asset --workspace .physrealx
physrealx list task --workspace .physrealx
physrealx list dataset --workspace .physrealx
physrealx list model --workspace .physrealx
physrealx list evaluation --workspace .physrealx
```

如需检查某次运行，将命令输出中的运行标识传给：

```bash
physrealx status RUN_ID --workspace .physrealx
```

下一步可以阅读[资产与任务](../user-guide/assets-and-tasks.md)，了解如何定义自己的实验输入。
