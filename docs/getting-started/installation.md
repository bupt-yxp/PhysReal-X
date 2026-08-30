# 安装与环境

PhysReal-X 核心支持 Python 3.8–3.12，导入时不会初始化 CUDA、启动仿真器或创建工作区。建议用
Python 3.11 运行平台核心，并让不同算法通过独立环境或容器接入。
本地元数据库使用 SQLite `STRICT` 表，因此系统 SQLite 必须为 3.37.0 或更高版本：

```bash
python -c 'import sqlite3; print(sqlite3.sqlite_version)'
```

## 从源码安装

```bash
git clone https://github.com/bupt-yxp/PhysReal-X.git
cd PhysReal-X
python -m pip install -e .
```

开发与文档环境：

```bash
python -m pip install -e '.[dev,docs]'
```

训练参考实现还需要 `train` 扩展及匹配的 PyTorch：

```bash
python -m pip install -e '.[train]'
```

项目的完整 CUDA 12.1 环境可直接创建：

```bash
conda env create -f environment.yml
conda activate PhysReal-X
```

Python 3.9–3.12 的训练扩展固定 NumPy 1.26.4；NumPy 1.26 不支持 Python 3.8，因此 Python 3.8
使用同属 NumPy 1.x 的 1.24.4。

## 初始化工作区

```bash
physrealx init .physrealx
physrealx doctor
physrealx methods --workspace .physrealx
```

工作区包含本地元数据数据库和内容寻址制品目录。方法列表同时显示能力、源码版本、许可证、执行
类型、是否已安装以及本机就绪状态；它不会把仅有描述符的方法标为可运行。

## 环境分层

```text
Python 3.8–3.12 核心
    ├── 领域模型、工作区、制品与调度语义
    ├── 资产 worker：COLMAP / OMNI-DC / Ditto / CoACD / OpenUSD
    ├── 运行与训练 worker：按需安装的仿真、数据和学习适配器
    └── 机器人 worker：按硬件配置独立部署的 ROS 2 节点
```

每个外部方法独立登记 Python/CUDA、环境摘要、源码 revision、输入输出契约和许可证。
本机进程 worker 通过工作区 `workers.d/*.json` 加载，发布的扩展包也可使用
`physrealx.workers` 入口。生产容器必须预安装 digest 固定的镜像；平台不会拉取镜像或权重。
当前外部 worker 协议包支持 Python 3.10–3.12，建议用 Python 3.11 执行安装器；
这不改变核心 SDK 的 Python 3.8–3.12 兼容范围。Ditto 等历史算法仍使用独立
Python 3.8 推理环境和受限桥接进程。

项目附带的无权重资产适配器可独立安装：

```bash
python3.11 workers/install.py \
  --root /data/physrealx-workers/envs \
  --workspace .physrealx \
  --colmap-binary /usr/local/bin/colmap \
  colmap coacd mujoco_sysid openusd
```

`mujoco_sysid` 是安装器键，登记的方法名为
`physrealx_mujoco_sysid_adapter`；它只从版本化的单关节实测证据辨识等效惯量、黏性阻尼和库仑摩擦，不是通用
MuJoCo 仿真后端。OMNI-DC 和 Ditto 还需官方源码、固定提交、权重和专用 CUDA
环境，详见 [OMNI-DC worker](../api/omnidc-worker.md) 和
[Ditto worker](../api/ditto-worker.md)。

仿真、离线渲染、Episode 转换、策略训练和世界模型均通过同一方法协议扩展，
但核心仓库不附带伪造这些能力的 toy worker。调用前必须安装真实适配器，并让
`px.methods.inspect(capability, name).ready` 通过环境和小样本探针。

## 验证安装

```bash
python -c 'import physrealx; print(physrealx.__version__)'
physrealx doctor
physrealx demo --workspace .physrealx-demo --episodes 4
```

只有在按 `environment.yml` 安装完整 CUDA 12.1 环境并期望使用 GPU 时，才应使用
`physrealx doctor --strict`；核心 CPU 安装不应被误判为 CUDA 部署。

开发者还应执行：

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest -m 'not gpu'
python -m mkdocs build --strict
```
