# 安装与环境

PhysReal-X 核心支持 Python 3.8–3.12，导入时不会初始化 CUDA、启动仿真器或创建工作区。建议用
Python 3.11 运行平台核心，并让不同算法通过独立环境或容器接入。

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
    ├── MuJoCo / Genesis / Gazebo 等仿真 worker
    ├── COLMAP / OMNI-DC / Ditto / CoACD 等资产 worker
    ├── LeRobot / Diffusion Policy / DreamerV3 等训练 worker
    └── ROS 2 真机 worker
```

每个外部方法独立登记 Python/CUDA、镜像摘要、源码 revision、输入输出契约和许可证，并通过
`physrealx.workers` 包入口发现。生产容器必须预先安装 digest 固定的镜像；平台不会自动拉取镜像或
下载权重。不要为了接入某个算法而放宽核心包的 Python 或 NumPy 约束。

## 验证安装

```bash
python -c 'import physrealx; print(physrealx.__version__)'
physrealx doctor --strict
physrealx demo --workspace .physrealx-demo --episodes 4
```

开发者还应执行：

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest -m 'not gpu'
python -m mkdocs build --strict
```
