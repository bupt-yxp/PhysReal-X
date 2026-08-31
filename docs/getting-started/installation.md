# 安装与环境

PhysReal-X 核心支持 Python 3.8–3.12；需要运行内嵌 ArtGS 时，统一使用项目固定的
Python 3.10、PyTorch 2.4.1 与 CUDA 12.1 环境。算法训练可由同一解释器启动子进程隔离故障，
但不要求第二个 Conda、venv 或工作区外部源码仓库。

本地元数据库使用 SQLite `STRICT` 表，因此系统 SQLite 必须为 3.37.0 或更高版本：

```bash
python -c 'import sqlite3; print(sqlite3.sqlite_version)'
```

## 推荐：创建统一环境

```bash
git clone https://github.com/bupt-yxp/PhysReal-X.git
cd PhysReal-X
conda env create -f environment.yml
conda activate PhysReal-X
python -m pip install --no-build-isolation -e '.[artgs,dev,docs,demo,train]'
physrealx doctor
```

`environment.yml` 只创建一个 `PhysReal-X` 环境并预装固定的 Python、Torch/CUDA 与构建工具；
紧随其后的根项目 extra 再安装开发、文档、Demo、训练与 ArtGS 依赖。
`--no-build-isolation` 仅防止 PyTorch3D/CUDA 扩展在临时 pip 环境中丢失已安装的 Torch，
不会创建第二个运行环境。ArtGS 主体、Gaussian rasterizer、Simple-KNN、PointNet2 和 GLM 源码均在
`algorithms/artgs/`；来源、提交、许可证与平台修改记录见其中的 `SOURCE.toml`。
其中 Gaussian rasterizer 与 Simple-KNN 限于非商业研究/评估；安装 extra 不会
改变上游许可边界，详见根目录 `THIRD_PARTY_NOTICES.md`。

已有兼容环境时，可直接安装到当前环境：

```bash
python -m pip install --no-build-isolation -e '.[artgs,dev,docs,demo,train]'
```

ArtGS 的 CUDA 扩展在首次执行前从仓库内源码原位编译，并用当前 `sys.executable` 导入验证；
平台不会查找或调用 `/home/.../ArtGS-training` 一类外部 clone。

## 仅安装核心

只使用领域模型、工作区和 CPU 参考切片时，不需要 GPU 依赖：

```bash
python -m pip install -e .
python -m pip install -e '.[dev,docs]'
```

训练 extra 在 Python 3.9–3.12 固定 NumPy 1.26.4；Python 3.8 使用 NumPy 1.24.4。
`artgs` extra 只在 Python 3.10 生效，其他版本中方法会被注册但探针明确返回不可用。

## 方法就绪检查

```bash
physrealx init .physrealx
physrealx doctor
physrealx methods --workspace .physrealx
```

```python
from physrealx import MethodCapability, PhysRealX

px = PhysRealX.open(".physrealx")
status = px.methods.inspect(MethodCapability.GEOMETRY_RECONSTRUCT, "artgs")
print(status.ready, status.detail)
```

`ready=True` 要求嵌入源码完整、Python/CUDA/PyTorch 版本匹配、`nvcc` 可用、依赖可导入且
GPU 可见。只有描述符存在不算就绪；权重、输入证据或显存不足时执行仍会在发布任何产物前失败。

COLMAP、ROS 2 和设备驱动属于系统级原生依赖，可在单独进程或机器人节点运行；这类进程边界用于
设备权限和故障隔离，不应产生另一套不受根项目管理的 Python 包版本。

## 验证安装

```bash
python -c 'import physrealx; print(physrealx.__version__)'
python -m pip check
python -m pytest -m 'not gpu'
python -m mkdocs build --strict
```

使用 GPU 时再执行严格检查：

```bash
physrealx doctor --strict
```
