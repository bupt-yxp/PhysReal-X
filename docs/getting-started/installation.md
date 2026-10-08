# 安装与环境

PhysReal-X 核心支持 Python 3.8–3.12。所有依赖和 Jupyter 内核都使用现有 `PhysReal-X` Conda 环境；实际兼容性由方法探针与执行验证决定，不要求已有环境与官方安装模板完全一致。

```bash
conda activate PhysReal-X
python -m pip install -e '.[physical,colmap,demo,dev,docs,train,visual-learning]'
python -m ipykernel install --sys-prefix --name physreal-x --display-name PhysReal-X
```

同名 Conda 环境可能位于不同目录；Jupyter 右上角的 **PhysReal-X** 名称不足以确认环境。
在 Notebook 中执行 `import sys; print(sys.executable, sys.version)`，核对其完整解释器路径与验收回执。
验收以当前 Notebook 内核的 `sys.executable` 为准；Notebook、脚本和配置不得硬编码本机 Conda
安装路径。

更新 SDK 源码后也必须点击 **Restart（重启内核）→ 全部运行**。可编辑安装只让新导入读取最新文件，
不会替换正在运行的内核中的旧类和已有实例。若磁盘源码有 `OperationManifest.output()`，
当前对象却报没有 `output`，应先重启内核；重新打开 Notebook 或清除输出不等于重启。

若出现 `AttributeError: 'RcParams' object has no attribute '_get'`，已复现的原因是
Matplotlib 3.5.3 与 matplotlib-inline 0.2.2 不兼容。项目 Demo 与 ArtGS 现统一要求 Matplotlib ≥3.8。
使用发生错误的解释器执行 `python -m pip install --upgrade 'matplotlib>=3.8,<4'`，然后在 Jupyter 点击
**Restart**；仅重跑单元格不会替换内核内已导入的旧模块。

六模块完整 Notebook 使用 CPU COLMAP/深度处理与 CUDA ArtGS、ACT、DINO-WM。完整执行需要实际 CUDA 设备、编译器和原生扩展；重建训练子进程使用当前 `sys.executable`。ArtGS 依赖在同一环境执行 `python -m pip install --no-build-isolation -e ".[artgs]"` 安装。

本地元数据库使用 SQLite `STRICT` 表，因此系统 SQLite 必须为 3.37.0 或更高版本：

```bash
python -c 'import sqlite3; print(sqlite3.sqlite_version)'
```

## 新安装的可选模板

```bash
git clone https://github.com/bupt-yxp/PhysReal-X.git
cd PhysReal-X
conda env create -f environment.yml
conda activate PhysReal-X
python -m pip install --no-build-isolation -e '.[artgs,physical,colmap,dev,docs,demo,train,visual-learning]'
physrealx doctor
```

仅在尚无项目环境时使用 `environment.yml`；它提供 Python 3.11 / Torch 2.5.1 / CUDA 12.1 的已知配置，预装构建工具；
紧随其后的根项目 extra 再安装开发、文档、Demo、训练、ArtGS 与 PyCOLMAP 依赖。
`--no-build-isolation` 仅防止 PyTorch3D/CUDA 扩展在临时 pip 环境中丢失已安装的 Torch，
不会创建第二个运行环境。ArtGS 主体、Gaussian rasterizer、Simple-KNN、PointNet2 和 GLM 源码均在
`algorithms/artgs/`；来源、提交、许可证与平台修改记录见其中的 `SOURCE.toml`。
其中 Gaussian rasterizer 与 Simple-KNN 限于非商业研究/评估；安装 extra 不会
改变上游许可边界，详见根目录 `THIRD_PARTY_NOTICES.md`。

已有兼容环境时，可直接安装到当前环境：

```bash
python -m pip install --no-build-isolation -e '.[artgs,physical,colmap,dev,docs,demo,train,visual-learning]'
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
`artgs` extra 提供 Python 3.10–3.12 的已验证依赖组合；已有 Python 3.10–3.12 环境也可使用实际兼容的依赖。运行探针检查 CUDA 运算与依赖加载，执行前验证原生扩展，不以 Torch/CUDA 版本字符串相同代替执行验证。
`colmap` extra 使用官方 PyCOLMAP 4.1.1，适用于 Python 3.10–3.12；核心 SDK 在
Python 3.8–3.9 仍可使用，但不会注册 `camera.recover:colmap`。

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

camera_status = px.methods.inspect(MethodCapability.CAMERA_RECOVER, "colmap")
print(camera_status.ready, camera_status.detail)
```

`ready=True` 要求对应运行时通过真实 API 探针。ArtGS 还要求嵌入源码、CUDA/PyTorch、`nvcc`
与 GPU 一致；COLMAP 要求当前解释器可导入固定版本 PyCOLMAP，并暴露特征、匹配和增量 SfM
接口。只有描述符存在不算就绪；输入证据或资源不足时执行仍会在发布任何产物前失败。

ROS 2 和设备驱动属于系统级原生依赖，可在单独进程或机器人节点运行；这类进程边界用于设备
权限和故障隔离，不应产生另一套不受根项目管理的 Python 包版本。

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

## 可选物理链路

在同一 Python 3.10–3.12 项目环境安装 `python -m pip install -e '.[physical]'`。该 extra 固定 NumPy 1.26.4、SciPy 1.13.1、Trimesh 4.5.1、CoACD 1.0.9、MuJoCo 3.3.7 与 usd-core 24.11。核心 SDK 的 Python 3.8–3.12 支持范围不变；ArtGS 的 CUDA/扩展条件仍需独立满足。

模块二的受限 Delaunay 重网格还需要 C++17 与 CGAL/GMP/MPFR 开发库。Ubuntu 安装命令为 `sudo apt install build-essential libcgal-dev libgmp-dev libmpfr-dev`；首次运行会编译小型原生工作进程并缓存。视频联合优化同时使用重建环境中的 Torch/PyTorch3D 和 `physical` extra 中的 Warp、IPC Toolkit 和 HiGHS，具体方法及验收见[物理资产构建](../user-guide/physical-assets.md)。

无界面 RGB/Depth 渲染可在启动解释器前设置 `MUJOCO_GL=egl`；缺少可用图形驱动会明确失败。数值物理运行无需创建渲染器。

MuJoCo 无窗口渲染使用 `PyOpenGL==3.1.10`。DTArt 的间接依赖 pyrender 采用官方固定提交 `a59963ef890891656fd17c90e12d663233dcaa99`，其 PyOpenGL 约束已放宽；避免旧发行版将环境降回缺少 EGL 类型的 3.1.0。

## Demo 的额外运行依赖

房间、机械臂与多模态预览调用 Blender；可执行文件需在 PATH 中。模块二的 VLM 使用 Notebook 显式配置的解释器与固定模型版本，允许与重建环境隔离；图像、模型版本和几何一致时才复用预测。

Isaac Sim 使用独立安装的 4.5.0 与其 `python.sh`，通过 `PHYSREALX_ISAAC_SIM_ROOT` 指定。原生相机需要可用的 GPU/RTX 运行环境，详见[仿真后端](../user-guide/simulation-and-data-generation.md)。
