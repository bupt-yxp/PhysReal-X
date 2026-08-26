# 安装与环境

PhysReal-X 的核心功能支持 Python 3.11 和 3.12。CUDA 训练不是核心包的强制依赖；只有使用参考训练器或其他 GPU 适配器时才需要安装 PyTorch。

## 使用 Conda 创建完整环境

推荐使用仓库内的环境文件创建 Python 3.11 与 CUDA 12.1 训练环境：

```bash
conda env create -f environment.yml
conda activate PhysReal-X
physrealx doctor
```

诊断命令会检查 Python、Conda、PyTorch、CUDA 运行时和可见 GPU：

```bash
physrealx doctor --json
```

需要在自动化环境中严格检查时，可以增加 `--strict`。发现运行时问题后，该命令会返回非零退出码。

```bash
physrealx doctor --strict
```

## 安装核心开发环境

如果只需要规格校验、数据生成或 CPU 开发，无需安装 PyTorch：

```bash
python -m pip install --editable '.[dev]'
```

如需同时构建文档：

```bash
python -m pip install --editable '.[dev,docs]'
```

## CUDA 说明

项目参考 GPU 环境使用 PyTorch 2.5.1 与 CUDA 12.1。PyTorch Wheel 已包含用户态 CUDA 运行库，系统只需提供兼容的 NVIDIA 驱动。

只有编译自定义 CUDA 扩展时才需要指定本机工具链：

```bash
CUDA_HOME=/usr/local/cuda-12.1 \
PATH=/usr/local/cuda-12.1/bin:$PATH \
python -m build
```

不要为了项目修改系统全局 CUDA 软链接。

## 验证安装

```bash
physrealx --help
physrealx doctor
python -c "import physrealx; print(physrealx.__version__)"
```

随后可以进入[快速开始](quickstart.md)，运行内置抽屉参考闭环。
