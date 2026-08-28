# 安装与环境

普通用户只需安装客户端并配置平台访问。Isaac Sim、Genesis、训练框架与 ROS 2 运行在隔离工作节点中，不进入用户的 Python 环境。

## 系统要求

| 项目 | 客户端要求 |
| --- | --- |
| 操作系统 | Linux x86_64 |
| Python | 3.11 或 3.12 |
| 存储 | 项目规格与下载缓存所需空间 |
| GPU | 不要求；仿真、训练和真机任务使用平台工作节点 |

工作节点的驱动、CUDA、GPU 和存储要求由具体后端决定，独立于客户端环境。

## 安装客户端

=== "Conda（推荐）"

    ```bash
    conda create -n PhysReal-X python=3.11
    conda activate PhysReal-X
    python -m pip install physreal-x
    ```

=== "Python venv"

    ```bash
    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install physreal-x
    ```

导入核心包不会初始化 CUDA、启动仿真器或连接机器人。

## 配置平台访问

```bash
physrealx profile add default \
  --server https://<平台地址> \
  --project embodied-manipulation
physrealx auth login --profile default
physrealx profile use default
```

配置档案保存平台地址、默认项目和凭据来源。访问令牌由系统密钥环、环境变量或部署平台的 Secret 管理，不写入项目文件和 Git。

验证身份与项目权限：

```bash
physrealx auth status
physrealx doctor
```

## 创建项目

```bash
physrealx project init open-drawer
cd open-drawer
```

推荐目录：

```text
open-drawer/
├── physrealx.yaml
├── assets/
├── scenes/
├── tasks/
├── runs/
├── training/
└── evaluation/
```

YAML 规格进入版本控制；图像、点云、权重和数据集上传到平台制品库，并通过内容摘要引用。

## 工作节点

共享平台已经配置工作节点，普通用户无需安装仿真后端。需要接入自有计算资源时，由平台管理员部署独立节点：

```text
平台控制面
    ├── Isaac Sim / Isaac Lab 工作节点
    ├── Genesis 工作节点
    ├── MuJoCo 工作节点
    ├── 训练工作节点
    └── ROS 2 机器人工作节点
```

每个节点向平台声明后端名称、版本、设备、物理能力、观测模式和动作类型。调度器在提交前完成能力匹配，用户 SDK 不直接连接节点。

查询当前项目可用能力：

```bash
physrealx workers list
physrealx workers inspect WORKER_ID
```

## 验证安装

```bash
physrealx doctor
physrealx project validate .
```

检查项包括：

- 客户端与 API 版本兼容性；
- 身份、项目权限和平台服务状态；
- 制品上传、下载和摘要校验；
- 任务所需后端与工作节点能力；
- 规格引用、单位、坐标系和信号模式。

如需机器可读结果：

```bash
physrealx doctor --output json
```

## 常见问题

### 客户端可登录，但任务一直等待

使用 `physrealx workers list` 检查是否存在满足后端版本、GPU、传感器和动作空间要求的可用节点，再检查项目配额与并发限制。

### 凭据在 CI 中不可用

为 CI 使用短期服务凭据，并由 CI Secret 注入；不要复制个人密钥环或把令牌写入 `physrealx.yaml`。

### 多个后端产生的信号不一致

运行 `physrealx validate run RUN.yaml`。平台会在提交前检查观测、动作、单位、坐标系和后端能力，避免执行后才发现模式不匹配。
