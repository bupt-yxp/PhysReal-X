# 安装与环境

PhysReal-X 采用“核心客户端 + 隔离工作节点”的安装方式。核心环境负责规格、编排、资源目录和数据访问；Isaac Sim、Genesis、训练框架与 ROS 2 保留各自运行时，避免 CUDA、Torch 和 Python 依赖互相污染。

## 系统要求

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Linux x86_64 |
| Python | 3.11 或 3.12 |
| 存储 | 本地开发至少预留 20 GB；多模态数据按任务容量规划 |
| GPU | 仅仿真渲染、训练或 GPU 后端需要；驱动版本以对应后端要求为准 |
| Git | 用于获取源码和管理配置 |

## 创建核心环境

=== "Conda（推荐）"

    ```bash
    git clone https://github.com/bupt-yxp/PhysReal-X.git
    cd PhysReal-X
    conda env create -f environment.yml
    conda activate PhysReal-X
    ```

=== "Python venv"

    ```bash
    git clone https://github.com/bupt-yxp/PhysReal-X.git
    cd PhysReal-X
    python3.11 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    python -m pip install -e .
    ```

开发、测试和文档依赖：

```bash
python -m pip install -e '.[dev,train,docs]'
```

## 初始化工作区

```bash
physrealx workspace init .physrealx
```

工作区只保存平台元数据、缓存和本地制品；项目规格建议放在源码仓库中：

```text
my-project/
├── assets/
├── scenes/
├── tasks/
├── runs/
├── training/
├── evaluation/
└── .physrealx/
```

## 安装仿真后端

仿真后端必须安装在独立环境或容器中，并通过工作节点接入：

```text
PhysReal-X 核心环境
        │
        ├── Isaac Sim / Isaac Lab worker
        ├── Genesis worker
        ├── MuJoCo worker
        └── ROS 2 robot worker
```

不要把 Isaac Sim 自带的 Kit、Torch 和扩展依赖直接安装进核心环境。工作节点启动后向平台登记：后端名称、版本、API 版本、设备、支持的物理能力、观测键和动作类型。

## 配置存储

本地开发使用：

```yaml
# physrealx.yaml
endpoint: local:///.physrealx
metadata_store: sqlite
artifact_store: local
```

团队部署使用远程控制面与对象存储：

```yaml
endpoint: https://physreal.example.org
project: embodied-manipulation
```

凭据通过环境变量、系统密钥环或部署平台的 secret 管理，不写入 YAML 和 Git。

## 验证安装

```bash
physrealx doctor
physrealx workers list
```

检查项包括：

- Python 与核心包版本；
- 工作区读写和制品摘要校验；
- 可用仿真、训练和机器人工作节点；
- CUDA、驱动与设备可见性；
- 后端 API 兼容性和能力声明。

如需机器可读结果：

```bash
physrealx doctor --json
```

## 常见问题

### 核心环境能看到 GPU，但仿真工作节点不可用

分别检查宿主驱动、容器设备映射、后端自身启动日志和工作节点注册状态。核心环境能导入 Torch 不代表 Isaac Sim 工作节点已经就绪。

### 本地制品增长过快

先使用 `physrealx artifacts usage` 查看按数据集和媒体类型统计的占用，再使用保留策略清理没有清单引用的缓存制品。不要直接删除摘要目录中的文件。

### 多个后端产生的信号不一致

运行 `physrealx validate run RUN.yaml`。平台会在提交前检查任务要求的观测、动作、单位、坐标系和后端能力，避免运行后才发现模式不匹配。
