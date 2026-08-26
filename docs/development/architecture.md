# 架构与 SOLID 原则

## 平台定位

PhysReal-X 是仿真器之上的具身数据与实验操作系统。它不实现物理 solver，而是统一真实资产、数字孪生、任务、仿真、Episode、数据集、训练、评测和真机反馈。

## 系统上下文

```text
研究者 / 数据工程师 / 机器人开发者
                    │
          SDK / CLI / Web / Remote API
                    │
┌────────────────── PhysReal-X 控制面 ──────────────────┐
│ 资源目录  任务编译  运行编排  数据集  训练  评测  反馈 │
└───────────────────────┬───────────────────────────────┘
                        │ 版本化规格 / 任务租约 / 事件
┌───────────────────────┴───────────────────────────────┐
│ 资产工作节点 │ 仿真工作节点 │ 训练工作节点 │ 机器人节点 │
└───────────────────────┬───────────────────────────────┘
                        │
             元数据仓库 + 内容寻址制品库
```

## 分层职责

### 领域层

只包含不可变模型、值对象、状态机和纯校验规则。禁止文件系统、网络、数据库、GPU、Torch、ROS 2 和仿真器副作用。

### 端口层

定义业务真正需要的最小协议：Repository、ArtifactStore、JobQueue、SimulationBackend、Policy、Trainer、AssetBuilder、DatasetExporter、RobotBackend 和 FeedbackAnalyzer。

### 应用层

实现用例：资源注册、任务编译、运行提交、Episode 发布、数据集查询、训练、评测、部署和再生成。应用层只依赖领域和端口。

### 适配器层

将 USD、Isaac、Genesis、MuJoCo、PyTorch、LeRobot、ROS 2、PostgreSQL、S3 和消息队列转换为端口。外部 SDK 类型不得穿透适配器边界。

### 入口层

SDK、CLI、REST/gRPC 和 Web 控制台只做认证、输入转换、调用应用服务和输出呈现，不复制领域规则。

## SOLID 落地

### 单一职责

- `AssetService` 不执行仿真；
- `RunOrchestrator` 不解析后端内部状态；
- `DatasetService` 不训练模型；
- `Trainer` 不注册资源目录；
- `RobotBackend` 不决定任务成功语义。

### 开闭原则

新增 Genesis、世界模型或数据导出格式通过新适配器完成，不修改 Episode 与应用服务主流程。

### 里氏替换

所有后端都必须满足相同生命周期、数据模式和失败语义；若某后端无法提供接触力，应通过能力声明拒绝，而不是返回伪造零值。

### 接口隔离

仿真后端、策略、训练器、制品存储和仓库使用不同协议。插件不被迫实现无关方法。

### 依赖倒置

应用层依赖抽象端口；本地 SQLite/SHA-256 存储和分布式 PostgreSQL/S3 都是可替换适配器。

## 控制面服务

| 服务 | 关键职责 | 不负责 |
| --- | --- | --- |
| Catalog | 不可变资源注册、解析、检索和退役 | 大制品内容传输 |
| Artifact | 上传、下载、摘要、保留和垃圾回收 | 业务资源语义 |
| Compiler | 场景/任务编译、能力匹配、执行计划 | 运行物理仿真 |
| Orchestrator | 状态机、租约、取消、恢复、预算和最终发布 | 后端物理细节 |
| Dataset | 查询、切分、合并、质量、统计和导出 | 模型训练循环 |
| Training | 训练任务、资源、检查点和模型登记 | 评测结论 |
| Evaluation | 冻结矩阵、指标、失败归因和比较 | 修改训练数据 |
| Deployment | 机器人门禁、部署生命周期和审计 | 自动放宽安全策略 |
| Feedback | 聚类失败、定位缺口、生成补数计划 | 隐式修改历史资源 |

这些是逻辑服务，不要求一开始拆成九个微服务。同一进程内仍需保持边界；只有负载、故障域或团队边界要求时再独立部署。

## 数据面

### 元数据仓库

保存小型规格、引用、状态、索引、审计和事务。团队部署使用关系数据库；本地开发使用 SQLite 实现同一 Repository 端口。

### 制品库

保存图像、点云、USD、Gaussian、Episode 分片、视频、权重和报告附件。对象名由摘要决定，URI 只是位置。

### 事件与任务队列

任务租约、进度事件和重试不作为资源真相来源。最终资源与状态仍由事务仓库决定，避免消息重复导致双发布。

## 运行原子性

```text
compile plan
  → create run(pending)
  → lease work
  → publish each Episode artifact + summary
  → verify expected set and quality gates
  → publish DatasetManifest
  → transactionally mark run succeeded
```

数据集清单和成功终态在同一事务语义中提交。失败运行可以保留诊断 Episode，但不能发布成功数据集。

## 研究成果落位

| 研究能力 | 平台位置 | 适配器类型 |
| --- | --- | --- |
| RDFC-GAN 深度补全 | 真实采集预处理 | `AssetBuilder` 子步骤 |
| DTF-Net 位姿与形状 | 数字孪生构建 | `AssetBuilder` |
| PD2GS 部件解耦与连续状态 | 资产视觉/结构构建与演示渲染 | `AssetBuilder` + renderer adapter |
| ArtGen 条件生成 | 资产扩展 | generative `AssetBuilder` |
| 高保真演示生成 | 场景、轨迹与渲染 | `Policy` + simulation adapter |
| CAPE 动作条件动力学 | 世界模型训练、规划和反事实 | `Trainer` + `Policy` + generator |
| VQ-VAE + Diffusion Policy | 策略学习 | `Trainer` + inference `Policy` |

论文方法的原生张量和脚本参数停留在适配器内部；平台只接收规范资产、信号、Episode、模型和报告。

## 建设顺序

### 第一阶段：稳定合同与参考纵向切片

优先完成 `AssetSpec`、`SceneSpec`、`TaskSpec`、`RunSpec`、Episode、数据集、模型、评测以及本地原子存储。用可控抽屉任务验证全链路。

### 第二阶段：生产仿真与多模态数据

接入 Isaac Sim/Isaac Lab，跑通 USD、RGB、Depth、关节、接触、多相机、并行运行、取消恢复和大制品写入。

### 第三阶段：训练与标准数据出口

依次接入可扩展模仿学习、LeRobot/RLDS 导出和在线 RL，验证流式读取、检查点与冻结评测。

### 第四阶段：真实资产与 Sim-to-Real

接入真实采集、PD2GS/重建、碰撞生成、物理校准和真实/仿真 1:1 重播。

### 第五阶段：VLA、世界模型和真机闭环

在数据和评测合同稳定后接入 VLA、CAPE 类世界模型、ROS 2 部署、失败归因和定向再生成。

## 最大工程风险

最大风险是同时搭建大量空壳模块，却没有一个生产级纵向切片。每阶段必须用真实制品、真实运行时、真实吞吐和失败恢复验收，不能以“类已经存在”代表能力完成。

## 明确禁止的反模式

- 在核心代码中 `if backend == "isaac"`；
- 用可变路径或 `latest` 表示正式资源身份；
- 把图像、点云或权重嵌入 JSON；
- 将任务成功写死在策略或后端中；
- 异常后仍发布部分数据集；
- 用零值伪装后端不支持的信号；
- 在核心长期保留旧字段别名和双 API；
- 为尚无消费者的假想功能创建抽象层。
