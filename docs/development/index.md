---
title: 开发总览
description: PhysReal-X 的架构边界、扩展协议、数据契约与工程质量规范
hide:
  - toc
---

# 开发总览

<div class="prx-page-lead" markdown>

PhysReal-X 保持小而稳定的领域核心，同时让数字孪生、仿真、数据、训练、世界模型和机器人能力独立演进。扩展从协议和契约开始，不把外部运行时类型带入核心领域。

[阅读架构原则](architecture.md){ .md-button .md-button--primary }
[开发第一个插件](plugin-development.md){ .md-button }

</div>

## 依赖方向

```text
领域模型 ← 应用端口 ← 应用服务 ← 基础设施与能力适配器 ← 用户入口
```

任何依赖反向流动都需要架构审查。例如 `TaskSpec` 不能导入 Isaac Sim 类型，数据集清单不能依赖 LeRobot 的内部类，应用服务不能直接写 S3 SDK 调用。

## 开发文档

<div class="prx-route-grid">
  <a class="prx-route-card" href="architecture/"><strong>架构与 SOLID 原则</strong><span>平台分层、服务边界、依赖方向和研究能力落位。</span></a>
  <a class="prx-route-card" href="data-contracts/"><strong>数据契约与血缘</strong><span>不可变资源、Episode、内容寻址、原子发布和迁移。</span></a>
  <a class="prx-route-card" href="plugin-development/"><strong>插件开发</strong><span>仿真、训练、资产、导出和机器人适配器实现规范。</span></a>
  <a class="prx-route-card" href="testing/"><strong>测试与质量门禁</strong><span>单元、契约、集成、端到端、性能与真机验收。</span></a>
</div>

## 仓库建议结构

```text
PhysReal-X/
├── src/physrealx/
│   ├── domain/          # 纯领域模型与规则
│   ├── ports/           # 仿真、存储、训练、机器人等协议
│   ├── application/     # 资产、运行、数据集、训练、评测服务
│   ├── infrastructure/  # 本地/远程仓库、制品、队列与传输
│   ├── sdk/             # 同步与异步客户端
│   └── cli/             # 薄命令行
├── packages/
│   ├── physrealx-isaac/
│   ├── physrealx-genesis/
│   ├── physrealx-mujoco/
│   ├── physrealx-ros2/
│   └── physrealx-lerobot/
├── tests/
│   ├── unit/
│   ├── contract/
│   ├── integration/
│   ├── e2e/
│   └── performance/
├── docs/
└── .github/workflows/
```

后端依赖较重且运行时冲突明显，优先独立包和独立环境，不放进核心包的默认依赖。

## 变更流程

1. 先在用户指南或 API 文档中定义问题和公共语义；
2. 更新领域模型或端口，并写失败用例；
3. 实现应用服务，不接触具体后端；
4. 在适配器中完成外部类型转换；
5. 通过共享契约测试和目标运行时集成测试；
6. 更新示例、迁移说明和文档；
7. 运行全部质量门禁后合并。

## 不保留无成本的兼容层

项目采用显式语义版本和 schema 版本。若旧接口语义错误或会污染长期边界，应删除并提供一次性迁移工具，不在核心长期维护别名、双写、静默字段映射和永久弃用分支。
