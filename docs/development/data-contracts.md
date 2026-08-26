# 数据契约与可追溯性

## 不可变资源

资产、场景、任务、数据集、模型、评测矩阵和报告一旦发布不可原地修改。资源身份为：

```text
resource_type + resource_id + semantic_version + sha256 digest
```

`resource_id@version` 便于人阅读，执行前必须解析为带摘要引用。

## 规范化序列化

摘要使用规范化 JSON：UTF-8、稳定键顺序、无多余空白、拒绝 NaN/Inf、时间标准化为 UTC。输入 YAML 与 JSON 若语义相同，应得到相同摘要。

## Episode 合同

普通步骤包含当前观测、状态、动作、奖励与折扣；最终步骤只包含最终观测和状态。时间戳严格递增，每个传感器可以有独立采样时间。

验证不变量：

- step index 从 0 连续；
- 仅首步 `is_first=True`；
- 仅末步 `is_last=True`；
- terminal 必须同时是 last；
- 成功必须有 terminal 末步且没有 failure reason；
- 失败必须有 failure reason；
- 每步观测、状态和动作模式不漂移；
- 资产、场景、任务引用均已解析。

## 多模态时间

控制时间线不等于传感器时间线。原始 Episode 保存所有真实时间戳；训练视图定义对齐策略并成为新的版本化转换：

```text
raw episode → align@1.0.0 → window@2.0.0 → normalized training view
```

不得在训练加载器中进行未记录的隐式插值。

## 原子制品发布

1. 写入临时对象；
2. 完成流式 SHA-256；
3. 同步并关闭；
4. 原子提升为摘要对象；
5. 重新验证大小和摘要；
6. 在事务中注册引用。

内容相同的制品自然去重；内容不同不能占用相同摘要路径。

## 可追溯关系图

```text
Capture ─→ Asset ─→ Scene ─→ Task ─→ Run ─→ Episode ─→ Dataset
                                                       │
                                                       └→ Training ─→ Model
                                                                        │
                                          Evaluation Matrix ────────────┤
                                                                        ↓
                                              Report ─→ Deployment ─→ Feedback
                                                                          │
                                                                          └→ Regeneration Run
```

所有边保存精确摘要。元数据目录可从任意报告反查模型、数据、任务、资产和原始采集。

## Schema 演进

每个顶层资源带 `schema_version`。演进规则：

- 新增有默认值的可选字段：可在同一 schema 主版本演进；
- 改变字段语义、单位、默认值或必填性：提升 schema 版本；
- 迁移是显式纯函数，输入旧资源、输出新资源和迁移报告；
- 迁移不覆盖旧摘要；
- 只维护受支持 schema 的迁移链，不在核心模型中永久保留旧字段别名。

## 真实、仿真与模型数据

`EpisodeSource` 必须明确：

- `real`：真实机器人或真实采集；
- `simulator`：物理后端执行；
- `world_model`：学习模型生成。

混合数据集保留每条 Episode 来源和父数据。模型生成数据不能冒充真实监督。

## 事务边界

- 单个 Episode：制品与摘要原子发布；
- 数据集：完整 Episode 集与清单原子发布；
- 模型：权重、输入输出模式和清单原子发布；
- 评测：指标、失败案例和报告原子发布；
- 运行终态与输出资源注册具有一致事务语义。

## 垃圾回收

只删除没有任何活动/保留资源引用、超过保留期且不在进行中上传的制品。垃圾回收必须先生成候选清单和审计记录，再执行可恢复删除策略。
