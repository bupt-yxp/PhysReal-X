# 数据集管理

PhysReal-X 按“观测—动作—状态—结果”组织 Episode。运行完成后，平台先校验每条 Episode 制品，
再发布引用这些制品的不可变 `DatasetManifest`。

## 读取清单与 Episode

```python
result = px.runs.wait(run)
dataset = px.datasets.get(result.dataset)
episodes = px.runs.episodes(run)

assert len(dataset.episodes) == len(episodes)
assert dataset.total_steps == sum(len(item.steps) for item in episodes)
```

清单中的 Episode 摘要记录运行 ID、逻辑索引、成功状态、步数、失败原因、内容摘要和制品引用。

## 质量检查

至少验证：

- 制品 SHA-256 和字节数完整；
- Episode 索引、控制时间戳和各传感器时间戳单调；
- 观测/状态信号集合及 dtype、shape、单位、frame 稳定；
- 非末帧具有动作、奖励和折扣，末帧不携带这些字段；
- 成功、终止和失败原因一致；
- 数据集总步数、来源运行和 Episode 摘要一致；
- 训练/验证/测试按资产、场景和采集批次无交叉泄漏。

## 训练格式

内部清单保持平台语义；独立安装的 LeRobot v3 适配器可生成训练视图，RLDS 用于外部
Episode 交换，rosbag2 + MCAP 保存真机原始流，DVC 管理大型数据快照。每次转换
都必须产生新制品、质量报告与完整来源。核心仓库不自动安装 LeRobot，也不在适配器
缺失时生成固定标量 Episode。

## 数据卡

派生数据集应记录用途和不适用范围、真实/仿真/世界模型比例、资产与失败分布、模态频率和同步
预算、随机化范围、质量结果、许可证、隐私和已知偏差。删除前必须检查模型、评测和部署引用，
不能直接清空对象存储目录。
