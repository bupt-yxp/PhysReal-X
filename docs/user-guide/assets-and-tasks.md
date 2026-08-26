# 资产与任务

PhysReal-X 将可训练数字资产与任务语义分开建模。资产描述“世界中有什么”，任务描述“需要完成什么、允许观察什么以及如何控制”。

## 资产规格

`AssetSpec` 是冻结且带语义版本的数字资产清单，主要包含：

- 根链接与链接集合；
- 可视和碰撞几何；
- 关节类型、父子关系、轴与限位；
- 质量、质心、惯量、摩擦和恢复系数；
- 参数来源、不确定度与置信度；
- 外部网格、USD 或高斯资产的制品引用。

大体积几何数据不直接嵌入 JSON，而是保存到制品库；规格只保留 URI、摘要和语义信息。

## 任务规格

`TaskSpec` 通过以下内容定义可执行任务：

- 目标资产引用；
- 自然语言任务说明；
- 初始条件、成功条件和终止条件；
- 观测信号的数据类型、形状、单位、坐标系和频率；
- 动作类型、信号结构和上下界；
- 最大执行步数。

参考开抽屉任务要求抽屉初始位置不超过 `0.05 m`，成功位置不低于 `0.30 m`，动作是范围为 `[-0.08, 0.08] m` 的单维关节增量。

## 校验规格

资产和任务文件可以使用 JSON 或 YAML：

```bash
physrealx validate asset asset.yaml
physrealx validate task task.yaml
```

领域模型会拒绝：

- 未声明字段；
- 非有限数值；
- 非法语义版本与摘要；
- 重复链接、几何或关节；
- 不一致的形状、范围、单位和坐标系；
- 引用不存在链接或资产的任务。

## 注册不可变版本

```bash
physrealx add asset asset.yaml --workspace .physrealx
physrealx add task task.yaml --workspace .physrealx
```

注册后，平台会将规范序列化为规范化 JSON，计算 SHA-256，并以“资源类型、资源标识、版本”作为不可变键。同一版本不能被不同内容覆盖；修改内容时必须发布新版本。

## 使用 Python 构建规格

```python
from physrealx.reference import reference_asset, reference_task

asset = reference_asset()
task = reference_task(asset)

print(asset.resource_id, asset.version)
print(task.instruction)
```

公开领域类型见[Python 接口](../reference/python-api.md)，版本解析和执行过程见[数据生成、训练与评测](workflow.md)。
