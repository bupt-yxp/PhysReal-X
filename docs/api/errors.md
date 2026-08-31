# 错误、状态与幂等

所有预期错误继承 `PhysRealXError`，包含稳定 `code`、可读 `message` 和结构化 `details`。调用方根据异常类型或 `code` 处理，不解析消息文本。

## 错误层级

```text
PhysRealXError
├── DomainValidationError
├── SourcePackageError
├── ResourceNotFoundError
├── ResourceConflictError
├── ArtifactIntegrityError
├── InvalidStateTransitionError
├── PluginError
│   └── PluginCompatibilityError
└── MethodExecutionError
    └── MethodUnavailableError
```

Pydantic 模型构造失败直接抛出 `pydantic.ValidationError`；多个已合法领域对象组合失败时抛出 `DomainValidationError`。

## 捕获错误

```python
from physrealx.errors import MethodExecutionError, SourcePackageError

try:
    source = px.assets.ingest(path, **options)
    result = px.recipes.execute(spec)
except SourcePackageError as exc:
    print(exc.code, exc.details.get("issue_codes"))
except MethodExecutionError as exc:
    print(exc.code, exc.details.get("issue_codes"))
```

## 重试规则

| 错误 | 默认是否重试 |
| --- | --- |
| 校验、冲突、能力不匹配 | 否；修改输入 |
| 源包、构建契约或插件错误 | 否；修复输入或插件 |
| 制品缺失或大小错误 | 否；隔离制品并调查来源 |
| 方法依赖未安装或就绪检查失败 | 否；修复项目可选依赖或方法配置 |
| 方法执行超时或异常退出 | 仅当任务本身声明可安全重试时 |

## 幂等语义

资源身份由 `resource_type + resource_id + version` 决定：相同身份返回原精确引用；内容变化必须
提升版本。领域对象直接注册时，若同一身份与现有清单语义不同则抛出冲突。

运行以 `RunSpec.run_id` 作为幂等身份；方法操作身份由精确方法、输入以及输出名称/语义契约的
稳定记录标识生成。
相同身份对应不同规格时抛出 `ResourceConflictError`。

```python
record = px.runs.submit(run_spec)
same_record = px.runs.submit(run_spec)
```

当前公共服务未提供取消、目录废弃或隐式覆盖接口。资源内容始终不可变，运行状态机的非法跳转
抛出 `InvalidStateTransitionError`。

## 部分结果

运行失败时：

- 已原子发布的 Episode 可以查询和诊断；
- 未完成或引用未验证的 Episode 不可见；
- 输出数据集不会发布为成功版本；
- `RunRecord.result` 在终态保存稳定错误码与信息。

训练、评测、导出和资产配方遵循相同原则：已成功的前缀操作可作为审计证据，但只有终端阶段
成功后才发布 `RecipeExecutionManifest`。

## 安全错误

部署方法必须把安全门失败作为结构化错误：

- 不自动重试动作；
- 部署进入受控停止或急停状态；
- 保存触发前后的证据窗口；
- 要求人工确认才能恢复。
