# 客户端与公共约定

`PhysRealX` 是工作区内的平台门面：领域对象留在轻量核心，仿真器、重建算法和训练框架通过
注册方法接入。它不通过不同 URI 把“本机直连”和“远程控制面”伪装成同一个连接对象。

## 打开工作区

```python
from physrealx import PhysRealX

px = PhysRealX.open(".physrealx")
```

打开工作区会初始化本地元数据数据库和标识符寻址制品存储，并加载内置及入口点方法；导入
`physrealx` 本身不会创建工作区、初始化 CUDA 或启动仿真器。

```python
px = PhysRealX.open(".physrealx", discover_plugins=False)
```

`discover_plugins=False` 适合测试和严格受控环境，只保留平台内置参考方法。

## 服务边界

| 服务 | 职责 |
| --- | --- |
| `methods` | 查询、校验并绑定已安装能力方法 |
| `assets` | 预检/导入已采集数据，构建、注册和验证数字孪生 |
| `scenes`、`tasks` | 注册并读取不可变场景与任务 |
| `runs` | 编译并执行仿真运行，读取 Episode |
| `datasets` | 读取运行原子发布的数据集 |
| `training` | 提交训练并读取模型清单 |
| `evaluations` | 执行独立种子评测并读取报告 |
| `artifacts` | 原子写入、读取并检查标识符寻址制品 |

业务代码不直接访问 SQLite，不持有仿真器 session，也不从模型路径推断版本。

## 能力方法

```python
from physrealx import MethodCapability

methods = px.methods.list(MethodCapability.POLICY_TRAIN)
trainer = px.methods.bind(
    MethodCapability.POLICY_TRAIN,
    "torch_bc",
    config={"epochs": 80, "device": "cpu"},
)
```

`MethodDescriptor` 是已安装实现的审计身份；`MethodSpec` 是某次配方或任务使用的冻结选择。
名称相同但源码 revision、许可证、输入输出契约或版本不同的方法不会被视为同一实现。

## 资源引用

`VersionedRef` 使用 `resource_type + resource_id + version` 表达不可变逻辑身份，并通过
`artifact_id` 指向平台制品。已发布版本不能覆盖，内容变化必须使用新版本。

```python
scene = px.scenes.get(scene_ref)
assert scene_ref.artifact_id.startswith("art_")
```

`ArtifactRef` 只包含平台生成的 `artifact_id`、字节数和媒体类型，不暴露可变文件路径，也不把
文件哈希当作业务身份。平台通过私有存储目录、原子写入、不可覆盖规则和声明大小检查维护制品边界。

## 写入与失败

`px.artifacts.export(destination, results, files={...})` 负责文件导出：先暂存并校验全部制品，再替换文件，最后写入 `results.json` 完成标记。校验失败保留此前输出；替换阶段失败则不保留旧完成标记。该接口不宣称多个外部文件可同时原子替换，调用方应避免并发写入同一导出目录。

- 同一资源身份重复注册时返回原引用；要发布变化后的内容必须显式提升资源版本。
- 运行 ID 固定请求身份；成功运行重复提交时返回原记录，失败运行不会自动伪装成新任务。
- 资产和数据集在所有子制品通过存在性、大小、Schema 与质量检查后原子发布，失败不会留下
  可见半成品。
- 预期错误继承 `PhysRealXError` 并携带稳定 `code`；CLI 默认输出错误码和消息，`--debug`
  才显示回溯。

## 并发与生命周期

每个 `PhysRealX` 实例拥有自己的仓储、制品存储和方法目录，不使用进程全局注册表。后端 session、
策略和临时资源由运行执行器按 `create → reset/step → close` 生命周期管理，异常路径同样关闭。

远程调度需要单独的控制面客户端和认证边界；本地工作区对象不接受 `local://` / `https://`
端点切换。二者可以共享领域模型，但不共享连接生命周期或错误重试实现。
