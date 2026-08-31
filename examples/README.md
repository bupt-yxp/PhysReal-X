# PhysReal-X 六模块 Demo

这里展示如何使用公开 Python SDK 编排六个业务模块。Demo 不包含算法实现；COLMAP、
Ditto、CoACD、仿真器和训练器必须先以项目可选依赖安装并注册为方法。

## 目录

| 模块 | Jupyter Demo |
| --- | --- |
| 真实资产接入与数字孪生 | [打开](real_assets_and_digital_twin/demo.ipynb) |
| 物理可信资产构建 | [打开](physical_asset_building/demo.ipynb) |
| 场景与任务生成 | [打开](scene_and_task_generation/demo.ipynb) |
| 物理仿真与多模态数据生成 | [打开](simulation_and_data_generation/demo.ipynb) |
| 数据管理与模型训练 | [打开](data_management_and_training/demo.ipynb) |
| 评测与真机反馈闭环 | [打开](evaluation_and_feedback/demo.ipynb) |

每个 `demo.ipynb` 按“单项功能说明 → 直接 SDK 调用 → 输出/门禁复核”组织。
方法、参数和待替换的证据引用均紧邻其 API；公共初始化只保留项目根目录、
工作区和输出目录。启动 Jupyter：

```console
python -m jupyter lab examples/real_assets_and_digital_twin/demo.ipynb
```

业务 API 全部直接出现在 Notebook 单元格中，不使用隐藏的流程函数、
预定义 YAML 或第二套 Demo 配置模型。

## 状态语义

- `COMPLETED`：配置的全部阶段真实执行并发布了契约产物；
- `BLOCKED`：缺少输入证据、方法依赖或探针未通过，没有伪造下游结果；
- `FAILED`：平台拒绝了无效契约、冲突资源或执行结果。

六个模块可以独立运行。跨模块衔接时，把上一模块发布的 `VersionedRef` 或
`ArtifactRef` 填入下一模块的对应 API 单元格，不传递临时文件名。
