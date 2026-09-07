# PhysReal-X

PhysReal-X 将真实采集、数字孪生、物理资产、场景任务、多模态仿真、策略与世界模型训练、评测反馈组织为可追溯的数据闭环。功能依据见两份 PPT 的[逐项对照](docs/acceptance.md)。

```text
实拍 RGB-D + 逆向建模 USD → 相机/深度/ArtGS/配准 → 统一物理资产与 USDZ
→ 场景与任务 → MuJoCo 多模态数据 → ACT / DINO-WM → 失败回流、恢复示范、再训练与配对评测
```

## 在现有环境运行

```bash
conda activate PhysReal-X
python -m ipykernel install --sys-prefix --name physreal-x --display-name PhysReal-X
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python -m examples.run_all
```

六个 Notebook 按依赖顺序执行，同用 `.physrealx`。输入是
`examples/real_assets_and_digital_twin/input/ArtModeling/cabinet2` 中的实拍与配套 USD。
也可逐个在 Jupyter 中选择 **PhysReal-X** 内核并全部运行。方法和参数直接写在各自 Notebook 中，跨模块自动读取已完成的上游回执。

完整环境已经在 Python 3.11、Torch 2.5.1+cu121 和 RTX 4090 上执行；ArtGS 使用当前解释器及仓库内 CUDA 扩展。新机器的依赖安装说明见[安装与环境](docs/getting-started/installation.md)。

## 六个模块

| Notebook | 实际执行内容 |
| --- | --- |
| [01 数字孪生](examples/real_assets_and_digital_twin/demo.ipynb) | 输入与环境、相机定尺、深度增强、几何与运动重建、位姿配准及关节校核 |
| [02 物理资产](examples/physical_asset_building/demo.ipynb) | CoACD、显式物性先验、PBR/纹理/UV、绑定高斯外观的 USDZ 封装与回读 |
| [03 场景任务](examples/scene_and_task_generation/demo.ipynb) | 机器人推杆与被动抽屉、有界布局、任务模板、控制/目标分离、MuJoCo 轨迹检查 |
| [04 仿真数据](examples/simulation_and_data_generation/demo.ipynb) | 双视角八种输出、刚体力觉、随机化、固定时长接触轨迹、实际状态驱动高斯外观 |
| [05 训练推演](examples/data_management_and_training/demo.ipynb) | 分组快照、ACT、DINO-WM 多步训练与基线比较、五秒事实/反事实真值检查 |
| [06 评测回流](examples/evaluation_and_feedback/demo.ipynb) | 实际失败归档、同初态恢复示范、防泄漏样本补充、ACT 再训练与独立种子配对评测 |

每个模块的 `output/` 保存执行副本、结果和回执；总报告为 `examples/output/notebook-acceptance.json`。必要阶段失败会抛出异常，不将跳过或占位输入记为完成。

## 架构

```text
examples / CLI
    ↓
领域服务：assets · scenes · tasks · runs · datasets · training · world_models · evaluations · feedback
    ↓
冻结领域契约 / 方法操作 / 运行与训练生命周期
    ↓ 协议
适配器：COLMAP · ArtGS · CoACD · OpenUSD · MuJoCo · ACT · DINO-WM
存储：SQLite 注册表 + 不可变制品库
```

核心导入不启动 CUDA、仿真器或网络。所有方法沿用一个工作区、资源身份和发布通道；输入、参数、种子、代码来源及训练环境随产物固定。算法源码保留在 `algorithms/`，不依赖工作区外部 clone 或第二套执行环境。

## 验收范围

真实 cabinet2 的 start/end 只移动下层抽屉，因此 ArtGS 学习两个运动组；USD 的三个结构部件、两个关节独立保留。尺度由 USD 高度辅助估计，再通过有留出检查的位姿/关节拟合对齐；仍属于模型条件证据，上层未观测运动与真实物性不由该配准证明。

USD 结构、默认质量/摩擦与仿真力觉均有明确来源。独立物性校准和真机验证需要实际测量、机器人资产、设备接口及部署日志。推杆演示、模拟事件、模型预测和软件测试不会被标为现场机械臂或独立物理验收。

## 开发验证

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy
MUJOCO_GL=egl python -m pytest -m 'not real_data' --cov=physrealx --cov-branch --cov-fail-under=80
python -m pip check
python -m build
python -m mkdocs build --strict
```

[用户指南](docs/user-guide/index.md) · [API](docs/api/index.md) · [方法目录](docs/api/capability-methods.md)

自有代码为 MIT。内嵌算法保留各自许可；ArtGS 的 rasterizer 与 Simple-KNN 限于非商业研究/评估，见[第三方声明](THIRD_PARTY_NOTICES.md)。
