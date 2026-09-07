# 01 · 真实资产与数字孪生

使用 cabinet2 实拍 RGB-D 与配套 USD，展示六个功能章节：输入与环境、相机与尺度、深度准备与增强、几何与运动重建、位姿配准与关节校核、结果与导出。算法由代码中的 `method` 指定。

在现有 PhysReal-X 环境中打开 [demo.ipynb](demo.ipynb)，核对完整解释器路径并全部运行。本模块是入口，没有上游模块；所选方法的实际依赖必须就绪。

实拍只移动下层抽屉，两个运动组与 USD 三个结构部件分别记录。米制定尺和位姿配准分别解决单位与空间对齐。

页面仅展示关键指标、RGB-D 和连续外观；完整证据及执行副本保存在 `output/`。主输出为 `reconstruction.zip`、`reference-model.json`、`reference-registration.json` 和 `digital-twin-report.json`。

CPU 表面融合保留为[可选诊断](../../docs/user-guide/real-assets-and-digital-twin.md#可选表面诊断)，不再作为默认重建的前置步骤。必要阶段失败会停止执行；证据范围见[验收说明](../../docs/acceptance.md)。
