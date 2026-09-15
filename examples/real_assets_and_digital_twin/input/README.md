# 输入数据

本目录存放 Demo 的原始输入数据。`ArtModeling/` 暂时由 `.gitignore` 排除，
数据只保留在本机，不进入 Git 提交。当前 Demo 约定的数据位置为：

```text
input/
└── ArtModeling/
    ├── cabinet2/
    ├── laptop1/
    └── oven4/
```

默认 Notebook 在 `assets.inspect_source` 单元格中直接选择
`input/ArtModeling/oven4`；处理其他对象时直接替换该 API 的 `source` 路径。
原始 RGB-D、逐帧 Mask 与配套元数据保留在 `input/`；Mesh、Gaussian、USD 和评测报告等派生结果写入 `output/`，不与原始输入混放。每个状态只有一个采集流，必须满足：

```text
<state>/img/<frame>.png
<state>/depth/<frame>.npy
<state>/mask/<frame>.png
```

三种模态使用完全相同的整数帧 ID；采集丢帧时允许编号不连续，不允许重新配错或伪造帧。Depth 是与 RGB 同分辨率的二维 uint16 NPY；Mask 使用 uint8 二值 PNG：背景 0、目标 255。Mask 属于数据输入，不由 PhysReal-X 在运行期生成。

输入不包含历史重建与补充对比结果。
