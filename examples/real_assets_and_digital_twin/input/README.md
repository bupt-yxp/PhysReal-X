# 输入数据

本目录存放 Demo 的原始输入数据。`ArtModeling/` 暂时由 `.gitignore` 排除，
数据只保留在本机，不进入 Git 提交。当前 Demo 约定的数据位置为：

```text
input/
└── ArtModeling/
    ├── cabinet2/
    └── oven4/
```

默认 Notebook 在 `assets.inspect_source` 单元格中直接选择
`input/ArtModeling/cabinet2`；处理其他对象时直接替换该 API 的 `source` 路径。
原始 RGB-D 与配套元数据保留在 `input/`；Mesh、Gaussian、USD 和评测报告等派生结果写入
`output/`，不与原始输入混放。

当前本机 Demo 输入内容：

| 内容 | 文件数 | 逻辑字节数 |
| --- | ---: | ---: |
| `cabinet2/` | 1,042 | 1,827,235,895 |
| `oven4/` | 814 | 1,719,748,846 |
| `artmodeling_structure.pdf` | 1 | 37,433 |

输入不包含历史重建与补充对比结果。
