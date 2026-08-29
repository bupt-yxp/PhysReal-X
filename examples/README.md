# 可复现实验

## cabinet2：真实采集数据到数字孪生资产

`cabinet2_end_to_end.py` 对已采集的 ArtModeling RGB-D 数据执行全量导入，并核验既有
ArtGS 位姿、渲染、网格和关节产物，随后完成碰撞几何简化、物性补全、USD 检查与
PhysReal-X 资产发布。

实验环境需要 NumPy 1.26、OpenCV、Pillow、SciPy、scikit-image、Matplotlib、Trimesh、
Open3D 和 OpenUSD。必须显式提供源数据、ArtGS 预处理目录、ArtGS 输出目录与全新的
输出目录：

```bash
PYTHONPATH=src python examples/cabinet2_end_to_end.py \
  --source /path/to/ArtModeling/cabinet2 \
  --artgs-processed /path/to/processed/cabinet2 \
  --artgs-output /path/to/output/cabinet2/artgs \
  --output /path/to/new/evidence-run
```

输出包含 `metrics.json`、`metrics.csv`、`asset.json`、`build_manifest.json`、中文报告和
六张阶段图。脚本不会重新训练 ArtGS，也不会把缺失标定或推断物性伪装为实测值；证据
来源和非阻塞质量问题会保留在构建清单中。
