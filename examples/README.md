# 可复现实验

## cabinet2：真实采集数据到数字孪生资产

`cabinet2_end_to_end.py` 对已采集的 ArtModeling RGB-D 数据和全新 ArtGS 训练输出执行完整核验，
检查 ArUco 位姿、深度、检查点、渲染、网格和关节产物，随后完成碰撞几何简化、物性补全、
OpenUSD 资产生成与 PhysReal-X 资产发布。

实验环境需要 NumPy 1.26、OpenCV、Pillow、SciPy、scikit-image、Matplotlib、Trimesh、
Open3D 和 OpenUSD。必须显式提供源数据、ArtGS 预处理目录、全新训练运行根目录与全新的
输出目录：

```bash
PYTHONPATH=src python examples/cabinet2_end_to_end.py \
  --source /path/to/ArtModeling/cabinet2 \
  --processed /path/to/processed/cabinet2 \
  --training-run /path/to/fresh-training-run \
  --output /path/to/new/evidence-run
```

输出包含 `metrics.json`、`metrics.csv`、`asset.json`、中文报告、阶段图、碰撞网格和
`cabinet2.usdc`。脚本要求 10,000/5,000/20,000 轮训练日志均已完成，且新检查点齐全；
不接受历史预计算输出充当本次训练结果。
