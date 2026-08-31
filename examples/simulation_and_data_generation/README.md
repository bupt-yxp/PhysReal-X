# 物理仿真与多模态数据生成

已注册场景、任务和已安装仿真/渲染/传感器方法均在对应 API 单元格中绑定：

```console
python -m jupyter lab examples/simulation_and_data_generation/demo.ipynb
```

Notebook 分别演示物理执行、轨迹重放渲染、事件流、点云和 Episode 封装。渲染必须
重放同一条物理状态轨迹；Episode 只有在状态、帧、动作和任务判据完全对齐后才可封装。
