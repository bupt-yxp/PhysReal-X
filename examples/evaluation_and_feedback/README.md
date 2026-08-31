# 评测与真机反馈闭环

固定基准、机器人接口、安全部署方法和已确认故障查询均在相应 API
单元格中就地绑定：

```console
python -m jupyter lab examples/evaluation_and_feedback/demo.ipynb
```

Notebook 分别演示固定评测、安全部署、遥测、失败归因和数据回流。部署阶段只生成
并验证计划；真正的 ROS2 控制由节点本地 Agent 执行，并强制经过 shadow、限速/限力
canary 和硬件安全门。反馈始终创建新数据集版本。
