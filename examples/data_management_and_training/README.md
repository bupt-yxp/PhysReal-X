# 数据管理与模型训练

Episode 集合、数据转换/快照方法和训练计划均紧邻对应 API 显式展示：

```console
python -m jupyter lab examples/data_management_and_training/demo.ipynb
```

Notebook 分别演示数据接入、快照、策略/世界模型训练、模型登记和推演。策略和世界
模型必须分别形成模型包；训练数据、切分、代码、环境、种子和权重制品必须冻结，
不能只凭训练损失登记模型。
