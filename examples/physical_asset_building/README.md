# 物理可信资产构建

Notebook 在对应 API 单元格中就地绑定逐 link 视觉 Mesh、实测系统辨识证据、
资产组装配方和已注册组装资源：

```console
python -m jupyter lab examples/physical_asset_building/demo.ipynb
```

Notebook 分别演示碰撞构建、物性辨识、资产组装和 OpenUSD 封装。单关节辨识结果
不会被当作完整资产物性；OpenUSD 只消费闭包完整的组装资源。
