# 02 · 物理资产

读取模块一 USD 参考，执行逐 link CoACD、显式物性先验、接触排除、资产校验及 OpenUSD 封装回读。

在现有 PhysReal-X 环境中打开 [demo.ipynb](demo.ipynb)，选择 PhysReal-X 内核并全部运行。先完成上游模块；`python -m examples.run_all` 会自动按顺序执行六个模块。

资产保持 prior；质量、惯量和摩擦没有伪装为实测。

执行副本与回执写入本模块 `output/`；异常会抛出并停止下游。详细对照见 [功能验收](../../docs/acceptance.md)。
