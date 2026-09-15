# 02 · 物理资产

读取模块一 USD 参考，执行逐 link CoACD、显式物性先验、接触排除、资产校验及 OpenUSD 封装回读。

在现有 PhysReal-X 环境中打开 [demo.ipynb](demo.ipynb)，选择 PhysReal-X 内核并全部运行。先完成上游模块；`python -m examples.run_all` 会自动按顺序执行六个模块。

资产保持 prior；质量、惯量和摩擦没有伪装为实测。

Notebook 将业务资源引用导出到 `output/results.json`；执行脚本另存 `execution.json` 与运行副本，失败停止下游。详细对照见 [功能验收](../../docs/acceptance.md)。

oven4 参考包含开放 CAD 表面：碰撞预处理显式使用 1 µm 焊接容差，跨材质接缝按几何位置合并；开放表面以双向外表面 P95 ≤ 3 mm 校验，不把有向体积估计当作体积真值。质量来源是凸碰撞代理体积之和乘假定密度，仍为未实测先验。
