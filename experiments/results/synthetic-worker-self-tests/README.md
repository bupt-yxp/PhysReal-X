# 隔离 worker 合成自检证据

此目录仅证明算法适配器能够执行及拒绝不合格输入，不是 `cabinet2` 的实测结果。

- `sysid/` 使用已知参数的合成关节测量，验证有界多初值辨识、留出集改进和置信区间。
- `openusd/` 使用已解析尺度、关节、视觉/碰撞网格及已辨识物性的合成闭包，验证自包含 USD 与 40 项往返检查。
- `coacd/` 由已安装的 CoACD 1.0.14 worker 实际分解米制合成 L 形凹网格：20 个输入三角形生成 2 个凸部件、28 个输出三角形，碰撞/源体积比为 1.0003866。`operation_manifest.json` 绑定输入摘要、方法、环境和两个输出契约。

在已登记且探针通过 CoACD 的工作区中可重跑该证据；输出目录必须为空：

```bash
PYTHONPATH=src python experiments/run_synthetic_coacd_evidence.py \
  --workspace /data/physrealx-workspace \
  --output /data/physrealx-coacd-self-test
```

`cabinet2` 的真实证据与阻塞项位于相邻的 `cabinet2-lifecycle/`；两类结果禁止混用。
