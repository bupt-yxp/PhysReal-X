# 外部方法 worker

PhysReal-X 核心包只定义能力、资源和证据契约；COLMAP、OMNI-DC、Ditto、CoACD、
PhysReal-X 单关节系统辨识适配器与 OpenUSD 封装均在独立环境中执行。系统辨识
适配器产生可写入 MuJoCo 的参数，但不是 MuJoCo 上游组件。工作区只保存本机
`workers.d/*.json` 部署绑定，平台不会下载源码、权重或镜像。

## 已提供的适配器

| worker | 能力 | 关键输入 | 发布输出 |
| --- | --- | --- | --- |
| `colmap` | `camera.recover` | 版本化 `ObservationSetManifest/1` | 相机、稀疏重建与报告 |
| `metric_alignment` | `camera.align_metric` | 未定尺相机与独立米制对应 | 米制相机与留出报告 |
| `omni_dc` | `depth.enhance` | 米制已标定 RGB-D、留出深度和固定权重 | 增强深度与质量报告 |
| `ditto` | `articulation.recover` | 真实交互前后点云、标定和固定权重 | 部件/关节模型与恢复报告 |
| `coacd` | `collision.build` | 米制、具名 link 的版本化视觉 Mesh | 凸分解 GLB 与绑定输入/输出摘要的质量报告 |
| `physrealx_mujoco_sysid_adapter` | `physics.calibrate` | 单关节实测 q/qdot、作用量与训练-留出切分 | 等效惯量、黏性阻尼、库仑摩擦与留出报告 |
| `openusd` | `asset.package` | 视觉、碰撞、关节和已辨识物性闭包 | USD 资产、清单与往返验证报告 |

这些名称表示适配器契约，不表示当前机器已安装对应算法。只有部署清单、
环境摘要、固定源码/权重和实际探针全部通过时，`px.methods.inspect(...).ready` 才为真。

## 安装核心适配器

外部 worker 协议包支持 Python 3.10–3.12，建议用 Python 3.11 创建这些
隔离环境。核心 SDK 仍独立支持 Python 3.8–3.12。

```bash
python3.11 workers/install.py \
  --root /data/physrealx-workers/envs \
  --workspace /data/physrealx-workspace \
  --colmap-binary /usr/local/bin/colmap \
  colmap metric_alignment coacd mujoco_sysid openusd
```

OMNI-DC 和 Ditto 还必须先按官方项目安装独立 CUDA 环境、固定提交和权重，再按
[OMNI-DC worker](https://bupt-yxp.github.io/PhysReal-X/api/omnidc-worker/) 与
[Ditto worker](https://bupt-yxp.github.io/PhysReal-X/api/ditto-worker/) 登记。
只读环境或不可变镜像层可使用 `--register-only`；该模式仅重建环境证明，不会
伪造升级或就绪状态。

## 执行边界

- v2 请求物化资源 manifest 及其全部 `ArtifactRef` 闭包，每个依赖都记录 owner、
  JSON Pointer、SHA-256、媒体类型和相对路径。
- 输入在执行前重算摘要并只读物化；worker 返回后再验证清单、依赖闭包和请求索引。
- worker 不使用 shell，不继承未声明环境变量，并受超时、日志和产物大小限制。
- 输出必须与操作契约的名称、Schema、媒体类型及顺序完全一致；闭包验证通过后
  才进入制品库。
- 未安装、环境锁被修改、证据不完整或探针失败都返回明确不可用，不降级为插值、
  硬编码轨迹或合成 Episode。

## cabinet2 证据实验

```bash
python experiments/run_cabinet2_evidence.py \
  --source /data/ArtModeling/cabinet2 \
  --workspace /data/physrealx-workspace \
  --output /data/physrealx-experiment
```

脚本只执行当前证据真正支持的阶段，并在第一个未满足的科学前置条件处停止。
它不会用 `sim/`、旧 USD、硬编码抽屉或固定标量 Episode 替代真实标定、关节恢复、
物性辨识、训练或真机验证。
