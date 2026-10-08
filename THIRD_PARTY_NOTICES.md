# 第三方源码与许可声明

PhysReal-X 自有代码使用 MIT 许可证。`algorithms/` 中的内嵌方法保留各自的上游许可，
不因归入同一代码仓库而改为 MIT。

## ArtGS

- ArtGS 主体：Apache-2.0，见 `algorithms/artgs/LICENSE`。
- `diff-gaussian-rasterization`：Gaussian-Splatting 自定义许可，仅允许非商业研究和评估，
  见 `algorithms/artgs/submodules/diff-gaussian-rasterization/LICENSE.md`。
- `simple-knn`：同上述 Gaussian-Splatting 自定义许可，见
  `algorithms/artgs/submodules/simple-knn/LICENSE.md`。
- GLM：MIT 或 Happy Bunny License，见
  `algorithms/artgs/submodules/diff-gaussian-rasterization/third_party/glm/copying.txt`。
- `utils/pointnet_lib`：上游 ArtGS 快照未为该目录附带独立许可文件，因此来源清单标记为
  `NOASSERTION`；商业发布前需由使用者完成单独许可审核。

由于 ArtGS 运行依赖上述受限及待审核组件，平台方法描述符使用
`LicenseRef-ArtGS-Research-Only`，不使用 Apache-2.0 冒充整体有效许可。

## DTArt

- `algorithms/dtart/`：NVlabs/DigitalTwinArt，提交
  `1a48b402e4bf4bb7731296e8e230f0db3d86fe4f`，保留 NVIDIA 非商业研究与评估许可。
- 本地改动记于 `algorithms/dtart/PHYSREALX.md`，许可见同目录 `LICENSE`。
- LoFTR 使用 Kornia 实现及 outdoor 预训练权重；权重不提交到源码仓库。

## COLMAP / PyCOLMAP

- `pycolmap` 是 COLMAP 的官方 Python 绑定，使用 BSD-3-Clause 许可证；它通过
  `python -m pip install -e '.[colmap]'` 作为可选依赖安装，不复制进本仓库。
- `camera.recover:colmap` 只在 PyCOLMAP 导入并通过最小功能探针后注册；核心平台未安装该
  extra 时不会发布不可执行的方法。
- `opencv-python` 使用 Apache-2.0 许可证，用于已声明 ArUco/AprilTag 家族的 ID 与
  亚像素角点检测；未声明字典时不自动猜测。

## SAM 2


## Optional physical pipeline bindings

The `physical` extra uses unmodified official distributions: NumPy 1.26.4 (BSD-3-Clause), SciPy 1.13.1 (BSD-3-Clause), Trimesh 4.5.1 (MIT), CoACD 1.0.9 (MIT), MuJoCo 3.3.7 (Apache-2.0), NVIDIA Warp 1.17.0 (Apache-2.0; its binary distribution includes separately licensed libmathdx), IPC Toolkit 1.6.0 (MIT), HiGHS 1.11.0 (MIT, `highspy`), and OpenUSD `usd-core` 24.11 (Apache-2.0 with its upstream modified terms). Their license files remain in the installed distributions. PhysReal-X adapters are first-party MIT code; no external cloned source tree or second interpreter is required.

## SceneSmith

- SceneSmith: MIT, pinned at commit `67cc408fd38334b4a926efef45e284302ed5055b`.
- PhysReal-X adapts its articulated retrieval, candidate generation and physical screening
  contracts. The complete upstream runtime is not copied into this repository; provenance and
  the upstream license are retained in `algorithms/scenesmith/`.

## Visual policy and world-model sources

- `algorithms/act/`: official LeRobot v0.3.3 ACT network, Apache-2.0, revision `b883328e6c95681ca90a18b102e4ae5e1f91e2bf`; framework integration removed, network equations retained.
- `algorithms/dinov2/`: official DINOv2 ViT encoder and required layers, Apache-2.0, revision `7764ea0f912e53c92e82eb78a2a1631e92725fc8`. The standard ViT-S/14 weights are obtained from Meta's official distribution.
- `algorithms/dino_wm/`: official causal ViT predictor, MIT, revision `0a9492fa12044b852ae9e001cc74604b79c8bb0c`. Device-safe masks, dependency injection, and the separately recorded platform residual/decoder adaptation are documented in `SOURCE.json`.

Each directory includes the upstream license and a source/adaptation manifest. Demo configuration and execution results do not claim upstream benchmark performance.

## CGAL（受限 Delaunay）

- 模块二的 `_restricted_delaunay.cpp` 使用系统 CGAL 的精确几何谓词和构造；CGAL 源码及二进制未内嵌到仓库。
- CGAL Delaunay/AABB 组件保留上游许可，参见系统安装的 CGAL 许可文件与 [上游许可说明](https://www.cgal.org/license.html)。运行时生成的工作进程不应被当作纯 MIT 二进制再分发。
- ArtMesh 的部件重网格及运动一致性公式作为方法参考；本项目不是作者完整训练代码的镜像。

## Mesh-Untangling

- `_surface_untangling.py` 依据 [官方 Gaussian 与 Minkowski 能量源码](https://github.com/Simulation-Intelligence/Mesh-Untangling/tree/main/include/energies) 独立实现 PyTorch 版本，加入表面积分、多姿态共享参数和观测约束。
- `algorithms/mesh_untangling/` 内嵌官方 CUDA 源码，锁定 `8cc1557f23562803c347b95ec24b7c445afe1d54`，保留 Apache-2.0 及 indicators/termcolor 许可；未包含数据集、结果和二进制。
- 原生路径使用官方自适应 Gaussian/Minkowski 展开和 IPC 恢复，平台扩展共享多姿态顶点投影与远场固定约束；具体修改见该目录 `SOURCE.toml`。这不等于已完成 ArtMesh，也不保证视频拟合或资产验收通过。
- 原有 PyTorch 局部联合求解器仍为独立路径；其固定带宽实现不被称为原生 Mesh-Untangling。所有候选须经过同一套导出网格、观测和连续运动验收。

## Clarabel

- 物理依赖中的 `clarabel==0.11.1` 用于观测约束解缠的稀疏二次投影；通过 Python 包调用，上游源代码未内嵌。
- 上游采用 [Apache-2.0](https://github.com/oxfordcontrol/Clarabel.rs/blob/main/LICENSE.md)。所有求解结果仍须通过本项目在原始单位下的约束与几何验收。
