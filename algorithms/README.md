# 内嵌算法源码

本目录保存由 PhysReal-X 统一维护、统一依赖和统一注册的开源算法源码快照。算法不再依赖工作区外部仓库，也不创建独立 Python/Conda 环境。

每个算法目录必须包含上游许可证、`SOURCE.toml` 来源清单和平台集成说明；`SOURCE.toml` 同时锁定可发布源码树摘要，任一源文件被替换都会使方法探针失败。数据集、权重、运行结果、编译缓存及嵌套 `.git` 不进入主仓库。项目根目录 `pyproject.toml` 是 Python 依赖的唯一来源，`environment.yml` 是推荐 Conda/CUDA 基线。
