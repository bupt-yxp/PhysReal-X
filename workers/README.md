# 外部方法 worker

PhysReal-X 核心包不导入 CoACD、MuJoCo、BlenderProc 或 LeRobot。每个方法安装在独立环境中，通过受约束的 JSON/文件协议执行；工作区只保存机器本地的 `workers.d/*.json` 部署绑定。

## 安装

```bash
python workers/install.py \
  --root /data/physrealx-workers/envs \
  --workspace /data/physrealx-workspace \
  --blender-runtime /data/blender/blender-4.2.1-linux-x64 \
  coacd mujoco blenderproc lerobot
```

安装器会执行三项门禁：依赖安装后 `pip check`、在工作区写入包版本与运行文件摘要、生成包含探测命令的部署清单。普通 worker 固定 NumPy 1.26.4；Blender 4.2 的嵌入式运行时使用其内置 NumPy 1.24.3，探针严格拒绝 NumPy 2.x。

只读环境或不可变镜像层可使用 `--register-only`：该模式不修改环境，只校验依赖并重新生成带 SHA-256 的部署证明。

## 执行边界

- 输入在执行前按 SHA-256 验证，并以只读文件物化。
- worker 不使用 shell，不继承未显式声明的环境变量，且受超时、日志和产物大小限制。
- 返回文件必须与操作契约的名称、类型及顺序完全一致，通过验证后才进入制品库。
- 未安装、环境锁被修改或探测失败的方法会明确报告 `unavailable`，不会降级成占位实现。

## 实验

```bash
python experiments/external_worker_pipeline.py \
  --source /data/ArtModeling/cabinet2/real/recon/artgsvideo_meshes/whole_mesh.ply \
  --workspace /data/physrealx-workspace \
  --output /data/physrealx-experiment
```

该验证仅覆盖已声明的能力：CoACD 碰撞分解、MuJoCo 关节执行、BlenderProc 同步模态渲染和 LeRobot v2.1 Episode 打包。Ditto 等需要专用输入与权重的方法必须通过独立 worker 和方法描述符另行安装，不得由本实验结果替代。
