# 策略与世界模型训练

训练输入必须是带摘要的不可变数据集；训练方法、推理方法、权重和配置进入模型清单。PyTorch、
CUDA 和具体算法保持在独立适配器环境中，不进入平台核心依赖。

## 选择方法

```python
from physrealx import MethodCapability

for method in px.methods.list(MethodCapability.POLICY_TRAIN):
    print(method.name, method.version, method.runtime)
```

| 任务 | 推荐开源方法 |
| --- | --- |
| 可执行契约/数值回归参考 | `torch_bc` |
| 低成本示范学习 | `lerobot_act`、`robomimic_bc` |
| 多峰、接触丰富动作 | `diffusion_policy` |
| 离散意图与连续动作 | `vqbet` |
| GPU 强化学习 | `rsl_rl_ppo` |
| 特征/状态世界模型 | `dino_wm`、`dreamerv3`、`tdmpc2` |
| 动作条件未来视频 | `irasim` |

未出现在方法目录中的名称不能提交；代码开源但权重使用其他许可证的方法需要单独登记和审核。
`torch_bc` 只检验平台训练契约，不代表 LeRobot/ACT、VLA 或真机策略已训练。

## 提交训练

```python
from physrealx import MethodCapability, TrainingSpec

trainer = px.methods.bind(
    MethodCapability.POLICY_TRAIN,
    "torch_bc",
    config={
        "observation_signals": ["drawer.position", "drawer.velocity"],
        "epochs": 80,
        "hidden_dim": 32,
        "learning_rate": 0.01,
        "device": "cpu",
        "seed": 17,
    },
)

model_ref = px.training.submit(
    TrainingSpec(
        resource_id="drawer.bc",
        version="1.0.0",
        dataset=dataset_ref,
        trainer=trainer,
    )
)
model = px.training.get_model(model_ref)
```

恢复预训练方法时，先把检查点写入制品存储，再传给 `px.methods.bind(..., checkpoint=ref)`；最终
清单不能使用 `latest`、外部可变路径或未校验的下载地址。

## 数据与信号约束

- 训练器只能读取数据集声明的观测信号；
- dtype、shape、单位、frame 和动作边界必须与部署策略一致；
- 图像预处理、时间窗口、动作 horizon、归一化和 tokenizer 都进入方法配置；
- 真实、仿真与世界模型生成 Episode 保留不同来源，不能在导出时抹平；
- 切分按资产、场景和采集批次防泄漏，validation/test 不参与定向再生成。

## 模型清单与验收

`ModelManifest` 保存训练/推理 `MethodSpec`、数据集摘要、权重摘要、信号 Schema、规范化配置和
指标。离线动作误差只用于调试；发布至少需要冻结任务成功率、Wilson 置信区间、推理延迟、碰撞/
超力/越界、未见资产最差分组和旧版本回归结果。

世界模型还需报告多步状态/位姿误差、漂移、接触 F1、规划成功率和反事实动作敏感性；
PSNR/SSIM/LPIPS/FVD 只能作为视觉辅助指标。
