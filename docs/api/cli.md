# 命令行

命令行与 Python 服务一一对应，适合项目初始化、CI 校验、批量提交和运维诊断。所有命令支持 `--output {table,json,yaml}`；预期错误默认不打印回溯。

## 全局形式

```bash
physrealx [--profile PROFILE] [--project PROJECT] [--debug] COMMAND
```

## 命令树

```text
physrealx
├── profile     add / list / show / use / delete
├── auth        login / status / logout
├── project     init / validate / apply
├── doctor
├── validate    asset / scene / task / run / training
├── workers     list / inspect
├── artifacts   upload / download / verify / usage
├── assets      register / import / build / get / list / calibrate / diff
├── scenes      register / compose / get / list
├── tasks       register / compile / get / list
├── runs        submit / get / list / watch / wait / cancel / resume / episodes
├── datasets    get / list / query / split / merge / validate / stats / export / diff
├── training    submit / get / list / watch / wait / cancel / resume
├── models      get / list / compare
├── evaluations submit / get / list / watch / wait / cancel / compare
├── deployments deploy / get / list / watch / stop / episodes
└── feedback    ingest / analyze / plan / execute
```

## 配置档案与身份

```bash
physrealx profile add default \
  --server https://<平台地址> \
  --project embodied-manipulation
physrealx auth login --profile default
physrealx profile use default
physrealx auth status
```

配置档案保存非敏感连接配置和凭据来源；凭据保存在系统密钥环或环境变量中。业务命令默认使用当前配置档案，也可以通过 `--profile` 显式选择。

## 项目

```bash
physrealx project init open-drawer
physrealx project validate .
physrealx project apply .
```

`project apply` 按依赖顺序注册资产、场景、任务和配置；任何冲突会停止，不覆盖不可变版本。

## 资源校验

```bash
physrealx validate asset assets/drawer.asset.yaml
physrealx validate scene scenes/drawer.scene.yaml
physrealx validate task tasks/open_drawer.task.yaml
physrealx validate run runs/generate.run.yaml
physrealx validate training training/dp.training.yaml
```

## 数据生成

```bash
physrealx runs submit runs/generate.run.yaml --wait
physrealx runs watch RUN_ID
physrealx runs cancel RUN_ID --reason "invalid camera calibration"
physrealx runs resume RUN_ID
physrealx runs episodes RUN_ID --success false --output json
```

## 数据集

```bash
physrealx datasets inspect dataset.open-drawer@1.0.0
physrealx datasets validate dataset.open-drawer@1.0.0 --level deep
physrealx datasets stats dataset.open-drawer@1.0.0 --group-by asset_instance
physrealx datasets export dataset.open-drawer@1.0.0 \
  --format lerobot-v3 \
  --destination s3://robot-data/open-drawer-v1
```

## 训练与评测

```bash
physrealx training submit training/dp.training.yaml --wait
physrealx evaluations submit evaluation/heldout.evaluation.yaml --wait
physrealx evaluations compare REPORT_A REPORT_B --baseline REPORT_A
```

## 部署与反馈

```bash
physrealx deployments deploy deployment/franka.shadow.yaml
physrealx deployments watch DEPLOYMENT_ID
physrealx deployments stop DEPLOYMENT_ID --reason "evaluation complete"
physrealx feedback analyze --deployment DEPLOYMENT_ID --failures-only
physrealx feedback plan ANALYSIS_ID --budget episodes=20000,gpu_hours=100
```

## 输出与脚本

```bash
run_id=$(physrealx runs submit run.yaml --output json | jq -r '.run_id')
physrealx runs wait "$run_id" --output json
```

结构化输出字段与 Python 模型一致。人类表格可能调整列宽和颜色，不应被脚本解析。

## 退出码

| 退出码 | 含义 |
| --- | --- |
| `0` | 成功 |
| `2` | 参数或规格校验错误 |
| `3` | 资源不存在 |
| `4` | 不可变版本或幂等冲突 |
| `5` | 能力或兼容性错误 |
| `6` | 任务失败终态 |
| `7` | 连接、超时或工作节点不可用 |
| `8` | 制品完整性错误 |
| `130` | 用户中断 |
