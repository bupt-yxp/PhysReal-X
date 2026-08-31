# 场景与任务生成

场景需求、确定性指令和方法参数均直接写在 API 调用中。候选
`SceneSpec`、`TaskSpec` 经过审核并注册后，再绑定到后续单元格：

```console
python -m jupyter lab examples/scene_and_task_generation/demo.ipynb
```

Notebook 分别演示场景草案、约束组合、任务编译、任务验证和轨迹生成。生成式方法
只提出草案，不能绕过对象绑定、可满足性、碰撞与安全门禁。
