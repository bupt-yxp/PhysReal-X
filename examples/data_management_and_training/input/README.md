# 编码器权重缓存

Notebook 首次从 Meta 官方固定 URL 下载 DINOv2 ViT-S/14，验证网络键、形状与有限值后转为 safetensors，保存于项目 `.physrealx` 制品库。`dinov2-encoder.json` 仅保存本工作区权重引用，已从 Git 排除；缓存不可解析时重新执行官方下载路径。
