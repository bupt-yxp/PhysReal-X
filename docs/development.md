# 开发与测试

## 安装开发依赖

```bash
python -m pip install --editable '.[dev,train,docs]'
```

核心开发不需要 Torch，可以省略 `train`。

## 质量门禁

```bash
python -m ruff format --check .
python -m ruff check .
python -m mypy
python -m pytest -m 'not gpu' --cov=physrealx --cov-branch
python -m build
python -m twine check dist/*
```

CUDA 冒烟测试单独运行：

```bash
python -m pytest -m gpu tests/integration/test_cuda.py
```

它只执行一个很小的矩阵乘法，用于确认 PyTorch CUDA 12.1 运行时可用。

## 文档开发

```bash
mkdocs serve
```

本地地址会根据 `site_url` 带上项目路径。提交前执行严格构建：

```bash
mkdocs build --strict
```

文档源码位于 `docs/`，导航、主题和站点元数据位于 `mkdocs.yml`。生成目录 `site/` 不进入 Git。

## 自动发布

`.github/workflows/docs.yml` 在以下情况下运行：

- 拉取请求修改文档相关文件时，执行严格构建；
- `main` 分支修改文档相关文件时，构建并上传 Pages 制品；
- 手动触发时，重新构建并部署。

发布使用 GitHub Pages 官方制品部署流程，不维护手工 `gh-pages` 分支。

## 修改原则

- 领域层不得导入存储、Torch、CUDA、命令行或仿真器；
- 新插件必须先有真实消费者和共享契约测试；
- 失败路径不得发布部分数据集或模型；
- 公共资源必须保留版本、摘要、种子和插件身份；
- 文档只能把已通过真实运行时验证的能力标记为支持。
