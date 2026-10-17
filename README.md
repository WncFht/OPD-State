# OPD-State

研究 on-policy distillation（OPD）的 prefix 访问测度、采样分布与收敛机制。训练实验直接修改本仓的 One-Shot-OPD 源码，数学推导与精确 CPU 实验独立保存。

| 入口                                                                   | 用途                                                   |
| ---------------------------------------------------------------------- | ------------------------------------------------------ |
| [研究正文](research/occupancy/2026-10-17-OPD访问测度与收敛机制研究.md) | 条件 KL、序列 KL、状态权重、混合 rollout 与可证伪实验  |
| [训练源码与原说明](training/one-shot-opd/README.md)                    | OPD、MOPD、GRPO、one-shot 与 template 的实现和数据准备 |
| [架构](docs/architecture.md)                                           | 研究、训练、结果与维护工具的职责和扩展位置             |
| [开发指南](docs/development.md)                                        | 环境、检查、hooks、CPU 复现与训练验收                  |
| [来源说明](docs/provenance.md)                                         | 上游版本、校验值与许可                                 |

## 开始开发

需要 Python 3.10+、Node 22+ 和 Git。下列安装步骤用于 Linux x86-64；其他平台按开发指南准备系统工具。开发环境只安装维护与 CPU 研究依赖，GPU 训练使用单独环境。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements-dev.lock
npm ci
.venv/bin/python scripts/install_dev_binaries.py
.venv/bin/python scripts/setup.py
.venv/bin/python scripts/run_checks.py --config scripts/checks.json --mode ci
```

CPU 数学核验可单独执行：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python research/occupancy/experiments/exact_checks.py
node research/occupancy/experiments/check_document.cjs
```

训练代码放在 `training/one-shot-opd/`，属于普通 Git 文件，修改后直接在 OPD-State 提交。其 launcher 默认按自身目录查找 vendored verl。模型、数据、训练 Python 和资源配置见[开发指南](docs/development.md#训练环境与验收)。

原有研究提交历史保留；原始研究需求归[需求记录](notes/2026-10-17-原始研究需求.md)，过程归[研究工作记录](notes/2026-10-17-研究工作记录.md)。
