# AGENTS.md

本仓直接基于 One-Shot-OPD 研究 on-policy distillation（OPD）的 prefix 访问测度与收敛机制。训练、数据与评测沿用根目录的原有结构；研究与开发资料统一归 `docs/`。先读 [README](README.md)、[代码结构](docs/architecture.md)和[开发指南](docs/development.md)。

- `recipe/` 是训练启动入口，`data/prep/` 是数据准备，`eval/` 是评测，`verl/verl/` 是训练实现；直接修改并随本仓提交。`verl/scripts/` 保留原框架脚本，根 `scripts/` 提供维护检查与 hooks。
- 算法变更先建立同模型、数据、预算与判分的基线。修改 rollout 分布、action 分布、teacher reward 或梯度归约时，分别写清数学语义和指标，不能用 CPU 构造替代真实训练实验。
- 研究正文、精确实验、结果与图件归 `docs/research/`；需求与过程记录归 `docs/notes/`。修改来源后重新生成派生物，保留条件与证据确定性。
- 文档使用 [maintain-docs](.agents/skills/maintain-docs/SKILL.md)，文字审阅使用 [maintain-prose](.agents/skills/maintain-prose/SKILL.md)，长期取舍使用 [maintain-decision-notes](.agents/skills/maintain-decision-notes/SKILL.md)。
- 完整检查为 `.venv/bin/python scripts/run_checks.py --config scripts/checks.json --mode ci`；安装两阶段 hooks 使用 `.venv/bin/python scripts/setup.py`。保留训练源码原有风格，不批量格式化；改动训练行为时运行相关产品测试。
- GPU 实验限物理设备 0–3；运行前核验空闲资源与独立环境，不修改共享训练环境。大型输出与本地训练数据保持忽略。行为、文档、配置和测试同改，提交使用英文 Conventional Commits。
