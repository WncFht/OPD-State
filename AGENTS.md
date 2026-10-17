# AGENTS.md

OPD-State 研究 on-policy distillation（OPD）的 prefix 访问测度与收敛机制。先读 [README](README.md)、[架构](docs/architecture.md)和[开发指南](docs/development.md)。

- `training/one-shot-opd/` 是直接修改和提交的训练源码；导入来源归 `provenance/one-shot-opd.json`，遵循 [training 约定](training/AGENTS.md)。算法变更先建立同模型、数据、预算与判分的基线，不能用 CPU 构造替代 LLM 实验。
- `research/occupancy/` 保存研究正文、数学核验与可审计结果。修改来源后重新生成派生物，保留条件与证据确定性。
- 文档归位使用 [maintain-docs](.agents/skills/maintain-docs/SKILL.md)，文字审阅使用 [maintain-prose](.agents/skills/maintain-prose/SKILL.md)，长期取舍使用 [maintain-decision-notes](.agents/skills/maintain-decision-notes/SKILL.md)。过程记录归 `notes/`，不写进使用说明。
- 维护工具在 `scripts/`，依赖由本仓锁定。完整检查为 `.venv/bin/python scripts/run_checks.py --config scripts/checks.json --mode ci`。安装两阶段 hook 使用 `.venv/bin/python scripts/setup.py`，不使用普通 pre-commit install 覆盖入口。
- GPU 实验限物理设备 0–3；运行前核验空闲资源与独立环境，不修改共享训练环境。大型输出归忽略目录 `runs/`；上游数据、checkpoint 与日志也保持忽略。
- 行为变化与文档、配置、测试同改。保留上游许可证，不批量格式化导入基线。提交使用英文 Conventional Commits。
