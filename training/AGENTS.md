# AGENTS.md — 训练源码

`one-shot-opd/` 保留上游 recipe、data、eval、verl 的目录关系；直接在其中修改，随 OPD-State 一起提交。根目录维护 formatter 跳过导入源码；Python 改动沿用其 `verl/pyproject.toml` 的规则，并运行相关训练测试。新增行为先有对应基线和反例。

调整 rollout 分布、action 分布、teacher reward 或梯度归约时，分别记录其数学语义和指标；保持随机种子、有效 token 预算、teacher/student checkpoint、采样设置可追踪。训练入口与数据说明见 [上游 README](one-shot-opd/README.md)。

源文件清单是导入时的基线，不随实验改动更新以掩盖差异。`scripts/check_upstream.py --pristine` 只验收原始导入；日常模式允许已修改源码并报告差异，仍检查文件缺失、源码语法和嵌套 Git 目录。
