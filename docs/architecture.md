# 架构

OPD-State 将同一个问题的理论分析与训练实现放在一个 Git 仓库中。研究正文引用可复现构造，训练代码保留自己的数据与运行协议，两者通过明确的实验假设、配置和结果关联。

## 目录职责

| 目录                       | 维护内容                             | 修改入口                                                   |
| -------------------------- | ------------------------------------ | ---------------------------------------------------------- |
| `training/one-shot-opd/`   | 可直接修改的完整上游训练源码         | `recipe/` 选配置，`verl/verl/` 改训练行为，`eval/` 改评测  |
| `research/occupancy/`      | 访问测度研究正文、精确实验、结果与图 | `experiments/` 为生成来源，`results/` 与 `figures/` 为产物 |
| `docs/`                    | 架构、开发和来源说明                 | 面向仓库使用者                                             |
| `notes/`                   | 需求与研究工作记录                   | 假设、过程和未决事项                                       |
| `.agents/`                 | 维护技能、共享参考和长期决策         | 根 AGENTS.md 提供入口                                      |
| `scripts/`                 | 检查与两阶段 hooks                   | `checks.json` 为本仓唯一检查集合                           |
| `provenance/`、`licenses/` | 导入版本、文件校验值和许可           | 新的上游同步单独审阅差异                                   |
| `runs/`                    | 本地实验大文件与临时状态             | Git 忽略，结论所需小结果另行入库                           |

## 训练链路与扩展位置

上游 `recipe/opd.sh`、`mopd.sh`、`grpo.sh` 共享 `_common.sh`，后者设置 vendored verl 的 PYTHONPATH，再调用 `verl.trainer.main_ppo`。`DOMAIN` 选择模型与数据，`MODE` 选择 full、oneshot 或 template。应保留这些相对路径关系；从任意目录调用 recipe 都能解析同一份源码。

PPO 训练循环在 `verl/verl/trainer/ppo/ray_trainer.py`；`opd_trainer_mixin.py` 注册并初始化 teacher worker，把 teacher 的逐 token 信号装入训练 batch。`verl/verl/workers/fsdp_teacher_worker.py` 负责教师打分，`core_algos.py` 处理优势与策略损失。后续访问测度实验先在已有链路定位采样、打分、归约三个动作，再按实际行为增加最小配置，避免同时维护第二套训练器。

logit fusion、同轨迹切换和 batch 混合尚未实现。实现时需分别固定 prefix 来源、action 采样分布和是否使用重要性修正；结果同时报告 KL、任务正确率和预算，不能把覆盖率单独当作成功条件。具体假设与判分见[研究正文](../research/occupancy/2026-10-17-OPD访问测度与收敛机制研究.md)。

## 导入与维护边界

训练源码作为普通文件纳入当前仓库，许可证、NOTICE 和原 README 保持原样。来源清单冻结导入基线；日常源码修改允许与它不同，严格导入检查用于核验初始快照。后续同步上游应在独立提交中审阅，保留本地算法改动。

根 formatter 与文档门控覆盖自有内容，导入训练源码使用上游风格和相关产品测试。基础检查会编译所有上游 Python 源码并检查 Shell 语法；这只证明语法成立，训练正确性仍需在独立 GPU 环境中验证。
