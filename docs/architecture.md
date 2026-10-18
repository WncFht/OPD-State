# 代码结构与修改入口

本仓以 One-Shot-OPD 的根目录作为开发根目录。训练代码直接在原位置修改，hooks 和研究资料是添加到代码库中的辅助内容。

## 训练与维护

| 位置            | 职责与修改方式                                                         |
| --------------- | ---------------------------------------------------------------------- |
| `recipe/`       | 启动训练、选择模型和数据、传递 Hydra 参数；新增实验配方从这里进入      |
| `data/prep/`    | 数据下载与转换；`data/README.md` 定义数据布局，实际数据文件由 Git 忽略 |
| `eval/`         | 评测入口；沿用原有任务协议，指标调整与实现一同提交                     |
| `verl/verl/`    | PPO 训练循环、rollout、教师打分、策略损失与分布式 worker               |
| `verl/scripts/` | 原框架的模型与运行辅助脚本                                             |
| `scripts/`      | 我们接入的维护检查、工具安装与 hooks                                   |

`verl/` 内部的 `setup.py`、`pyproject.toml` 和 `verl/` 包目录是上游原有结构。保留这一层可继续使用原安装入口和 recipe 的 PYTHONPATH 解析；日常改训练代码直接进入 `verl/verl/`。

## 研究与开发资料

所有人工阅读的项目资料统一归 `docs/`。`docs/research/occupancy/` 包含研究正文，以及随正文复现的 `experiments/`、`results/` 和 `figures/`；`docs/notes/` 保存需求、尝试和工作记录。两者是同一文档目录中的不同用途，使用说明通过链接引用结果。

开发说明归 [development.md](development.md)，导入版本与许可归 [provenance.md](provenance.md)。`.agents/` 保留工具要求的技能入口和决策格式。模型、数据、checkpoint、日志与大文件输出按 `.gitignore` 排除；`node_modules/` 只供本地维护工具使用。

## 算法修改链路

`recipe/opd.sh`、`mopd.sh`、`grpo.sh` 通过 `_common.sh` 解析根目录的数据与 vendored verl，再调用 `verl.trainer.main_ppo`。`DOMAIN` 选择模型与数据，`MODE` 选择 full、oneshot 或 template。训练命令从仓库根目录执行，例如 `MODE=oneshot bash -o pipefail recipe/opd.sh`。

PPO 主循环在 `verl/verl/trainer/ppo/ray_trainer.py`，OPD 的教师注册和逐 token 信号处理在 `opd_trainer_mixin.py`，教师打分在 `verl/verl/workers/fsdp_teacher_worker.py`，优势与策略损失在 `verl/verl/trainer/ppo/core_algos.py`。实验直接修改这些源码和对应 recipe，使用普通 Git 提交记录。

`recipe/math_alignment.sh` 提供 JustRL-1.5B 与 R1-Distill-1.5B 双向对齐配方。`opd_topk_tail.py` 在 student head 上精确求和，用带行为概率修正的 sampled tail 补回完整条件 KL 梯度；teacher worker 返回相同候选的概率和补集质量，`workers/utils/losses.py` 对冻结 score 系数求导。该路径仅支持 fresh student、单 batch 单更新和共同有效词表；条件梯度无偏不等于完整序列目标梯度无偏。`eval/measure_student_kl.py` 独立生成 student 轨迹并计算完整词表 KL，数学依据见[采样估计器推导](research/occupancy/2026-10-18-Sampled-token与无偏尾部估计.md)，配置、指标与验收范围见[双向实验运行与验证](research/occupancy/2026-10-18-双向实验运行与验证.md)。

固定 teacher 与初始 student 轨迹由 `data/prep/build_rollout_cache.py` 生成，`fixed_rollout.py` 验证并读取缓存；当前 student 和冻结 teacher 仍在训练时重新评分。`recipe/math_cached.sh` 选择缓存来源，所有设置均只更新 student。长时固定缓存训练由 `plateau.py` 合并每 50 步的验证与训练残差，记录经验平台判断。

共享 prefix 的 logit fusion 已在 `verl/verl/workers/rollout/fusion/` 实现，通过 vLLM worker 扩展接收当前 actor 参数并生成融合轨迹，`recipe/math_fusion.sh` 提供训练入口。验证始终使用独立 student；完整两步调试与 λ=0.5 的正式 50 步实验均已完成。缓存身份、权重版本、后端限制、测试和结果见[统一实验协议](experiment/2026-10-19-rollout来源与logit融合实验.md)。正式训练链路尚未接入轨迹内 teacher/student 切换及 batch 混合；相关假设与判分归[研究正文](research/occupancy/2026-10-17-OPD访问测度与收敛机制研究.md)。

### 当前 fusion 的计算与训练

当前 `name=vllm` 表示服务容器、RPC 和验证设施，fusion 的训练生成实际由 `LogitFusionDecoder` 中两套 Transformers 模型执行。每个 actor rank 对应一个单卡 fusion 副本；生成用 student 是训练 actor 的独立推理副本，另一套是冻结 teacher。actor 更新后的 named tensors 经 BucketedWeightSender/Receiver 原地写入推理 student，完整接收后才发布版本；第 $k$ 步生成要求版本为 $k-1$。同一参数流目前还更新原生 vLLM student，原生引擎用于 student-only 验证。

每个 prompt 只使用一个 token 序列。两个模型分别 prefill 相同 prompt，保存各自 KV；每个解码位置分别得到 logits，在有效词表内转为 FP32 做加权，softmax 后只采样一个 token，再把同一个 token 送给两个模型。正式配方的 λ=0.5、温度为 1、无 top-p/top-k 截断。生成结束后，训练 actor 重新计算 student log-prob，FSDP teacher 重新评分；现有 OPD 信号与优化器只更新训练 student，fusion 不是一个新增的可训练混合模型。

默认 `cuda_graph_flash` 的 prefill 使用 eager 前向，decode 捕获两套模型的前向并使用 Flash KV attention；FP32 logits 融合、softmax 和随机采样仍在图外执行。StaticCache 按 batch 与总长度容量建立，已结束行不会被新请求补入，只以 mask 排除；长短响应混批仍有空槽计算。λ=0/1 虽有正确的融合端点，当前生成循环仍执行两个模型，尚未剪去无用前向。行为 log-prob 在 decoder 中返回，但 worker 当前只汇总为指标，未作为逐 token 字段交给 learner。

重构提案、替代方案、相关推理实现与验证条件统一维护于 [Agent Note：可组合 rollout 与原生 vLLM 推理](../.agents/notes/proposed/architecture/2026-10-18-可组合rollout与原生vLLM推理.md)。
