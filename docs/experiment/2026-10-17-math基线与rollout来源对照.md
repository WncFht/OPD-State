# Math 基线与 rollout 来源对照实验

## 目标与证据边界

先验证本仓的 math on-policy distillation（OPD，同策略蒸馏）能在物理 GPU 0–3 完成 50 次参数更新，训练结束后仅评测一次当前内存权重，不保存 checkpoint。W&B project 固定为 `OPD-State`。这一步验证训练链路、记录与评测是否成立；50 步结果只代表有限预算表现。

随后比较三种轨迹来源：当前学生在线生成、固定教师生成的缓存、初始学生生成的缓存。第三种即 Semi-OPD。这里的 teacher rollout 也采用固定缓存，以便对应论文的来源对照；每步重新调用教师生成属于另一项实验。

需要分别回答两个问题：相同优化步数下的末步准确率是否不同；扩大训练预算后是否出现不同平台。前者可以从 50 步对照开始，后者需要多个预算与随机种子，并检查末段趋势。单条 50 步曲线不能证明渐近收敛点不同。

成功条件是训练确实消费预定来源的轨迹，当前学生的 log-probability 每步重新计算，教师固定，指标可核对，末步评测读取更新后的学生。若缓存组仍在线生成、将初始学生 log-probability 当成当前优势、实际更新次数不符、W&B 未上传，或中途出现 validation/checkpoint，则该运行不满足对应协议。

## 论文与现有研究的关系

Zhao 等的 _When Do We Need On-Policy Distillation?_[^semi]将初始学生轨迹固定，预计算教师在这些轨迹上的 log-probability，在训练中持续重算当前学生 log-probability。教师轨迹消融用教师生成的固定缓存替换初始学生缓存，其他设置保持一致。这与单条轨迹内 student→teacher 切换不同。

论文的来源对照同时改变 prefix、被采样 action、轨迹长度与推理内容。即使固定 prompt、模型与步数，末步准确率差异也不能直接归因于 prefix 访问测度。现有[访问测度研究](../research/occupancy/2026-10-17-OPD访问测度与收敛机制研究.md)已给出这个限制，并区分条件 KL、序列 KL 与任务正确率。

论文在去重后的 14,116 个 DAPO prompt 上训练，每步 128 个 prompt、每题一条轨迹；温度与 top-p 均为 1，学习率 $10^{-6}$。非思考配置主要使用 4,096 的训练响应上限，思考配置使用 16,384，评估通常允许 32K，指标为 AIME24/25/26 与 HMMT Feb26 的 avg@32。论文基准表还平均了第 60、80、100 步检查点；汇总表与不同图件存在 cap 和基准范围差异，应按目标实验逐项核对。

本地 parquet 的格式化 prompt 去重后是 17,255 条，仍不同于论文的 14,116 条。严格复现需要继续核对数据 revision、英语筛选及预处理规则，不能仅凭“DAPO”名称认为数据相同。

本地已有的 DeepSeek-R1-Distill-Qwen-1.5B 与 JustRL-DeepSeek-1.5B 属于思考模型配对。第一阶段使用它们验证可运行性，不声称复现论文低 overlap 配对上的准确率平台。若要验证论文 handoff 图的原始现象，需另行准备该图对应的 Qwen3 配对，并统一思考模式和 rollout cap。

## 50 步运行配置

入口为 [math_baseline.sh](../../recipe/math_baseline.sh)，沿用 [opd.sh](../../recipe/opd.sh) 与 [_common.sh](../../recipe/_common.sh)。下表是第一阶段的显式配置，不代表论文复现设置。

| 项目                    | 50 步运行设置                                                                    | 选择理由                                            |
| ----------------------- | -------------------------------------------------------------------------------- | --------------------------------------------------- |
| GPU                     | 物理 0、1、2、3，四张 H800 80GB                                                  | 用户指定；不使用其他卡                              |
| Student                 | DeepSeek-R1-Distill-Qwen-1.5B                                                    | 本地已有的原始权重                                  |
| Teacher                 | JustRL-DeepSeek-1.5B，冻结                                                       | 本地已有的同规模数学教师                            |
| 训练数据                | `MD-OPD/data/math/DAPO/train.parquet`，17,917 行，格式化 prompt 去重后 17,255 条 | 先复用现有数据；尚未按论文去重                      |
| 验证数据                | MATH-500，500 题                                                                 | 首先验证标准 math 评测链路                          |
| 更新数                  | 50                                                                               | 用户指定                                            |
| Prompt batch / 每题轨迹 | 64 / 1                                                                           | 沿用现有配方的 batch 规模                           |
| Prompt / 训练响应 cap   | 1,024 / 4,096 tokens                                                             | 控制首轮运行成本；需报告过滤与截断比例              |
| 训练采样                | temperature 1、top-p 1                                                           | sampled-token 基线                                  |
| 优化器                  | AdamW，学习率 $10^{-6}$，单 PPO epoch                                            | 复用现有优化路径                                    |
| 训练信号                | `token_reward_direct`，`LOG_PROB_TOP_K=0`                                        | 教师—学生 sampled-token log-ratio，不混合正确率奖励 |
| KL 附加损失             | 关闭                                                                             | 避免引入另一项 reference 正则                       |
| 全局归约                | 有效 response token 的 token-mean                                                | 保持与现有配方一致                                  |
| 验证协议                | 最后一步 MATH-500 greedy@1，响应 cap 8,192                                       | 单次末步评测；与论文 avg@32 分开报告                |
| 中途验证                | 关闭：`val_before_train=False`，`test_freq=50`                                   | `test_freq=-1` 会连最终验证也关闭                   |
| 保存                    | `save_freq=-1`，`resume_mode=disable`                                            | 直接评测内存权重，不写 checkpoint                   |
| 记录                    | W&B online，project `OPD-State`，另留本地日志                                    | 训练指标与运行证据可核对                            |
| 数据顺序                | `data.seed=42`                                                                   | 固定首轮 prompt 顺序                                |

4K 是调试基线的预算选择；思考模型在该 cap 下的截断可能限制准确率。末步结果必须和长度、截断率一起阅读。后续三组对照首先共用同一 cap，再单独研究 8K/16K，不能仅提高某一组的长度。

### 运行与验收

训练 Python 默认是工作区独立环境 `envs/opd-state/bin/python`。可通过 `OPD_TRAIN_PYTHON` 指定完整、独立的训练环境；不修改原有共享环境。当前 CUDA 12.6 适配复用了相邻 MD-OPD 的 `cef14fa` 补丁，涉及 vLLM 版本门槛、参数解析器导入位置和 GPT-OSS 的延迟导入；适配不改变 OPD 更新式。旧版 vLLM 的后台服务显式启用 `VLLM_USE_V1=1`，并按实际接口能力设置多模态缓存与 CUDA Graph 默认项。该开关选择 vLLM 推理引擎；`trainer.use_v1=False` 则保留含 OPD mixin 的 veRL 训练路径，两者是独立开关。四卡两步更新与末步评测已通过，[机器记录](results/2026-10-17-math-smoke.json)保存 loss、梯度、环境和数据哈希。调试响应 cap 为 256，不将该准确率当作正式 math 基准。

```bash
bash recipe/math_baseline.sh
```

末步更新后的权重先同步到 rollout engine，再执行 `_validate()`。checkpoint 保存位于 `save_freq > 0` 条件内，因此关闭保存不影响内存权重评测。运行目录位于 `runs/math-baseline/<experiment-name>/`；日志与 W&B 保存模型版本、数据版本、参数及环境证据，末步逐题输出与分数另写入运行目录的 `validation/`，用于核对准确率；凭据由已有 netrc 读取，不写入仓库。

调试先用 1–2 步、短响应、小 batch 和四条验证样本确认完整路径。该阶段的准确率不用于和正式 50 步结果比较。正式验收检查：step 1–50 的更新指标完整，loss、梯度与奖励有限，learning rate 有效，更新后的 rollout 被使用；validation 只出现在末步；没有 checkpoint 文件；W&B run 的 project、配置和步数与本地日志一致。

## 三种算法的统一定义

令 $p_0$ 为初始学生，$p_{\theta_k}$ 为第 $k$ 次更新前的学生，$q$ 为冻结教师，$x_i$ 为按固定顺序读取的 prompt。轨迹 $y_i$ 的第 $t$ 个有效 response token 为 $a_{it}$，prefix 为 $s_{it}=(x_i,y_{i,<t})$。EOS token 包含在有效集合中，padding 不参与损失；达到响应 cap 的轨迹记录截断标志。

对一个 batch，先固定当前学生锚点 $\bar\theta=\theta_k$，计算

$$
\ell^S_{it}=\log p_{\bar\theta}(a_{it}\mid s_{it}),\qquad
\ell^T_{it}=\log q(a_{it}\mid s_{it}),\qquad
A_{it}=\operatorname{stopgrad}(\ell^T_{it}-\ell^S_{it}).
$$

论文的 sampled-token 更新可写为

$$
\mathcal L_k(\theta)=-\frac{1}{N_k}\sum_{i,t}m_{it}A_{it}\log p_\theta(a_{it}\mid s_{it}),\qquad
N_k=\sum_{i,t}m_{it}.
$$

当前上游实际调用 PPO clipped surrogate：其比率分母为本步重算的 $p_{\bar\theta}$，另含 clipping 与 dual-clip。首版固定一次完整 optimizer update、一个 PPO epoch、全局 mini-batch 等于 train batch，micro-batch 只做梯度累积；在更新起点比率为 1，对应上述局部 sampled-token 梯度。若增加 PPO epoch 或拆成多个 optimizer mini-batch，必须重新核验与裸 surrogate 的差异，不能仍假定相同。

### 当前学生在线生成

每步从 $p_{\theta_k}$ 采样全新轨迹，重新计算该 batch 的学生与教师 log-probability，用上述信号更新一次学生，再把新权重同步到生成端。下一步使用新的学生采样。

```pseudocode
\begin{algorithm}
\caption{当前学生在线轨迹蒸馏\\{\small\textit{Semantics: 根据正文重构}}}
\begin{algorithmic}
\INPUT initial student $p_0$; frozen teacher $q$; prompt batches $X_1,\ldots,X_K$; response cap $H$
\OUTPUT final student $p_{\theta_K}$
\STATE $\theta_0\gets\theta(p_0)$
\FOR{$k=1$ \TO $K$}
    \STATE $\bar\theta\gets\theta_{k-1}$; synchronize rollout weights to $\bar\theta$
    \STATE independently sample one $y_i\sim p_{\bar\theta}(\cdot\mid x_i)$ for each $x_i\in X_k$, stopping at EOS or $H$
    \STATE construct valid-token masks $m_{it}$, including EOS and excluding padding
    \IF{$\sum_{i,t}m_{it}>0$}
        \STATE compute current $\ell^S_{it}$ and frozen-teacher $\ell^T_{it}$
        \STATE $A_{it}\gets\operatorname{stopgrad}(\ell^T_{it}-\ell^S_{it})$
        \STATE perform one complete optimizer update on $\mathcal L_k$, producing $\theta_k$
    \ELSE
        \STATE $\theta_k\gets\theta_{k-1}$; record skipped update
    \ENDIF
\ENDFOR
\RETURN $p_{\theta_K}$
\end{algorithmic}
\end{algorithm}
```

在线组每步改变采样策略。若出现空有效 token batch，该次不是有效更新；需要显式报告，不能在“50 次更新”的计数中静默抵消。

### 固定教师轨迹缓存

训练前在所选 prompt 集上每题独立采样一次 $y_i\sim q(\cdot\mid x_i)$，固定轨迹，预计算教师在每个有效 token 上的 log-probability。训练学生从同一 $p_0$ 开始，每步读取 prompt 对应的固定教师响应，重算当前学生 log-probability 和 $A$，更新学生。训练阶段不再调用教师生成，也不做教师轨迹上的 SFT 热启动。

### 初始学生轨迹缓存（Semi-OPD）

训练前在相同 prompt 集上每题独立采样一次 $y_i\sim p_0(\cdot\mid x_i)$。教师在这些固定轨迹上打分并缓存。训练从同一 $p_0$ 开始，每步重算当前学生 log-probability 和 $A$；响应、mask 和教师打分保持固定。

两个缓存组可用同一个训练过程，只改变预生成轨迹的模型：

```pseudocode
\begin{algorithm}
\caption{固定来源轨迹蒸馏\\{\small\textit{Semantics: 根据正文重构}}}
\begin{algorithmic}
\INPUT initial student $p_0$; frozen teacher $q$; source $\mu\in\{p_0,q\}$; ordered prompt batches $X_1,\ldots,X_K$; response cap $H$
\OUTPUT final student $p_{\theta_K}$
\STATE sample one response per unique prompt from frozen $\mu$, stopping at EOS or $H$
\STATE cache token IDs, masks, truncation flags and $\ell^T_{it}=\log q(a_{it}\mid s_{it})$
\STATE $\theta_0\gets\theta(p_0)$
\FOR{$k=1$ \TO $K$}
    \STATE load cached trajectories for $X_k$ in the same prompt order as the online arm
    \STATE $\bar\theta\gets\theta_{k-1}$; recompute $\ell^S_{it}=\log p_{\bar\theta}(a_{it}\mid s_{it})$
    \IF{$\sum_{i,t}m_{it}>0$}
        \STATE $A_{it}\gets\operatorname{stopgrad}(\ell^T_{it}-\ell^S_{it})$
        \STATE perform one complete optimizer update on $\mathcal L_k$, producing $\theta_k$
    \ELSE
        \STATE $\theta_k\gets\theta_{k-1}$; record skipped update
    \ENDIF
\ENDFOR
\RETURN $p_{\theta_K}$
\end{algorithmic}
\end{algorithm}
```

缓存组与在线组的 action 分布不同。首版忠实保留论文“不做行为重要性修正”的语义，同时记录该限制。缓存中的 $\log\mu$ 可供诊断，但不能替代本步的 `old_log_probs`：PPO 内部锚点应是当前学生，不能把比率变成 $p_\theta/p_0$ 或 $p_\theta/q$。

### 来源变化同时改变优化测度

对固定缓存诱导的 token 测度 $\nu_\mu(s,a)$，在本步更新起点，未作行为重要性修正的梯度为

$$
g_\mu(\theta)=\mathbb E_{(s,a)\sim\nu_\mu}[(\log p_\theta(a\mid s)-\log q(a\mid s))\nabla_\theta\log p_\theta(a\mid s)].
$$

记 $r_\theta(s,a)=\log p_\theta(a\mid s)-\log q(a\mid s)$。缓存是有限集合，其权重与 $\theta$ 无关，冻结教师满足 $\nabla_\theta\log q=0$。逐项求导得到

$$
\begin{aligned}
F_\mu(\theta)&=\frac12\sum_{s,a}\nu_\mu(s,a)r_\theta(s,a)^2,\\
\nabla_\theta F_\mu(\theta)
&=\frac12\sum_{s,a}\nu_\mu(s,a)\,2r_\theta(s,a)\nabla_\theta r_\theta(s,a)\\
&=\sum_{s,a}\nu_\mu(s,a)r_\theta(s,a)\bigl(\nabla_\theta\log p_\theta(a\mid s)-0\bigr)\\
&=g_\mu(\theta).
\end{aligned}
$$

该等式要求参与取对数的概率为正，缓存权重固定，且采用同一 token 归约；它描述精确期望的局部梯度，不把多步 PPO clipping 当成相同目标。两个缓存组因此同时改变 prefix 权重和 action 权重。在线组的 action 随当前学生变化，不能把上述固定测度求导直接套过去。

首轮三组对照检验整体训练处理的差异。若要独立解释访问测度的作用，下一阶段应在这些 prefix 上统一使用当前 student action，或计算同一条件 KL；具体构造见研究正文第 8.5 节。

## 对照设置提案

实现缓存路径后，先沿用已跑通的 JustRL 配对做短工程对照，确认三组都读对轨迹、更新一次并用最终学生评测。当前先执行在线 50 步 baseline；缓存生成与其余训练待这里的设置确定后执行。

正式检验来源差异，建议优先使用论文的 Qwen3-1.7B（non-thinking）← Qwen3-4B-Instruct-2507 配对。论文对该配对报告初始 overlap 62.5%；对当前同名 JustRL 配对报告 79.7%，并将后者列为 rollout 长度与正信号比例趋势的例外。上述百分比是论文报告，当前本地 50 步没有测量 overlap。因此，JustRL 适合作为已跑通的工程起点与较高 overlap 对照，不宜单独承担“低 overlap 下三组来源会不同”的主检验。

| 阶段           | 配对与预算提案                                                                                | 用途                                           |
| -------------- | --------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| 工程验收       | 当前 1.5B/JustRL 配对，三组短运行，1 个 seed                                                  | 验证缓存语义和梯度，不比较平台                 |
| 来源差异主实验 | Qwen3-1.7B non-thinking ← Qwen3-4B-Instruct-2507；4K cap；先 batch 64；50 步；seed 42、43、44 | 先检验有限预算末步差异，三组共用 prompt 与配置 |
| 配对稳健性     | JustRL 配对使用共同设置重复三组                                                               | 判断效应是否依赖模型配对                       |
| 论文协议复现   | 另行对齐 batch 128、数据过滤、各配对 cap 和 AIME/HMMT avg@32                                  | 与论文数字比较                                 |

Qwen3 配对需要准备指定 revision，并显式设置 non-thinking；缓存前验证 chat template、prompt token ID 与 tokenizer 支持集。不能把默认 thinking 开关或 teacher 自带模板的差异混入来源对照。之后可在不改变 sampled-token 训练信号的前提下单独测量初始 top-64 overlap；该观测与 `LOG_PROB_TOP_K` 的训练目标开关分开。

三组使用同一 prompt 清单和 batch 顺序，每条 prompt 在两个缓存组各有一条固定响应，在线组每次按当前学生重新采样。

若目的是让固定缓存被重复吸收，单个 epoch 不足以检验“重复使用相同状态”。建议固定一个可审计 prompt 子集，例如 1,024 条，batch 64 则每 16 步遍历一次；50 步会复用三轮以上。该设置与全量 17,917 条数据的 50 步在线调试不相同，应另起共同对照，不能直接拿首个 baseline 与子集缓存组比较。

若目的是靠近论文，则统一去重后全量 prompt、batch 128、思考模型 16K cap，以及论文对应的评估协议。上游仓库当前的数据和第一阶段协议并不自动满足这些条件，需单独准备去重清单、基准与预算。

| 控制项 | 首轮三组共同要求                                          |
| ------ | --------------------------------------------------------- |
| 初始化 | 相同 student 权重与全新 optimizer 状态                    |
| 教师   | 相同冻结 checkpoint、tokenizer 与打分温度                 |
| Prompt | 相同 ID 集、相同顺序、相同过滤与 chat template            |
| 生成   | temperature 1、top-p 1、相同响应 cap、相同 EOS 规则       |
| 更新   | 相同步数、batch、学习率、PPO epoch、归约与梯度裁剪        |
| 验证   | 只生成 student 响应；相同题目、采样和 cap                 |
| 保存   | 三组均不保存模型 checkpoint；固定缓存作为数据单独保存     |
| 预算   | 报告有效训练 token 和生成/打分时间；等步数并不等 token 数 |

只有末步评测时，平台证据可通过各自独立的 50、100、200、400 步运行收集，每个运行仍只在最后评测一次。这会重复计算，但不需要中途验证或落盘 checkpoint。训练预算先按短运行的实际时间估算；若末段准确率仍变动，应扩大预算并保留“不足以判断平台”的结论。

任务准确率报告每题得分及均值，配对比较相同题目的得分差。greedy@1 可报告题目层面的配对 bootstrap 区间，并单独报告训练 seed 的离散性。若改用 avg@N，先在题内平均 N 次，再在题目间平均；不能把 N 次采样当成独立训练重复。当前 MATH-500 结果不能与论文 AIME/HMMT 的均值直接对比。

### 判分与否决条件提案

首轮按配对 seed 比较 teacher-cache、initial-student-cache 与 online-student 的末步准确率差，三个比较同时报告效应大小与逐题得分差。暂将 2 个百分点作为值得继续追踪的实际差异阈值；该阈值用于参数讨论，不替代区间估计。只有一个 seed 或区间跨零时，保留为探索性结果。若对三组两两差异作显著性判断，需处理三次比较，例如 Holm 校正。

若差异小于阈值且区间仍很宽，只能说证据不足。要支持“在该预算下实际等价”，区间需整体落在预先指定的等价范围内。若扩大训练预算后差异持续缩小，则否决“已经观察到不同平台”的表述；若增加长度或统一有效 token 预算后排序改变，则优先解释长度或预算的作用。

平台候选至少要求两个更长预算的末步结果趋稳、跨 seed 的排序一致，并报告训练目标和截断率的末段状态。即使准确率趋稳，也只称该协议下的有限预算平台，不据此宣称学生策略或访问测度相同。

## 代码支持方案

现有代码已经支持第一组。`ray_trainer.py` 在每步调用 `async_rollout_manager.generate_sequences()`，重算 `old_log_probs` 后，`OPDTrainerMixin.compute_opd_token_reward()` 调用 FSDP teacher，构造逐 token 信号，再由 `token_reward_direct` 与策略损失更新学生。验证也复用当前学生的 rollout engine。

`MODE=oneshot` 只切换到单 query 数据集，仍然每步用当前学生生成响应，并不提供 Semi-OPD 缓存语义。`ACTOR_MODEL_PATH` 同时决定被训练的学生和在线生成模型；teacher-cache 组应在预生成入口选择教师作为 source，训练阶段仍加载原始学生，不能通过替换 actor 路径实现。

现有 `AgentLoopManager.generate_sequences()` 带 `SkipManager` 装饰器，`verl/utils/skip/rollout_skip.py` 已能序列化和加载 DataProto，可复用其接入点与存储方式。但它目前按 global step 选择缓存；`repeat` 可以加载最近的一整批结果，缺少缓存时会调用当前生成器补齐。直接启用会有三项问题：以前在线训练逐步留下的快照不是统一初始学生生成；跨 batch 重放可能和当前 prompt、ground truth 不匹配；cache miss 会引入当前学生的新轨迹。其缓存也位于教师评分之前，不包含后续计算的 OPD teacher log-probability。

因此，首版优先扩展现有缓存入口：固定来源预生成，使用 prompt ID/输入哈希校验，固定缓存组缺失即报错，保留已有 validation 绕过缓存的机制。无需建立第二套训练器。

实现可分两步。最小版本只缓存轨迹，继续调用现有冻结 teacher worker 实时评分，保持当前学生打分、优势和损失路径原样。因为 q 固定且处于 eval 模式，在同一 token/prefix 上重算教师分数与预先缓存对应于同一数学目标；两者的数值差异需要实测核对。该版本已能比较三种轨迹来源，但不具备论文避免重复教师前向的效率收益。随后在首步梯度和评分一致性通过后，再加入 teacher log-probability 缓存并移除重复打分。建议改动如下：

| 修改位置                                    | 要实现的行为                                                                                                                           |
| ------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `recipe/` 与配置                            | 显式选择 `rollout_source=online_student/cached_teacher/cached_initial_student`，缓存路径与版本必填                                     |
| `data/prep/` 下独立缓存生成器               | 冻结来源模型，按统一 prompt 清单采样一次，保存 token ID、mask、EOS/截断、生成设置与来源哈希                                            |
| 教师评分入口                                | 效率版本对缓存轨迹仅评分一次，保存 teacher log-probability；最小版本继续复用现有实时评分                                               |
| `SkipManager` / `RolloutSkip` 与 batch 构造 | 复用生成钩子读取缓存，按 prompt ID/输入哈希匹配，缺失即报错；复用 padding 工具生成同一 DataProto，保持 mask、position ID 与 batch 顺序 |
| `_compute_old_log_prob()`                   | 三组都按本步当前学生重算，缓存初始学生概率只作诊断字段                                                                                 |
| `compute_opd_token_reward()`                | 最小版本保持原路径；效率版本从验证过的缓存读取 teacher log-probability，再计算当前 log-ratio                                           |
| 权重同步与验证                              | 缓存训练不需要每步同步到生成端，末步评测前必须同步最终 student；验证始终使用 student                                                   |
| 日志与审计                                  | 标记来源、缓存哈希、采样版本、prompt ID 顺序、更新次数、训练 token 数与截断率                                                          |

持久缓存使用数据 revision 与原始行号或规范化输入哈希生成稳定 `prompt_id`；当前训练器每步创建的随机 `uid` 仅用于运行时分组，不能拿来索引跨步缓存。加载时保留当前 batch 的 `uid` 和 ground truth，按稳定 ID 匹配响应，避免把旧 batch 的元数据一同重放。

缓存元数据至少包含 student/teacher revision、tokenizer 与 chat-template 哈希、数据 ID 清单哈希、source、seed、temperature、top-p、cap、EOS 规则、评分 dtype/温度。教师 log-probability 与 token/mask 绑定；任何来源变化都应拒绝读取旧缓存。只缓存 teacher logits 中被采样 token 的概率，不需要保存整个词表张量。

缓存阶段可以用四卡并行生成；效率版本另做一次教师打分，训练阶段无需重复调用 teacher worker。最小版本保留现有 teacher worker，每步在固定轨迹上评分。两版都在最终评测前同步生成服务；运行计时区分一次性准备成本与重复训练成本，准确率对照先保证相同更新语义。

### 实现验收

1. 对固定合成 batch，缓存 teacher log-probability 与实时打分一致；mask、EOS、position ID 对齐。
2. 在线生成与缓存读取相同轨迹时，学生 log-probability、优势和首步梯度一致。
3. 修改当前学生后，缓存 token 和教师分数不变，学生分数及优势变化。
4. 缓存组训练阶段不调用生成器；效率版本还需证明不重复调用教师评分。在线组每步确实生成新轨迹并同步版本。
5. 缓存来源、tokenizer、模型或数据清单不匹配时明确失败；验证数据不进入训练缓存。
6. 三组都只在末步评测当前学生，且关闭 checkpoint 保存；空有效集合与训练失败非零退出或明确记为无效更新。

在这些行为通过后，再做小型 GPU 两步比较，最后运行三组 50 步。状态作用的独立识别另增加“不同来源 prefix、统一 student action 或完整条件 KL”的对照，避免把 action 采样偏差解释成状态测度效应。

### 参考文献

[^semi]: Zhao 等。When Do We Need On-Policy Distillation? arXiv 2610.11291, 2026. [arxiv.org](https://arxiv.org/abs/2610.11291). 公式与设置按工作区提供的论文源码核对，重点为 sections/02_method.tex、05_why.tex、06_more.tex、09_appendix.tex。
