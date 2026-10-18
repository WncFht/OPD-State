# Rollout 来源与 logit 融合实验

## 进度与主要结果

这是一组统一的 math 蒸馏实验：比较当前 student、固定 teacher、初始 student 缓存三种轨迹来源，再加入共享 prefix 的 logit fusion。四个设置均已完成 50 次 student 更新和最终三基准 eval16，日志、逐题输出与 W&B 指标核对一致。训练、缓存生成和结果处理脚本均随仓保存；正式运行只使用物理 GPU 0–3，不保存模型 checkpoint。

在线 student、teacher 轨迹、Semi-OPD、fusion 的 macro mean@16 分别为 **64.84%、62.52%、64.59%、64.16%**。目前没有证据表明 λ=0.5 fusion 稳定提高平均正确率。四组都只有一个验证预算点，不能据此判断收敛。长时 Semi-OPD 正在运行，同一缓存最多重复训练 800 步，每 50 步执行完整三基准 eval16；第 50 步评测仍在生成，尚无长时准确率曲线。

用户指定的两个独立评测 run 分别是 `math_teacher`（u87buult）与 `math_student`（6uorrk2k）。[^reference-teacher][^reference-student] API 已核对模型名称、finished 状态、val_only、三个基准、每题 16 个响应及采样和长度配置；它们提供 teacher 与初始 student 的准确率参考。远端原始响应、数据和权重哈希尚未核对，因此图中将其标为外部参考，不视为本地同一次训练的 step-0 测量。筛选后的配置和指标保存在[参考数据](results/reference-accuracy.json)。

## 50 步准确率与 pass@k

下表单位为百分比。前两行是独立模型评测，其余四行均是经过 50 步训练后的 student。macro 对三个基准等权；micro 对全部 570 个题目等权。

| 模型或训练设置               | MATH-500 mean@16 | AMC23 mean@16 | AIME25 mean@16 | macro mean@16 | micro mean@16 |
| ---------------------------- | ---------------: | ------------: | -------------: | ------------: | ------------: |
| 初始 student，外部参考       |            83.25 |         72.03 |          23.75 |         59.68 |         79.33 |
| Teacher，外部参考            |            89.11 |         89.38 |          39.17 |         72.55 |         86.50 |
| 在线 student rollout         |            86.24 |         80.16 |          28.12 |         64.84 |         82.75 |
| Teacher rollout 后的 student |            85.38 |         75.31 |          26.88 |         62.52 |         81.59 |
| Semi-OPD 后的 student        |            86.70 |         78.75 |          28.33 |         64.59 |         83.07 |
| Fusion λ=0.5 后的 student    |            86.28 |         78.91 |          27.29 |         64.16 |         82.65 |

![四个训练设置的末步准确率与 teacher、初始 student 外部参考](figures/accuracy-step50.png)

准确率图只画实际完成的 step-50 结果，没有连接训练前后两点来伪造中间验证曲线。所有训练后 student 的平均正确率高于给定初始 student 参考、低于 teacher 参考；严格的训练增益估计仍需在本地核对同一模型资产和数据，并补充训练 seed。

| 设置         | MATH-500 pass@16 | AMC23 pass@16 | AIME25 pass@16 | macro pass@16 | micro pass@16 |
| ------------ | ---------------: | ------------: | -------------: | ------------: | ------------: |
| 在线 student |            95.60 |         95.00 |          50.00 |         80.20 |         93.16 |
| Teacher 轨迹 |            96.20 |         95.00 |          53.33 |         81.51 |         93.86 |
| Semi-OPD     |            96.40 |         95.00 |          46.67 |         79.36 |         93.68 |
| Fusion λ=0.5 |            95.80 |         95.00 |          53.33 |         81.38 |         93.51 |

![相同 16 个响应上的标准 pass@k](figures/pass-at-k.png)

Fusion 的 macro mean@16 比在线组低 0.682 个百分点，macro pass@16 高 1.178 个百分点；AIME25 至少一次正确的题数分别为 16/30 与 15/30。平均正确率与多次采样覆盖表达不同性质，不能只选较高的一项宣布整体更好。

四设置共有的 570 个题目以基准名、输入、标准答案构成的内容哈希配对。以下区间以在线组为参考，按基准分层成对 bootstrap 题目，5,000 次、seed 42，单位为百分点。

| 设置减在线 student | macro mean@16 差值 | 95% 题目 bootstrap 区间 |
| ------------------ | -----------------: | ----------------------: |
| Teacher 轨迹       |             -2.319 |        [-4.055, -0.755] |
| Semi-OPD           |             -0.245 |         [-1.829, 1.310] |
| Fusion λ=0.5       |             -0.682 |         [-2.149, 0.681] |

这些区间表示对题目选择的敏感性，不覆盖训练 seed 方差，也未作多重比较校正。teacher 组在当前单 seed 运行的平均正确率较低；Semi 与 fusion 相对在线组的区间都包含零。它们不能证明算法普遍排序或不同渐近平台。完整逐基准差值、未舍入指标和来源哈希见[四设置比较](results/2026-10-18-four-setting-comparison/analysis.json)。

## 训练指标与图的含义

![KL 代理、逐 token 奖励、log-ratio RMS、梯度范数、熵和 actor loss](figures/training-optimization.png)

每条线是一个真实训练 run 的 50 个原始步骤，没有平滑，也没有将 batch 波动画成训练 seed 置信带。颜色在所有图中固定对应同一轨迹来源。绘图数据、输入文件哈希、调色板、脚本版本和机器审计保存在 `figures/`。

对有效 token，记当前 student 与冻结 teacher 的对数概率差为

$$
d_{it}=\log p_{\theta_k}(a_{it}\mid s_{it})-\log q(a_{it}\mid s_{it}),\qquad
r_{it}=-d_{it}.
$$

图中的指标采用以下语义。

| 图中指标                    | 日志字段或计算           | 含义与解释限制                                                                                                           |
| --------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Sampled log-ratio，KL proxy | `-opd/token_reward_mean` | 实际来源轨迹上的 $\operatorname{mean}(d)$；缓存和融合的 action 分布不是当前 student，均值可为负，不能当作完整条件 KL     |
| Teacher token reward        | `opd/token_reward_mean`  | 有效 token 平均 $\operatorname{mean}(r)$，来自 teacher 的稠密蒸馏信号，不是答题正确率奖励                                |
| Log-ratio RMS               | `opd/log_ratio_rms`      | $\sqrt{\operatorname{mean}(d^2)}$，避免 signed mean 的正负抵消；不同来源的取样分布不同，不能直接作为共同测度上的对齐排名 |
| Gradient norm               | `actor/grad_norm`        | 裁剪前的全局梯度 L2 范数；阈值为 1，日志可大于 1                                                                         |
| Student entropy             | `actor/entropy`          | student 在该来源 prefix 上的条件熵平均，prefix 集随来源变化                                                              |
| Actor surrogate loss        | `actor/loss`             | 当前 PPO surrogate；单次更新起点与负 token reward 接近，负值不表示 KL 为负                                               |

当前运行没有保存全词表条件 KL。仅对给定 prefix、由当前 student 采样的 action，$\mathbb E_{a\sim p_\theta}[d]$ 才是该 prefix 上的 reverse KL；有限样本均值还存在采样误差。在线更新又对 prefix 采样路径停止梯度，因此该局部解释也不等于完整序列 KL 的全部梯度。图中保留“KL proxy”标识，避免把不同数学量混为一谈。

| 来源         | 第 50 步 loss | 第 50 步 grad_norm | 第 50 步 RMS | 前十步 RMS 均值 | 后十步 RMS 均值 |
| ------------ | ------------: | -----------------: | -----------: | --------------: | --------------: |
| 在线 student |      0.054435 |           0.572402 |     0.371102 |        0.546104 |        0.383639 |
| Teacher 轨迹 |     -0.046815 |           0.519051 |     0.296459 |        0.438914 |        0.309305 |
| Semi-OPD     |      0.092669 |           0.803625 |     0.414405 |        0.547853 |        0.424606 |
| Fusion λ=0.5 |     -0.002646 |           0.529680 |     0.314388 |        0.427681 |        0.324119 |

四组 RMS 的后十步均值低于前十步，但这只描述各自训练轨迹上的残差。Fusion 的 signed mean 接近零，RMS 仍为 0.314；奖励均值接近零不足以支持对齐完成。非零 batch 梯度本身也不能证明期望梯度没有收敛，长时判断需要共同缓存上的多个窗口。

![响应长度、4K 触顶比例、序列奖励与 actor 更新耗时](figures/training-rollouts.png)

响应长度与触顶比例用于判断 4K 训练预算的影响。`clip_ratio` 表示长度达到 cap，不单独判定该位置是否恰好为自然 EOS。缓存保留 finish reason，但上图使用四组共同记录的长度指标。序列奖励为 token reward 的逐响应求和均值，会受长度影响；actor update time 仅含参数更新阶段，不包含轨迹生成、缓存准备或验证。

## 共同设置与数据范围

| 项目           | 正式对照设置                                                                                                        |
| -------------- | ------------------------------------------------------------------------------------------------------------------- |
| 可训练 student | DeepSeek-R1-Distill-Qwen-1.5B，每组从相同初始资产和全新 optimizer 开始                                              |
| 冻结 teacher   | JustRL-DeepSeek-1.5B；生成 teacher 轨迹与训练监督使用同一资产                                                       |
| 设备与环境     | 物理 GPU 0–3，四张 H800 80GB；独立 `envs/opd-state`；Torch 2.7.1+cu126、vLLM 0.10.0+cu126、Transformers 4.57.6      |
| Prompt 数据    | 相邻 `MD-OPD/data/math/DAPO/train.parquet`，17,917 行，prompt cap 过滤后 17,909 行                                  |
| 配对消费顺序   | 原 RLHFDataset、StatefulDataLoader、seed 42 的前 50 batch，共 3,200 条消费记录；不声称是全量去重语料                |
| Batch 与更新   | 每批 64 prompt、每题一条响应；全局 mini-batch 64、每步一个 PPO epoch，micro-batch 只作梯度累积                      |
| 长度与采样     | prompt cap 1,024；训练响应 cap 4,096；temperature 1、top-p 1、top-k 不截断、repetition penalty 1                    |
| 优化           | AdamW，恒定学习率 $10^{-6}$、无 warmup、梯度裁剪 1；有效 response token 的 token-mean                               |
| 蒸馏目标       | `token_reward_direct`，sampled-token 信号，`LOG_PROB_TOP_K=0`；无额外 reference KL 正则或行为重要性修正             |
| 验证           | MATH-500 500 题、AMC23 40 题、AIME25 30 题，每题 16 次，共 9,120 条响应；temperature 0.7、top-p 0.9、cap 31,744     |
| 验证批量       | `data.val_batch_size=32`，共 18 批；不减少每题的 16 次采样                                                          |
| 保存与记录     | 50 步组仅最终验证；长时组每 50 步验证；无训练前验证；`save_freq=-1`、`resume_mode=disable`；W&B project `OPD-State` |

配对 schedule 的 50 个 batch 的 prompt 长度最小值、均值和最大值与原在线 baseline 日志全部匹配。用随仓配置独立重建后，全部索引、prompt token、parquet 和 prompts JSONL 的 SHA-256 都与生产 schedule 一致，见[重建证据](results/2026-10-18-schedule-reproduction.json)。长时 Semi 重复消费同一 3,200 条记录，每 50 步覆盖一轮。

研究来源是 Zhao 等的 _When Do We Need On-Policy Distillation?_[^semi]。该论文的 Semi-OPD 固定初始 student 的轨迹，teacher 来源消融替换轨迹生成方。本实验采用本地已有的 R1-Distill/JustRL 配对、batch 64、4K 训练响应和三基准 eval16，不是该论文 14,116 条去重 DAPO、batch 128、AIME/HMMT avg@32 的严格复现。论文的不同配对还区分思考模式与 4K/16K cap；本地格式化 prompt 去重后 17,255 条，也不等于论文数据。

若后续检验低 overlap 下的来源效应，可准备 Qwen3-1.7B non-thinking 与 Qwen3-4B-Instruct-2507，固定 revision、模板、长度和数据，再增加 seed 42/43/44；这是后续可变选择，尚未执行。当前没有测量该模型配对的 top-64 overlap，不用模型名称替代测量。

## 四种轨迹来源与 student 更新

所有设置只更新 student，teacher 始终冻结并处于 eval/no_grad 模式。Teacher rollout 意味着 teacher 生成响应，student 接受训练；最终验证仍由 student 独立生成。各组的生成方和生成时机如下。

| 设置            | 行为分布                                                                    | 生成时机               | 参数更新方 |
| --------------- | --------------------------------------------------------------------------- | ---------------------- | ---------- |
| Student rollout | 当前 student $p_{\theta_k}$                                                 | 每次更新前在线生成     | student    |
| Teacher rollout | 固定 teacher $q$                                                            | 训练前预生成一次并缓存 | student    |
| Semi-OPD        | 初始 student $p_0$                                                          | 训练前预生成一次并缓存 | student    |
| Logit fusion    | 当前 student 与固定 teacher 的共享 prefix 融合策略 $\mu_{\theta_k,\lambda}$ | 每次更新前在线生成     | student    |

Teacher rollout 的在线重新采样变体尚未运行。重复使用有限缓存与持续生成新 teacher 响应的差异需要单独验证。Semi-OPD 在这里始终指初始 student 一次性生成缓存，不是单条轨迹前段 student、后段 teacher。

令第 $k$ 步更新前的 student 为 $p_{\bar\theta}$，$\bar\theta=\theta_{k-1}$，teacher 为 $q$。对 prefix $s_{it}=(x_i,y_{i,<t})$、实际 action $a_{it}$ 和有效 token mask $m_{it}$，每步重新计算

$$
\ell^S_{it}=\log p_{\bar\theta}(a_{it}\mid s_{it}),\qquad
\ell^T_{it}=\log q(a_{it}\mid s_{it}),\qquad
A_{it}=\operatorname{stopgrad}(\ell^T_{it}-\ell^S_{it}).
$$

EOS 属于有效响应，padding 不参与 loss。一次完整 optimizer update 在更新起点的局部梯度对应

$$
\mathcal L_k(\theta)=-\frac{1}{N_k}\sum_{i,t}m_{it}A_{it}\log p_\theta(a_{it}\mid s_{it}),\qquad
N_k=\sum_{i,t}m_{it}.
$$

实现复用 PPO clipped surrogate，比率分母是本步重算的当前 student 锚点 $p_{\bar\theta}$。单个全局 mini-batch、单 PPO epoch 的更新起点比率为 1，对应上述局部梯度；若增加 epoch 或拆分多个 optimizer mini-batch，需重新核对 clipping 和 dual-clip 的影响，不能继续假定相同。

```pseudocode
\begin{algorithm}
\caption{统一轨迹来源蒸馏\\{\small\textit{Semantics: 根据正文重构}}}
\begin{algorithmic}
\INPUT initial student $p_0$; frozen teacher $q$; ordered prompt batches $X_1,\ldots,X_K$; source $\mu$; response cap $H$
\OUTPUT updated student $p_{\theta_K}$
\IF{$\mu\in\{q,p_0\}$}
    \STATE independently sample and cache one response per scheduled record from frozen $\mu$, stopping at EOS or $H$
    \STATE store token IDs, source identity and finish reason; reject missing or mismatched records
\ENDIF
\STATE $\theta_0\gets\theta(p_0)$
\FOR{$k=1$ \TO $K$}
    \STATE $\bar\theta\gets\theta_{k-1}$
    \IF{$\mu\in\{q,p_0\}$}
        \STATE read the fixed responses for $X_k$; never fill a cache miss by online generation
    \ELSE
        \STATE synchronize the student inference replica to $\bar\theta$
        \STATE independently sample one response per prompt from current student or shared-prefix fusion, stopping at EOS or $H$
    \ENDIF
    \STATE construct valid response masks including EOS and excluding padding; reject an empty valid-token batch
    \STATE recompute current $\ell^S_{it}$ and frozen-teacher $\ell^T_{it}$ on these exact trajectories
    \STATE $A_{it}\gets\operatorname{stopgrad}(\ell^T_{it}-\ell^S_{it})$
    \STATE update only student parameters once with the current-student PPO anchor, producing $\theta_k$
    \IF{$k$ is a validation step}
        \STATE synchronize the updated student and evaluate student-only responses on all three benchmarks
    \ENDIF
\ENDFOR
\RETURN $p_{\theta_K}$
\end{algorithmic}
\end{algorithm}
```

该伪代码概括来源选择和更新顺序；当前缓存版本只保存轨迹，teacher 的 token 分数在每步重算，没有保存 teacher log-probability 或旧 student 概率。缓存 miss 或协议不符直接失败，不改用在线采样。论文预缓存 teacher 分数的效率收益没有包含在当前实现中。

固定来源同时改变 prefix 与 action。设固定缓存 token 测度为 $\nu_\mu(s,a)$，其权重与 $\theta$ 无关，teacher 也固定。在更新起点，未作行为重要性修正的梯度为

$$
g_\mu(\theta)=\mathbb E_{(s,a)\sim\nu_\mu}\left[(\log p_\theta(a\mid s)-\log q(a\mid s))\nabla_\theta\log p_\theta(a\mid s)\right].
$$

记 $d_\theta=\log p_\theta-\log q$，则固定测度下的平方残差满足

$$
\begin{aligned}
F_\mu(\theta)&=\frac12\sum_{s,a}\nu_\mu(s,a)d_\theta(s,a)^2,\\
\nabla_\theta F_\mu(\theta)
&=\frac12\sum_{s,a}\nu_\mu(s,a)\,2d_\theta(s,a)\nabla_\theta d_\theta(s,a)\\
&=\sum_{s,a}\nu_\mu(s,a)d_\theta(s,a)\left(\nabla_\theta\log p_\theta(a\mid s)-0\right)\\
&=g_\mu(\theta).
\end{aligned}
$$

这里的等式依赖固定缓存测度与冻结 teacher；在线 student 或 fusion 的行为分布随更新变化，不能直接将整段训练解释为同一个固定 $F_\mu$ 的下降。来源差异同时包含 prefix、action、长度和内容差异，末步正确率不能单独识别 prefix 访问测度的因果作用；相关推导见[访问测度研究](../research/occupancy/2026-10-17-OPD访问测度与收敛机制研究.md)。

Fusion 每个 token 都在同一 prefix 上计算

$$
\mu_{\theta,\lambda}(a\mid s)=\operatorname{softmax}\!\left(\frac{(1-\lambda)z_{p_\theta}(s)+\lambda z_q(s)}{T}\right)_a,\qquad \lambda=0.5,\quad T=1.
$$

每次只从融合分布采样一个 token，再将同一个 token 送入两个模型各自的 KV cache。两个模型不能各自生成后拼接。teacher 固定，student 推理副本随 actor 完整权重流更新；第 $k$ 步生成版本必须为 $k-1$，不完整或过期权重拒绝生成。融合策略的 log-probability 只作诊断，不用作 PPO old-log-probability 或重要性权重；监督仍来自原始 $q$。

## 评测计算与收敛条件

某题 $n=16$ 个响应中 $c$ 个正确，题内 mean@16 为 $c/16$，再在题目间平均。标准无放回 pass@k 为

$$
\widehat{\operatorname{pass@k}}=1-\frac{\binom{n-c}{k}}{\binom{n}{k}},\qquad k\in\{1,2,4,8,16\}.
$$

当 $n-c<k$ 时分子为零。pass@1 等于 mean@16；pass@16 是 16 个响应中是否至少一个正确。原生 veRL 的 bootstrap best@k 使用有放回采样，语义不同；例如恰好一个正确时，标准 pass@16 为 1，有放回 best@16 的期望为 $1-(15/16)^{16}$。原生 mean/std/best/worst/maj 保留，额外输出精确 pass@k，不相互改名。

验证必须包含三个完整基准、570 个不同题目和每题恰好 16 次二值评分。采集器从原始 JSONL 独立重算，与日志和 W&B 全部精确指标逐项核对，保存逐题 CSV 哈希及内容身份。CSV 固定 LF，避免 Git 换行归一化改变校验值。

长时 Semi-OPD 运行名为 `math-semi-opd-long-800steps-eval16-20261018-092912`，W&B 为 uq1riryx。[^long-run] 其[启动配置证据](results/2026-10-18-semi-long-startup.json)确认完整 eval16、test_freq=50、16 个 epoch、最大 800 步、恒定学习率、无 warmup 和无 checkpoint。模型每 50 步重复同一缓存，适合比较共同状态上的残差窗口；完整验证每次约 95–101 分钟，远长于约七分钟的 50 步缓存更新阶段。

经验平台判据在长时结果产生前固定：至少 300 步，最后四个连续、间隔 50 步的验证点，macro mean@16 极差不超过 0.01；对应四个训练窗口的 token 加权 RMS 相对极差不超过 5%，平均 grad_norm 相对极差不超过 10%。相对极差为最大值减最小值后除以均值，分母下限 $10^{-12}$。

第 $b$ 批有效 token 数 $N_b$、RMS $r_b$ 的窗口合并值为

$$
r_{\mathrm{window}}=\sqrt{\frac{\sum_b N_b r_b^2}{\sum_b N_b}}.
$$

batch 均为 64，使用平均响应长度作权重与 $N_b$ 等价。训练器和事后分析共用 `plateau.py`：观察不足为 `insufficient_observation`，达到条件为 `observed_plateau`，其余为 `criterion_not_met`。达到条件后在验证结束时正常退出；800 步仍不满足则记录 `budget_exhausted=true`，再按实际趋势决定预算。该判据是有限预算经验平台，不证明梯度为零、已与 teacher 对齐或渐近收敛。

## 实现核验与性能

| 实现层             | 已核验的行为与证据                                                                                                                      |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| 固定缓存           | SQLite 只读完整性、模型完整文件签名、tokenizer/模板、prompt ID/内容、响应 cap/EOS 与配对数据哈希；miss 拒绝；50 步全命中                |
| 当前 student 更新  | 缓存不含 old_log_probs，当前 actor 每步重算；teacher requires_grad=False、eval/no_grad；验证前同步最终 student                          |
| CPU 与维护         | 12 项原生 CPU 测试覆盖融合端点、共享 prefix、EOS、padding、缓存重置、teacher 冻结和版本；27 项维护测试涵盖指标、缓存、RPC 与平台控制    |
| Fusion 数值        | 同 FlashAttention KV 内核 graph/eager logits 最大差为零；静态/动态 FP32 最大 logits 差 $4.96\times10^{-5}$、分布 TV $2.27\times10^{-6}$ |
| Attention 独立核验 | 含左 padding、GQA、prefill/decode 的原始 Q/K/V 对 FP32 数学内核，相对 L2 误差 0.00159–0.00199，阈值 0.005                               |
| 权重流与训练桥接   | GPU 版本 0→1→2 原地更新、改变 lm_head 后输出变化、恢复后还原、teacher 探针不变；完整两步训练和 96 条短验证通过                          |
| 正式 Fusion        | 50 步版本严格为 0–49、λ=0.5、loss/梯度/行为 log-probability 有限；12,436,547 个有效响应 token 与四 worker mask 计数对应                 |

数值与训练证据见 [FP32 检查](results/2026-10-18-fusion-gpu-fp32.json)、[独占 GPU 基准](results/2026-10-18-fusion-gpu-isolated.json)、[两步训练](results/2026-10-18-fusion-two-step-smoke.json)和[Fusion 不变量](results/2026-10-18-math-fusion-lambda05-50steps-eval16-20261018-065746/fusion-invariants.json)。两步调试使用训练 cap 64、验证 cap 128，六题各 16 次，只验证集成，不计为任务成绩。

跨 BF16 attention 内核存在多层舍入差异：batch 1/8/16 的最大分布 TV 约 0.029–0.0314，最大 logits 差不超过 0.375。最初 TV 0.01 阈值未通过；完成 FP32 mask 和独立 attention 核验后，跨 BF16 阈值设为 TV 0.05、logits 差 0.5，同内核 graph/eager 阈值保持 $10^{-5}$。该验收不保证跨后端逐 token 相同。Fusion 使用 Hugging Face 双模型解码，其他来源使用原生 vLLM，因此实验比较包含后端实现差异，不能单独识别完全相同内核下的 λ 效应。

生产后端 `cuda_graph_flash` 以两个独立 StaticCache 保留 KV，用物理有效长度与左 padding 约束注意力，并保留 GQA，避免复制 KV。prefill 使用 eager，单 token 双模型 decode 捕获 CUDA Graph。当前限 full-attention、同 dtype、固定 default RoPE、CUDA BF16/FP16、单 GPU 推理副本、单轮纯文本、每题一个 rollout、全参数 actor；LoRA、FP8、TP/PP>1 和多模态拒绝启用。

独占 H800、固定 KV 容量 5,120、生成 128 个新 token 的热调用微基准如下。它比较两种双模型实现，不代表完整 4K 训练相对原生 vLLM 的加速。

| Batch | DynamicCache eager，token/s | CUDA Graph + FlashAttention KV，token/s | 加速比 | 图路径峰值分配显存，GiB |
| ----: | --------------------------: | --------------------------------------: | -----: | ----------------------: |
|     1 |                       36.69 |                                  119.78 |   3.26 |                    7.03 |
|     8 |                      219.06 |                                  872.09 |   3.98 |                    9.10 |
|    16 |                      445.93 |                                1,663.44 |   3.73 |                   11.46 |

| 来源         | 缓存预生成，秒 | 训练步骤合计，秒 | 最终验证，秒 |
| ------------ | -------------: | ---------------: | -----------: |
| 在线 student |         不适用 |          2,348.7 |      5,723.3 |
| Teacher 轨迹 |          545.4 |            409.6 |      6,068.4 |
| Semi-OPD     |          537.5 |            410.2 |      5,708.2 |
| Fusion λ=0.5 |         不适用 |          2,999.0 |      5,907.6 |

各计时范围不含相同的启动开销，不能直接相加作为严格的端到端时间。Fusion 生成计时为 2,450.6 秒，训练总耗时仍高于在线单模型组；缓存组的成本比较须计入预生成。

## 复现与结果位置

从仓库根目录执行。正式模型、数据和独立环境位于工作区相邻目录。已有缓存或 schedule 时生成器拒绝覆盖；独立复核应换输出目录。`OPD_SOURCE_COMMIT` 可固定历史提交；正式生产源码使用每个 run 的私有快照，后续开发不会改变已运行任务。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/verl" \
    ../envs/opd-state/bin/python data/prep/build_rollout_cache.py schedule \
    --baseline-config data/prep/configs/math-rollout-schedule.json \
    --output runs/rollout-sources/paired-schedule --steps 50 --batch-size 64
OPD_FREEZE_SOURCE=1 bash recipe/prepare_math_cache.sh cached_teacher runs/rollout-sources/teacher-cache
OPD_FREEZE_SOURCE=1 bash recipe/prepare_math_cache.sh cached_initial_student runs/rollout-sources/student-cache
OPD_FREEZE_SOURCE=1 bash recipe/math_online.sh
OPD_FREEZE_SOURCE=1 bash recipe/math_cached.sh cached_teacher runs/rollout-sources/teacher-cache
OPD_FREEZE_SOURCE=1 bash recipe/math_cached.sh cached_initial_student runs/rollout-sources/student-cache
FUSION_LAMBDA=0.5 FUSION_BACKEND=cuda_graph_flash OPD_FREEZE_SOURCE=1 \
    bash recipe/math_fusion.sh
STEPS=800 TOTAL_EPOCHS=16 OPD_FREEZE_SOURCE=1 \
    bash recipe/math_cached.sh cached_initial_student runs/rollout-sources/student-cache \
    +trainer.plateau_min_steps=300
```

这些训练依次运行，不能同时占用同一四张卡。持久串行调度入口 `recipe/run_math_sources.py` 保存 source commit、命令、日志和退出码，等待指定 teacher pipeline 完成后安排 GPU 核验和后续阶段；复用它时必须填写现场 PID 和实际运行目录。失败阶段先检查日志，不覆盖缓存，也不停止其他 Ray 作业。W&B 使用既有 netrc，配方设置 project 为 OPD-State 并清除不匹配的继承 API key。

结果采集、远端核验和题目比较：

```bash
python3 eval/collect_math_experiment.py --run runs/rollout-sources/实际运行名 \
    --output docs/experiment/results/实际结果目录 --steps 实际步数 --interval 50
env -u WANDB_API_KEY NETRC="$PWD/../.secrets/netrc" WANDB_CONFIG_DIR="$PWD/../config/wandb" \
    WANDB_BASE_URL=https://api.wandb.ai ../envs/opd-state/bin/python \
    eval/verify_wandb_math.py --result docs/experiment/results/实际结果目录
.venv/bin/python eval/analyze_math_sources.py \
    --result student-online=在线结果目录 --result teacher=teacher结果目录 \
    --result semi-opd=Semi结果目录 --result fusion-lambda05=融合结果目录 \
    --reference student-online --output 比较结果目录 --step 50
```

参考 run 导出与 Python 绘图已实际执行。绘图只读取已归档结果，离线重画不访问 W&B；`FIGURE_VIZ_HOME` 可指定外部 figure-viz 技能根目录，默认使用工作区 `.agents/skills/figure-viz`。该资源不随训练仓库复制，跨机器须另外准备。

```bash
env -u WANDB_API_KEY NETRC="$PWD/../.secrets/netrc" WANDB_CONFIG_DIR="$PWD/../config/wandb" \
    WANDB_BASE_URL=https://api.wandb.ai ../envs/opd-state/bin/python \
    eval/export_math_references.py --output docs/experiment/results/reference-accuracy.json
.venv/bin/python eval/plot_math_experiment.py
```

图件提供 PNG 和 PDF，`figures/training-plotted.csv` 保存 200 个训练点，`accuracy-plotted.csv` 保存四设置、四聚合范围的末步指标，`audit.json` 保存输入和输出哈希及机器审计。原始逐题表、训练 CSV、精确验证 JSON 和 W&B 核验保留在各结果目录，合并报告不删除数据证据。

| 设置         | W&B run                | 结果摘要                                                                                            |
| ------------ | ---------------------- | --------------------------------------------------------------------------------------------------- |
| 在线 student | g50xyfia[^online-run]  | [summary.json](results/2026-10-18-math-student-online-50steps-eval16-20261018-044021/summary.json)  |
| Teacher 轨迹 | va2nhhra[^teacher-run] | [summary.json](results/2026-10-18-math-teacher-cache-50steps-eval16-20261018-005033/summary.json)   |
| Semi-OPD     | scodl08e[^semi-run]    | [summary.json](results/2026-10-18-math-semi-opd-50steps-eval16-20261018-025524/summary.json)        |
| Fusion λ=0.5 | g5m60gys[^fusion-run]  | [summary.json](results/2026-10-18-math-fusion-lambda05-50steps-eval16-20261018-065746/summary.json) |

Teacher 组训练源码固定于 `350a286be0d586f04e4cd0b0e89624a336b17a81`，其余正式来源组、fusion 与长时组固定于 `4eb96bb1ab62fc7bac735c463dd54bafd315a62a`。采集、远端核验、分析和绘图源码分别记录哈希，不把后续分析提交误当作训练版本。配对 parquet SHA-256 为 `a98110172a296c791ceb10c27de9f335b905ffb46f98ab119383ddcb9886c760`；完整模型和缓存签名见各组 manifest 与 fusion model-provenance。

## 历史调试与后续范围

最初在线链路验证采用 MATH-500 greedy@1、8,192 cap，结果为 355/500，即 71%，W&B 为 finished，50 次更新和无 checkpoint 已核对。它只作为工程链路证据，不能与当前 eval16 三基准成绩直接比较；[原摘要](results/2026-10-18-math-baseline-summary.json)、[训练记录](results/2026-10-18-math-baseline-steps.csv)和[逐题分数](results/2026-10-18-math-baseline-scores.csv)继续保留。早期四卡两步 smoke 的[环境与数据证据](results/2026-10-17-math-smoke.json)同样不作为准确率结论。

剩余工作是取得长时 Semi 的每 50 步完整评测，检查共同缓存上的残差与经验平台；若仍有持续趋势，再决定延长预算。更强的来源效应结论还需多 seed、长度与有效 token 预算控制，以及相同 prefix/action 测度的诊断。缓存 miss 在线补采样、旧学生概率替代当前 PPO 锚点、teacher 被更新、fusion 两模型接收不同 prefix、验证使用 teacher/fusion、缺基准或缺采样数均属于失败条件，不能计为有效结果。

### 参考文献

[^semi]: Zhao 等。When Do We Need On-Policy Distillation? arXiv 2610.11291, 2026. [论文](https://arxiv.org/abs/2610.11291). 设置按工作区论文源码 sections/02_method.tex、05_why.tex、06_more.tex、09_appendix.tex 核对。

[^reference-teacher]: wncfht. One-Shot-OPD math_teacher u87buult. W&B 2026. [用户指定 teacher 评测](https://forge.coreweave.com/wandb/wncfht/One-Shot-OPD/runs/u87buult).

[^reference-student]: wncfht. One-Shot-OPD math_student 6uorrk2k. W&B 2026. [用户指定初始 student 评测](https://forge.coreweave.com/wandb/wncfht/One-Shot-OPD/runs/6uorrk2k).

[^online-run]: wncfht. OPD-State g50xyfia. W&B 2026. [在线 student 训练](https://wandb.ai/wncfht/OPD-State/runs/g50xyfia).

[^teacher-run]: wncfht. OPD-State va2nhhra. W&B 2026. [Teacher 轨迹训练 student](https://wandb.ai/wncfht/OPD-State/runs/va2nhhra).

[^semi-run]: wncfht. OPD-State scodl08e. W&B 2026. [Semi-OPD 训练](https://wandb.ai/wncfht/OPD-State/runs/scodl08e).

[^fusion-run]: wncfht. OPD-State g5m60gys. W&B 2026. [Fusion 训练](https://wandb.ai/wncfht/OPD-State/runs/g5m60gys).

[^long-run]: wncfht. OPD-State uq1riryx. W&B 2026. [长时 Semi-OPD](https://wandb.ai/wncfht/OPD-State/runs/uq1riryx).
