# Semi-OPD 50 步结果与 teacher 轨迹对照

## 结果与范围

Semi-OPD 已完成 50 步 student 更新和三基准完整 eval16。MATH-500、AMC23、AIME25 的 mean@16 分别为 86.70%、78.75%、28.3333%，三基准等权 macro mean@16 为 64.5944%。原始 9,120 条响应、570 题和全部精确指标经独立重算，与训练日志和 W&B 远端摘要一致；训练进程退出码为 0，远端状态为 finished，保存了全部 50 步记录。[^wandb-semi-50]

这里的 Semi-OPD 指训练前用初始 `DeepSeek-R1-Distill-Qwen-1.5B` 一次性生成缓存，随后固定复用。训练更新当前 student，监督教师 `JustRL-DeepSeek-1.5B` 保持冻结；最终成绩由更新后的 student 独立生成得到。teacher 轨迹组同样训练 student，两组改变的是训练轨迹来源。

当前只完成两个固定来源组。在线 student 的同协议对照正在启动，Fusion 50 步和长时 Semi-OPD 尚未完成，因此本文只比较两个来源在单个训练 seed、50 步预算下的结果。

## 准确率与覆盖

验证使用 MATH-500 的 500 题、AMC23 的 40 题和 AIME25 的 30 题，每题 16 个响应，temperature 0.7、top-p 0.9、响应 cap 31,744。下表为百分比，teacher 列来自[teacher 轨迹结果](2026-10-18-teacher轨迹50步结果.md)。mean@16 等于标准 pass@1；标准 pass@16 表示 16 个响应中至少一次正确的题目比例。

| 基准              | teacher mean@16 | Semi mean@16 | teacher pass@16 | Semi pass@16 |
| ----------------- | --------------: | -----------: | --------------: | -----------: |
| MATH-500          |           85.38 |        86.70 |           96.20 |        96.40 |
| AMC23             |           75.31 |        78.75 |           95.00 |        95.00 |
| AIME25            |           26.88 |        28.33 |           53.33 |        46.67 |
| macro，三基准等权 |           62.52 |        64.59 |           81.51 |        79.36 |
| micro，570 题等权 |           81.59 |        83.07 |           93.86 |        93.68 |

Semi-OPD 的三个基准 mean@16 均高于 teacher 轨迹组。AIME25 的 pass@16 则从 16/30 题降到 14/30 题；平均正确率与题目覆盖比例给出了不同的比较，不能概括为 Semi-OPD 在全部指标上更好。

完整的 mean@16、pass@1/2/4/8/16 与逐题正确次数保存在 [Semi-OPD 结果目录](results/2026-10-18-math-semi-opd-50steps-eval16-20261018-025524/summary.json)。两组使用相同的 570 个题目内容身份；原生 bootstrap best/maj 等字段保留在运行日志，未冒充标准 pass@k。

## 成对题目比较

以下为 Semi 减 teacher 的 mean@16 差值，单位是百分点。区间使用按基准分层、成对重采样题目的 bootstrap，5,000 次、seed 42。macro 的三个基准等权，micro 的 570 题等权；原始差值与区间保存在[比较产物](results/2026-10-18-two-source-comparison/analysis.json)。

| 基准     | mean@16 差值 | 95% 成对题目 bootstrap 区间 |
| -------- | -----------: | --------------------------: |
| MATH-500 |       +1.325 |              [0.575, 2.138] |
| AMC23    |       +3.438 |              [0.313, 7.031] |
| AIME25   |       +1.458 |             [-1.667, 4.792] |
| macro    |       +2.074 |              [0.505, 3.641] |
| micro    |       +1.480 |              [0.757, 2.237] |

在这组固定 seed、固定预算的观测中，macro、micro、MATH-500、AMC23 的题目重采样区间不含零，AIME25 的区间包含零。该区间表示对题目选择的敏感性，不估计训练随机种子之间的方差，也不能证明渐近收敛点不同。轨迹来源同时改变 prefix、采样 action 和生成内容，尚不能将差异单独归因于 prefix 访问测度。

## 训练条件与收敛证据

训练使用同一配对顺序的 3,200 条缓存记录，每批 64 条，学习率 $10^{-6}$、恒定学习率且无 warmup、每步一个 PPO epoch、token 平均归约。训练响应 cap 4,096，缓存采样 temperature/top-p 均为 1。全部 50 步缓存命中率为 1，最终读取计数 3,200；每步重算当前 student 与冻结 teacher 的 log-probability，没有缓存旧 student 概率，没有在线补采样。

初始 student 缓存生成耗时 537.5 秒，包含 12,482,086 个响应 token；训练步骤计时合计 410.2 秒，最终验证 5,708.2 秒，约 95.1 分钟。三项计时的启动范围不同，不能直接相加充当严格的端到端计时。只在第 50 步验证，没有保存模型 checkpoint，GPU 为物理 0–3。

末步 actor loss 为 0.0926685，梯度范数为 0.803625，log-ratio RMS 为 0.414405。前十步 RMS 均值为 0.547853，最后十步为 0.424606。只有一个验证预算点，当前不足以判断已经收敛。teacher 与 Semi 的残差取自不同轨迹，不能将两组 RMS 数值直接当作同一分布上的条件 KL 比较。

后续长时 Semi-OPD 使用同一缓存重复训练，每 50 步执行完整 eval16；从 300 步起检查预先固定的经验平台判据，最大首轮预算为 800 步。设置与停止条件见[统一协议](2026-10-18-三基准评测与轨迹来源实验.md)。

## 来源与复现

Semi 训练源码固定在 `4eb96bb1ab62fc7bac735c463dd54bafd315a62a`，teacher 训练使用 `350a286be0d586f04e4cd0b0e89624a336b17a81`，各自源码提交和采集器哈希分别记录。Semi 缓存数据库 SHA-256 为 `0a3c3dff4d507e6e3e3a532ddd45e22d58325d379618f47b7b33baf4e78028b0`；配对 parquet 与 teacher 组相同，为 `a98110172a296c791ceb10c27de9f335b905ffb46f98ab119383ddcb9886c760`。模型、tokenizer、原始响应及逐题 CSV 的校验值保存在结果目录的 manifest 与验证元数据中。

本报告的两个来源比较可由已归档逐题表重算：

```bash
.venv/bin/python eval/analyze_math_sources.py \
    --result teacher=docs/experiment/results/2026-10-18-math-teacher-cache-50steps-eval16-20261018-005033 \
    --result semi-opd=docs/experiment/results/2026-10-18-math-semi-opd-50steps-eval16-20261018-025524 \
    --reference teacher --step 50 \
    --output docs/experiment/results/2026-10-18-two-source-comparison
```

### 参考文献

[^wandb-semi-50]: wncfht. OPD-State Semi-OPD 50-step run scodl08e. W&B 2026. [训练记录](https://wandb.ai/wncfht/OPD-State/runs/scodl08e).
