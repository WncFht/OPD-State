# Rollout 来源与 logit fusion 工作记录

## 目标与验收

继续完成 teacher-cache、Semi-OPD 和 lambda=0.5 当前 student/teacher logit fusion 的实际训练。模型、采样、cap、优化器、统一 AMC23、AIME25、MATH-500 每题 16 次采样的评测，原 greedy baseline 仅作工程历史，GPU 限物理 0–3，W&B project 仍为 OPD-State，不保存模型 checkpoint。

缓存实验先选择原 baseline DataLoader 前 50 个 batch 的 3,200 次 prompt 消费，保留原始行身份与顺序，分别从冻结 teacher 和初始 student 生成一次响应。两个 50 步缓存组严格读取缓存，miss、prompt、tokenizer、配置或内容不匹配即失败；每步重算当前学生概率，冻结教师仍实时打分。长 Semi-OPD 重复同一缓存，每 50 步评测，并结合多次评测和固定缓存上的 log-ratio RMS 判断有限预算平台。原 baseline 后 20 步的 loss 仍下降，不能宣称已收敛。

Fusion 定义为同一 prefix 上 (1-lambda)*student_logits+lambda*teacher_logits，采样后两个模型都接收同一 token；teacher 固定，student 每步同步当前训练权重。lambda 的端点、常数平移、EOS、padding、KV 更新、不同批大小及权重同步必须验证。先记录独立 GPU 解码基准，再优化缓存/批处理和热路径，不能以形状检查代替实际运行。正式 lambda=0.5 训练 50 步，目标仍为原始教师的 sampled-token 信号，行为分布改变单独记录。

失败条件：缓存组悄悄在线补采样、缓存旧学生概率被当作当前 PPO 锚点、验证使用 teacher/fusion 而不是 student、fusion 两模型收到不同 prefix、数值或权重版本不一致，均不计为完成。工作区原有 docs/research/AGENTS.md 删除继续保留，不纳入任务提交。

## 实现选择

先实现严格的轨迹读取与现有 DataProto/训练器集成，复用实时教师评分，避免同时重写优化路径。已有按 step 的 SkipManager 存储不能直接提供按稳定样本身份重放的保证，因此固定来源配置需要明确的缓存读取接口和验证；序列化采用可校验的 SQLite 数据与独立 manifest。数据准备使用原 RLHFDataset、sampler 与 StatefulDataLoader 生成配对顺序。

## 评测修正与当前证据

用户补充评测要求：每题采样 16 次，并在 AMC23、AIME25、MATH-500 同时计算 mean/pass 等指标。原 greedy 71% 不可作为正式来源对照，在线组需重跑。新增标准无放回 pass@k，不将原生 bootstrap best@k 改名冒充。

已生成 3,200 条配对 schedule，50 个 batch 的 prompt 均值/最小/最大长度全部与旧 baseline 日志相符。单卡 teacher 生成 8 条短轨迹完成；四卡缓存训练完成两步，cache_hit=1，读取计数 8→16，当前学生梯度有限；三基准短验证共 96 个响应、每基准两题、每题 16 次采样，完整运行退出码为 0。

指标汇总微基准为 20 个合成题、每题 16 次采样，参考耗时 1.796 秒、向量化 0.135 秒，所有指标差值为零，约 13.3 倍。另有多数投票平局与 NumPy 状态对照，原生 Torch 的 EOS=padding、位置与旧概率字段检查通过。缓存模型同步优化从每步同步改为仅评测前同步，保持每步 actor 概率与教师打分。
