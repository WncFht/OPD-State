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

## Fusion 实现方向

采用同 GPU 上的两份 Hugging Face 推理模型，分别维护独立 KV cache，共用已采样 token；student 推理副本随 actor 的原生权重流更新，teacher 永不更新。通过 vLLM worker extension 的独立批量 RPC 生成训练轨迹，验证仍走原生 student vLLM 路径，避免把融合分布用于评测。当前模型显存允许复制两份推理模型。

先实现 DynamicCache eager 参考，随后比较 StaticCache 和 CUDA Graph 热路径。固定 cache、输入缓冲、4D attention mask 与 RoPE position 让 decode 图可复用；prefill 不捕获。图模式的适用模型和 mask 条件明确限制，失败不静默改策略。数学端点、同一 prefix、padding/EOS、缓存复位、当前学生版本、teacher 不变及权重同步都要验证。只有 GPU 基准证明正确与可用后才运行 lambda=0.5 的 50 步。

正式 teacher pipeline 已提交至本地独立进程，先复验仅验证前同步，再生成 3,200 条 teacher 轨迹并训练/评测 50 步。生产源码固定在 350a286be0d586f04e4cd0b0e89624a336b17a81 的私有快照，后续 fusion 开发不改变其导入代码。

teacher 完整缓存已生成，50 步更新已到达最终验证。同步优化后的两步 smoke 验证仅同步一次，step 1 不同步、step 2 验证前同步，96 条短验证记录满足三基准各两题、每题 16 次采样。真实结果采集器已在该 smoke 上独立重算并逐字段匹配训练日志。

7b9b99f 提交共享 prefix 的 fusion 候选实现、三个解码后端、版本化参数流、训练桥接、基准脚本与配方。12 项原生 CPU 测试通过，Hydra 配置 dry-run 通过，完整维护检查无阻断项；GPU 基准和两步训练尚未验证。缓存生成 wrapper 和结果采集器作为可复现入口保存，避免仅保留临时运行命令。

GPU 数值验证随后完成。原 SDPA 静态缓存与动态参考在 FP32 下最大 logits 差 4.96e-5、分布 TV 2.27e-6，同内核 graph/eager 差为零；BF16 的跨内核误差超过最初 0.01 的 TV 阈值，逐层舍入需要独立区分。新增 FlashAttention KV 路径，含左 padding、GQA、prefill/decode 的原始 attention 与 FP32 数学内核相对 L2 误差 0.00159–0.00199。同内核 graph/eager 仍要求差小于 1e-5；跨 BF16 内核按 TV 0.05、logits 差 0.5 验收，三种 batch 实测 TV 不超过 0.0314。共享 GPU 结果已归档，未作为独占吞吐结论。

真实 GPU 已验证 student 参数版本 0→1→2、原地改动 lm_head 后输出改变、恢复参数后原输出恢复、图对象保持不变、教师每个参数的前 32 个元素不变。正式 GPU 调试仍需跑完整训练桥接，不能将独立模型测试当成端到端成功。顺序调度脚本将固定提交、保存命令与退出码，并在 teacher 成功退出后启动独占基准、两步调试和后续来源实验。
