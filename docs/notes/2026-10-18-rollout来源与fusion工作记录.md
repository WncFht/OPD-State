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

01:19 已启动实际调度进程 3846840，目录 `runs/rollout-sources/suite-20261018-0118`，固定源码 68bb191e0db4a0e976b54239c1328572409f8835。进程与状态均核对，当前为 `waiting_for_teacher`，等待 pipeline PID 3749135 和 `TEACHER_TRAIN_FINISHED` 成功标志。teacher 已完成三个验证 batch，最终指标尚未写出；没有将排队任务视为实验完成。顺序为独占 GPU 基准、Fusion 两步调试、初始 student 缓存、Semi-OPD 50、在线 student 50、通过调试的 Fusion 50、Semi-OPD 300。所有正式验证保持三基准、每题 16 次，长时组每 50 步验证。完整维护检查与 12 项 CPU 测试通过，归档 GPU 检查的两个源码 SHA-256 均与该提交一致。

进一步检查实际 RPC 链路，vLLMHttpServer.collective_rpc 原方法只 await，没有 return，导致生成端虽然完成计算，ServerAdapter 只能收到 None。补上返回值，并以原方法 AST 编译执行回归测试，核对生成 payload 的对象身份及全部转发参数；同时核对安装的 vLLM AsyncLLM/core client 均返回 worker 的列表结果。六项生命周期/RPC 测试通过。原等待队列以 SIGINT 正常退出以更换固定提交；该动作只作用于 PID 3846840，没有停止 teacher 或其他 GPU 作业。新版本检查完成后重新启动等待队列。

新增题目内容身份、逐题 CSV 与采集器源码哈希，以及成对题目 bootstrap 比较，避免随机 validation UID 无法跨运行配对。平台判据在长时结果产生前固定：至少 300 步、最后连续四个验证点，macro mean@16 极差至多 1 个百分点、窗口 RMS 相对极差至多 5%、平均梯度范数相对极差至多 10%。它只表示有限预算经验平台，不表示梯度为零、已与教师对齐或渐近收敛。固定缓存训练增加验证后正常停止机制，最大预算从 300 提高至 800；期间保持同一训练进程，不写模型权重。到 800 仍不满足时明确标记预算耗尽。

七项结果分析/停止测试已通过，包括正负对照、按 token 数合并 RMS、最低观察预算、步数缺口、逐题指标篡改拒绝，以及从真实 trainer AST 编译执行的正常提前停止与预算耗尽分支。含 teacher 重复响应的两步 smoke 已重新采集，题目内容身份与 CSV 哈希校验通过。GPU 长时控制尚未实际触发，不能把这些 CPU 证据当成已经取得收敛结果。

01:32 新队列进程 3881040 已启动，状态与活进程核对为 `waiting_for_teacher`，目录 `runs/rollout-sources/suite-20261018-0132`，固定提交 4eb96bb1ab62fc7bac735c463dd54bafd315a62a。该版本包含 b1000c7 的 RPC 修复、内容身份、比较脚本和 300–800 步平台停止。原 teacher 进程 3749135 仍在运行，完整验证已完成五个 batch，尚无最终指标。新版本完整维护检查通过 26 项测试、无阻断项；训练环境 12 项原生 CPU 测试和长时配置 dry-run 通过。

02:41 teacher 50 步与 18 批完整验证全部完成，进程退出码 0。原始 9,120 条响应、570 题及全部精确指标与日志逐项一致；W&B va2nhhra 状态 finished，全部 50 步的 loss/梯度有限，远端验证点只有 50。mean@16 为 MATH-500 0.85375、AMC23 0.753125、AIME25 0.26875。缓存生成 545.4 秒、更新计时合计 409.6 秒、最终验证 6,068.4 秒。结果与来源元数据已归档，仍不能与旧 greedy 数值直接比较。

GPU 释放后，实际独占性能基准通过；batch 1/8/16 的图路径相对 DynamicCache eager 加速 3.26/3.98/3.73 倍，图路径 batch 16 为 1,663.44 token/s，峰值分配显存 11.46 GiB。同内核图/eager 差值仍为零，源码哈希与固定提交对应。队列随后进入真实 Fusion 两步训练；已观察到学生更新和原生权重同步，完整短验证尚需结束。共享 GPU 耗时仍只保留为诊断，性能取舍以独占报告为准。

Fusion 两步训练随后正常退出，三基准六题各 16 次短验证共 96 条通过独立采集。学生权重版本为 0→1，lambda 0.5，两步梯度范数为 4.36988、2.95250，行为 log-probability 有限，原生 student-only 验证路径正常。独立解码与完整训练桥接两层均通过，证据已归档；该短 cap 调试不计为正式准确率。当前队列进入初始 student 缓存生成，随后训练 Semi-OPD 50 步。

归档时修正 CSV 默认 CRLF 与 Git LF 归一化造成的校验值差异：采集器固定 LF，重新采集并验证 Git index 中的逐题 CSV 哈希与元数据一致。凭据检查对 `tokenizer_signature` 的误报经本地 tokenizer 独立重算确认；仅为 `docs/experiment/results/*/cache-manifest.json` 中该字段的完整小写 64 位十六进制 SHA-256 增加例外。六种路径、字段和格式边界的正负对照通过，其他 API-key 形式仍被拦截。远端 W&B 核验代码保存为独立脚本并已实际执行。

初始 student 缓存已完整生成，3,200 条、12,482,086 个响应 token，生成耗时 537.5 秒，数据/模型/tokenizer 签名与配对协议一致。队列已启动 Semi-OPD 50 步训练。补齐可移植的 schedule 准备配置，从本地模型与 DAPO 数据独立重建，不再要求历史 W&B 配置导出；逐行 prompt、全部选中索引、parquet 与 prompts JSONL 的 SHA-256 均与现有生产 schedule 完全一致。数据身份的文件路径规范化保持已有绝对路径身份不变，配置与生成器源码哈希单独记录。新输出目录用于核验，没有替换正在消费的生产数据。

Semi-OPD 运行 `math-semi-opd-50steps-eval16-20261018-025524`、W&B scodl08e 已完成 50 次更新并开始末次完整验证。第 50 步日志需等验证结束后才写出；目前 1–49 步的 loss、梯度和 RMS 均有限，缓存命中率全部为 1。前十个已记录步骤的 RMS 均值 0.547853，最后十个已记录步骤（40–49）为 0.426536，第 49 步梯度范数 0.840146。任务准确率和正式收敛判断仍等待完整评测及在线学生对照，不能把已启动验证记作结果完成。

Semi-OPD 随后完成全部验证并正常退出，W&B scodl08e 为 finished，全部 50 步与远端指标核验通过。mean@16 为 MATH-500 0.867、AMC23 0.7875、AIME25 0.283333，macro 0.645944。相对 teacher 轨迹组的 macro 差值为 +2.074 个百分点，成对题目 bootstrap 95% 区间 [0.505, 3.641]；该区间不覆盖训练 seed 方差。AIME25 pass@16 从 16/30 降为 14/30，不能按平均正确率提升宣称所有指标更好。缓存读取 3,200 条，末步 RMS 0.414405、梯度范数 0.803625；后十步 RMS 均值 0.424606。更新计时 410.2 秒、验证 5,708.2 秒。结果、远端核验和两个来源的逐题配对比较均已归档。

队列已启动在线 student 的同协议 50 步对照，W&B g50xyfia。当前仍等待三个来源的完整比较；正式 Fusion 50 步与长时 Semi-OPD 将按既定顺序继续，生产源码仍固定为 4eb96bb1ab62fc7bac735c463dd54bafd315a62a。
