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

在线 student 随后完成 50 步和全部验证，进程退出码 0，W&B g50xyfia 为 finished；远端 50 步与全部精确指标核验通过。mean@16 为 MATH-500 0.862375、AMC23 0.8015625、AIME25 0.28125，macro 0.648396、micro 0.827522。末步 loss 0.054435、梯度 0.572402、RMS 0.371102；前十步 RMS 均值 0.546104，最后十步 0.383639。训练步骤合计 2,348.7 秒，验证 5,723.3 秒。三个来源逐题集合一致，在线与 Semi 的 macro 差距仅 0.245 个百分点，其配对题目区间包含零；teacher 相对在线 macro 低 2.319 个百分点，区间 [-4.055, -0.755]。三个来源均只有一个验证预算点，当前不能确认收敛，长时 Semi 的共同缓存窗口与多验证点仍有必要。

三来源分析阶段正常退出，队列已启动正式 λ=0.5 Fusion，W&B g5m60gys。已核对前六步：学生版本连续为 0–5、lambda 始终为 0.5、loss/梯度/行为 log-probability 均有限；完整 4K 响应训练路径正常，第六步平均响应长度 3,722.53 token、生成计时 48.08 秒。正式结果仍需等待全部 50 步与三基准 eval16，短路径验收不替代最终实验。

Fusion 随后完成全部 50 步和三基准 eval16，进程退出码 0，W&B g5m60gys 为 finished。mean@16 为 MATH-500 0.86275、AMC23 0.7890625、AIME25 0.272917，macro 0.641576；相对在线 macro 差值 -0.682 个百分点、题目 bootstrap 区间 [-2.149, 0.681]，未显示稳定平均正确率增益。全部 50 步的学生版本严格为 0–49、lambda 0.5、loss/梯度/行为 log-probability 有限，四 worker 生成计数与全局响应 mask 对应，共 12,436,547 个有效响应 token。生成计时 2,450.6 秒、更新步骤合计 2,999.0 秒、最终验证 5,907.6 秒。末步 loss -0.002646、梯度 0.529680、RMS 0.314388；初始模型与词表身份同预生成缓存独立核验一致，最终学生权重未保存。四设置逐题比较和完整核验均已归档。

长时 Semi-OPD `math-semi-opd-long-800steps-eval16-20261018-092912` 已启动，W&B uq1riryx。远端实际配置核验：test_freq=50、eval n=16、三基准完整校验开启、验证 cap 31,744、最早平台检查 300、最大 800、16 个 epoch、save_freq=-1、resume_mode=disable，轨迹来源为 cached_initial_student。已观察到有效更新与 100% 缓存命中，并进入第 50 步完整验证；尚无长时收敛结果。初期 27 步与独立 Semi 50 步实验的 RMS 最大差异约 0.00042，梯度最大差异约 0.0100，不将固定种子解释为分布式混合精度训练逐 bit 相同。

用户再次确认 teacher rollout 的语义后，复核实际缓存 manifest 的 source=cached_teacher、生成器模型选择、训练 actor 更新与冻结教师的 requires_grad=False/no_grad 路径：teacher 生成轨迹，始终由 student 接受更新，最终验证也由 student 独立生成。统一协议补齐四设置的生成方、生成时机与参数更新方；同步 README、架构和开发入口，移除 teacher 报告中“尚无其他同协议结果”的过时状态。已有运行源码快照保持原样。长时任务 PID 3881040 和会话 77179 仍存活，第 50 步完整验证已完成 4/18 批，尚无平台判断。

按用户要求将整个来源与融合对照统一为一项实验，七篇分散的实验文档合并为 docs/experiment/2026-10-19-rollout来源与logit融合实验.md，更新 README、架构、开发和历史工作记录的引用；原始 CSV/JSON 与来源核验保留。新增 eval/export_math_references.py，实际导出用户指定的 u87buult（math_teacher、JustRL）和 6uorrk2k（math_student、R1-Distill）评测摘要，核对 val_only、模型名、三基准与 eval16/0.7/0.9/31744 配置；不宣称核验了远端原始响应和资产哈希。新增 eval/plot_math_experiment.py，用四组已完成结果绘制优化六指标、rollout 四指标、末步准确率与 pass@k 四张图，输出 PNG/PDF、绘图 CSV、来源/输出哈希和机器审计。逐张查看后无重叠或裁切，四图审计均为空；signed log-ratio 明确为 KL proxy，缓存组不冒称完整条件 KL，准确率不补造中间轨迹。长时训练继续原进程与私有源码快照，第 50 步验证推进到 7/18 批。

按用户反馈，将 acc 改为按基准独立缩放的点图，pass@k 同样缩放并加入初始 student/teacher 的精确 pass@k 六曲线，保留 PNG/PDF、坐标范围与参考绘图 CSV。只对不同基准的纵轴一致性审计声明作用域豁免，其他图件规则继续执行。新增 CPU 缓存诊断脚本：核对两个 SQLite 完整性/哈希、3,200 个配对 prompt、Semi 全部 50 步与长时已记录 49 步的 batch 均值，均与固定缓存精确一致；固定的逐样本长度与不同 batch 的均值波动分开解释。穷举二元 token、长度 1–4 的 900 对序列，验证 action 前 prefix 计数及终止边界。真实共同前缀中位数 10、均值 16.675625，配对位置 Jaccard 0.227%，明确不是语义覆盖，也没有同来源重复采样的对照。

逐题核对上游原始 problem/answer 与三个评测 parquet 一致，MATH level 1–5 为 43/90/105/128/134；AMC 来源 URL 显示 12A 22 题、12B 18 题；AIME I/II 各 15。基准组成、难度、题目统计粒度以及 state 的表示簇覆盖/访问频率 JSD/固定 probe 完整条件 KL 设计统一追加到现有报告，未另建实验文档。状态测量设计固定 teacher 编码器与聚类、固定缓存抽样位置，增加同来源独立 rollout 噪声对照和 state/action 分离，避免把表示漂移或稀疏抽样当成新状态。已完成 CPU prefix 诊断，表示空间覆盖与固定 probe KL 仍是下一轮设置，未修改正在运行的私有源码快照。

按用户反馈，将“基准组成、难度与统计粒度”移到准确率表和图之前，补充 macro（基准等权）与 micro（题目等权）的公式和在线组例子。重新核对 2610.11291 方法与附录：论文在训练前为全部 14,116 个训练 prompt 一次生成并预计算教师分数；本地仅缓存与 50 步对照配对的 3,200 条记录，当前教师分数每步重算。新增一张 Mermaid 合图，将两个训练前缓存生成路径、四种每步轨迹来源、共同评分/student 更新、权重或缓存的下一步入口和 student-only 验证放在一图；明确每个 run 独立选择一条来源，训练期间不调用初始 student 补采样。长时 Semi 原进程继续运行，第 50 步验证已完成 14/18 批。

为长时运行增加完整验证前缀的阶段采集模式 --snapshot：只接受目标步完整训练行、三基准响应和精确指标一致的结果，忽略后续正在写入的步骤，保留快照范围及原始日志字节前缀哈希，默认最终采集条件保持。四项回归验证通过；真实在线组的 50 步、570 题、9,120 个响应按快照与默认两种模式重采集，验证记录与训练 CSV 和原归档完全一致。核验产物只读已有运行，新输出在忽略目录，不替换正式结果。

按用户对流程时间、显存及固定 checkpoint 实时生成的提问，新增资源汇总与持续监控脚本、W&B 系统事件导出。分阶段计时显式扣除 adv 内嵌 teacher 打分，累计分解与训练总计一致；actor allocator 与整卡 NVML 采样分开，W&B 历史最高采样分别为在线 45.77、teacher 62.32、Semi 47.43、Fusion 59.30 GiB。只导出 GPU 0–3 的选定事件，按日志时间戳估计训练/验证分界，未把采样峰值当作连续峰值或完整初始化记录。11:09 启动额外两秒整卡监控，进程 105248，监视队列 3881040，输出在 runs/rollout-sources/resource-monitor-20261018-1110；跟踪 PID 启动身份防复用。资源和拟议按需首次生成缓存/fresh 的差异追加到统一报告，没有修改原长时运行。

长时 Semi 已完成第 50 步完整验证并继续第二轮。阶段采集从 570 题、9,120 个响应独立重算，macro mean@16=0.6480277778，MATH-500/AMC23/AIME25 为 0.863875/0.803125/0.2770833333，全部精确字段与日志一致；摘要标记 through_step_snapshot，未宣称训练结束。平台状态 insufficient_observation。长度诊断重跑已覆盖到第 71 步，全部观测与固定缓存精确相等，51–71 步确实重现首轮 1–21 批。

重新逐项核对 arXiv 2610.11291v1 英文原文 §2.2 公式 (2)–(3)、附录 A 与 C：全部 14,116 个 prompt 在训练前各生成一次，缓存响应与教师分数固定，当前 student 分数每步重算。补充“p0/q”表示生成来源二选一及 fresh 重采样与固定缓存的区别；明确本地 3,200 条是最初 50 步配对范围，长时重放不代表论文全量 prompt 覆盖。已有原进程保持运行。资源解释计入一次性缓存成本：训练循环约 5.73 倍，准备加首轮更新约 2.48 倍，均不称为含验证和初始化的端到端加速。

新增独立长时绘图脚本，固定日志前缀到第 99 步，仅画第 50 步已完成的完整验证。匹配第 1–49 与第 51–99 步相同 batch，49/49 批 RMS 下降，token 加权 RMS 0.49846→0.31001，平均梯度范数 1.10335→0.49712，逐批长度完全相同。PNG/PDF、绘图 CSV 与程序/样式/日志/输入输出哈希一同归档，图件审计无问题；准确率仅一个点，不宣称收敛。正文与图仍归统一实验报告。

按用户追问重新拆解首轮实际工作量：两组都生成 3,200 条，在线 12,407,645、离线 12,482,086 个响应 token，离线没有省掉首轮 rollout。冻结生成入口每卡提交 64 条，在线全局 batch 64 分四卡平均约 16 条；每卡离线 800 条分 13 批，对比在线 50 批。实际配置还存在生成 log-probability、模型长度/调度 token 上限与显存预算差异，因此不将全部吞吐收益因果归于 batch。在线最慢请求延迟累计 1,723.94 秒，占 gen 1,787.12 秒的 96.47%；权重同步累计 114.17 秒，训练前构造持久引擎，循环不反复从磁盘加载 checkpoint。

新增只读 CPU 核验脚本 eval/analyze_math_rollout_costs.py，读取日志中的实际配置、冻结生成器与 launcher、训练 CSV、schedule 与四个生成 shard，保留输入哈希与重复题目配对。首轮含准备的净差 1,401.05 秒，生成/准备差 1,246.18 秒、同步差 108.52 秒，其余 46.35 秒。3,200 个源索引和 ID 唯一，但只有 3,175 个不同 chat prompt，25 个内容各出现两次且不在同一 batch；PPO epoch 为 1，四组重复内容相同。补充论文成本表的 100 步不足完整一轮，以及前两个模型配对 4K/16K cap 的比较条件。结论和详细组成写回统一实验报告，正式长时训练与完整验证继续原配置。

为运行中的完整验证快照增加 W&B 显式 --snapshot 核验，从对应历史步骤读取全部精确指标，不用可被后续结果覆盖的 summary，并逐步核对 loss、grad_norm 与本地训练 CSV。默认核验仍要求 finished、完整步骤和最终 summary；阶段与完整运行的输入声明互斥。新增四项回归覆盖未来记录/summary 改写、非 macro 指标篡改、训练缺步和数值不一致等条件。实际长时 uq1riryx 第 50 步已远端核验通过，50 步 loss/grad_norm 与全部三基准精确指标一致，报告保留当时 running 状态、观察到的未来步骤及本地证据哈希。第 100 步验证仍在运行，未写出完整结果。

复核真实已完成在线组时确认，归档后的 W&B history 会为非验证行补上请求字段且值为 None，运行中 history 可直接省略这些行。核验按非空 macro 识别验证事件，不把补空行误算为额外验证；实际验证行的全部精确指标与两侧数值有限性继续检查。回归 fixture 同时返回补空的普通训练行，覆盖该归档行为。修复后用默认模式实际重核在线 g50xyfia，finished 状态、50 步本地/远端 loss 与梯度、验证点和最终 summary 全部通过；输出保存在 runs/rollout-sources/snapshot-audit-20261018-105935/final/wandb-verification.json，原归档未覆盖。长时第 50 步也重新核验通过；正式训练推进到第 100 步完整验证的 5/18 批。

按用户要求重构统一报告的显存与速度部分，集中列出四组实际生成后端、每卡请求数、引擎预算、训练/验证 cap、CPU offload、阶段时间、吞吐、actor allocator 与整卡采样；区分完整准备、训练循环、完整验证三个成本范围。核对四组实际配置与 engine_workers 源码，actor 参数/optimizer 及 teacher 参数均 offload，权重同步后 actor 参数移回 CPU；在线每卡约 16 条来自全局 64 条分四卡，不能归因于更新权重常驻 GPU 导致容量不足。Fusion 的 HF 双模型与双 StaticCache 不受原生 vLLM 0.55 预算控制，文档补充重复驻留的待测优化空间，没有修改正式训练行为。

新增 eval/plot_math_resources.py，读取已有 W&B GPU 0–3 采样并核对哈希，捕获每卡 81,559 MiB 容量记录，以 79.65 GiB 为参考绘制四组阶段耗时/显存总览，以及四张卡的显存与 GPU busy 时间线。两图均输出 PNG/PDF，逐张查看无重叠或裁切，机器审计均为空；保存分片绘图 CSV、逐卡阶段统计、实际设置与全部输入/输出哈希。整卡最高采样占容量分别为在线 57.47%、teacher 78.25%、Semi 59.55%、Fusion 74.45%；历史离线生成没有显存采样，不填零、不声称覆盖瞬时峰值。正式长时 Semi 原进程和两秒监控继续运行。

用户要求停止频繁盯跑、改为每小时检查。新增 eval/watch_math_progress.py，首次也等待 3,600 秒，之后每小时核对 PID 启动身份和完整日志；仅对具有原始响应文件及完整验证日志的候选步调用既有采集器，输出保存在忽略的 hourly-watch/，不修改正式训练。两个回归验证响应文件不能单独证明完成、部分写入的未来行不被计入。后台进程 PID 304315 已启动，状态文件位于 runs/rollout-sources/math-semi-opd-long-800steps-eval16-20261018-092912/hourly-watch/state.json；首次检查时间为 2026-10-18 13:16:37（Asia/Shanghai），启动核验确认 last_checked_at_utc 仍为 null。后续 agent 不再按分钟检查 Semi，按该小时状态记录处理结果；独立两秒 GPU 遥测继续用于曲线。

补充速度判断边界：同 checkpoint、实际 batch、长度分布、生成分数与推理执行路径一致时，纯 rollout 吞吐应接近；仅统一参数上限不等于实际 batch 一致。完整首轮还包含参数同步、阶段切换与缓存/初始化成本，两者总时间不保证相同。当前两组都重新做 teacher 评分，没有将尚未实现的 teacher 分数缓存归为当前收益。

按用户追问，核对 vLLM 0.10.0 的 GPU worker/model runner 与 HF 4.57.1 缓存说明，补充 prefill/decode、模型权重/活跃 KV/临时张量/引擎预留池的区别。活跃 batch 变小不保证 vLLM 固定池在 nvidia-smi 中变小，也不会自动把空间让给另一模型。按本地 28 层、2 KV head、128 head dim 与 BF16 推导每序列每 token 28 KiB，给出固定全局 batch 与固定每卡 batch 的多卡 KV 表和稳态吞吐近似，明确不是实测加速。内容继续归统一报告。仅终止等待下次时点的会话内计时 cell 724，以便响应用户问题；后台每小时检查 PID 304315 和 GPU 遥测均未停止，没有提前读取 Semi 进度。

按用户要求全面审阅实时 rollout 重构，沿当前 math 的 recipe→main_ppo_v0→RayPPOTrainer→actor/rollout worker→IPC/vLLM/HF decoder→teacher scoring→actor loss→student-only 验证逐段核对，涉及 32 个文件；范围与 SHA 保存在 docs/experiment/results/2026-10-18-rollout-refactor-code-audit.json，源码基线 d17d6c6。13 项源码观察包括 actor/rollout 共用 model config、actor 流覆盖 fusion base、固定模型不适用 step-1 版本约定、sleep level 2 的恢复来源、KV-only 方法空实现、原生与 HF 双参数写入、旧 HF 类不在 async registry、旧 50 批数据与平台判据耦合。没有把整个上游所有后端称为已审阅范围。

在既有 docs/architecture.md 写入拟议方案，并从统一实验报告链接：ModelRef、RolloutPolicySpec、RolloutCoordinator 与 GenerationBackend 支持三种单模型和两种 shared-prefix fusion；训练时实时生成与 on-policy 语义区分，初始 student fresh 不冒称固定缓存 Semi。保留现有 actor 更新、teacher 评分和完整 eval16；建议先用同一 HF 单/双模型内核比较五种来源，再独立接入 vLLM 单模型性能路径。完整数据清单与预算分开，统一轨迹、mask、行为概率诊断和原子参数版本，控制 CUDA Graph 地址与 KV 生命周期。首版明确成功/否决条件、逐文件归属、两步 GPU 与后续 50 步验收，不把提案当作新功能已通过。

此次只修改设计与代码核验记录，没有切换正式生成路径或改变运行中训练。终止的仅是会话内等待 cell 734，以便响应代码审阅请求；每小时后台观察和独立 GPU 遥测继续保留。

用户扩展需求：同一响应 teacher→student 或 student→teacher 的分段切换，batch 内按比例分配来源，并要求尽量使用 vLLM。据此修订既有架构提案，替换上一版 HF-first 次序：原生单模型、双向分段、batch 混合优先接入，精确 fusion 优先验证原生双模型扩展；HF decoder 保留为数值参考与历史复现，不再作为新 pipeline 的默认执行后端。参数分别为 λ、响应 token 切换位置/总预算比例、全局 batch 配额，可组合但按后端已验收能力开放。默认两段都训练 student，EOS 结束整条响应，切换不重套模板，总 response cap 不翻倍；明确 batch 比例与实际 token 比例不同。

补充核对安装的 vLLM 0.10.0 V1，15 个相关源码文件的范围与哈希追加到原审阅 JSON。确认请求级自定义 logits_processors 被拒绝，worker extension 是带名称冲突检查的 RPC 扩展；仓外模型注册可作为原生 paired wrapper 的入口。KV 规格来自已注册 attention，但两个模型相同整数层号会在 bind_kv_cache 触发 NotImplementedError，需全局唯一缓存层编号或局部适配，不能只加 branch 前缀。双 hidden-state、profile、参数加载和 graph 仍待真实验证，未声称 wrapper 已可运行。

分段接口审阅还确认本仓 generate 会原地 pop 采样参数、把显式 0 预算提升为 1，并把 stop/length 都折为 completed；新协调器须独立参数副本、预检阶段预算并保留原始结束原因。跨模型不能沿用 KV，后段要按自己的权重重新 prefill 原 prompt 和前段响应，成本在表与公式中单列。多引擎与单引擎整组换权重均保留为基准候选；sleep 每进程仅支持一实例，双引擎需要独立 worker、联合预算和 Ray 资源归属。按来源聚合、小分组集中、连续批处理、chunked prefill、graph 与有界队列按依赖验收，不预报加速倍数。

设计继续放 docs/architecture.md，统一实验报告仅更新链接与当前推荐，未增加实验正文文件。没有修改训练实现、共享环境或运行中任务，也没有提前读取长时 Semi 进度。

设计核验：32 个项目源码和 15 个环境源码哈希均匹配，22 项观察定位仍有效；两段拟议 YAML 可解析，架构正文 57 个数学表达式通过 KaTeX。docs 模式、本地链接、研究正文审计、Markdown lint、Prettier、中文间距与 diff 检查通过。没有为文档变更重跑 GPU 实验，以上检查不证明新增生成策略已可运行。

按用户要求调研多模型融合推理，复核当前 math_fusion→trainer→worker/RPC→HF decoder→teacher 评分/student 更新的链路，在架构说明补充实际计算后端、两模型共同 token 流、独立 KV、prefill eager、仅 decode 前向入 graph、图外采样、StaticCache 无连续补槽、端点仍双前向与参数双写。明确训练对象是单独 actor，生成 student 只接收参数副本，验证始终单独 student。温度 1 下 logit fusion 对应有效词表分布的归一化几何平均，概率平均不同；未改变正式运行或历史指标。

外部实现审阅固定到 vllm-logits a987755（2026-08-18）和 CoS b7c5553（2025-06-16），12 个文件及其哈希追加到既有代码审阅 JSON。前者在一个 vLLM 模型内运行两个 native backbone、两个 LM head，最接近拟议 paired wrapper，但依赖 0.15.x、模式取进程环境变量、默认 protected-token 覆盖和短同权重 canary 均不能直接用于本项目。CoS 基于 0.6.5 安装路径，有 ensemble worker/sampler 与融合目标拒绝采样；默认 we 配方混概率并返回第一模型 log-prob，不能直接作为本项目 raw-logit fusion 与行为分数。

执行外部 compute_logits 的 AST 提取方法作 CPU 算术核验，使用两个三维合成 logits、identity head、λ=0.5。无 protected token 时等于当前 FP32 加权与归一化几何平均；概率平均和 protected-token 覆盖相对目标分布的 TV 分别为 0.13520、0.26861。该核验没有建立 vLLM 引擎、读取 checkpoint 或调用 CUDA，不证明外部运行时正确。UltraFuser 附录 C 的多实例逐步交互与 ThinkMerge §4.1 的多上下文合并仅作为论文层参考，未称为已核验可运行的双 checkpoint 源码。

方案继续以原生非 speculative 的精确 wrapper 为首个验收目标，参考仓外注册、双 hidden/LM head、权重映射和 KV 总预算；CoS 在保持同一融合目标分布、正确拒绝采样与 KV 回退后再做加速基准。原有“普通 speculative 不是 fusion”的限制保留，同时明确融合目标 CoS 属于可验证的后续优化。调研及取舍仍归 docs/architecture.md，未另建实验文档，外部缓存位于忽略的 runs/code-review/。

调研交付核验：59 个项目、环境和外部固定提交的文件哈希匹配，架构正文 70 个数学表达式通过 KaTeX，docs 模式与 Markdown/Prettier 检查通过；中文间距检查指出的一处混合拼接表达已改为中文。没有安装外部依赖或运行其 GPU canary，新增后端仍属于待实施方案。

用户要求继续做真实实验并把提案归 Agent Note。已将 architecture.md 中全部拟议重构与外部实现调研迁入 .agents/notes/proposed/architecture/2026-10-18-可组合rollout与原生vLLM推理.md，保留 proposed 状态；架构页只保留已实现行为和入口链接，统一实验报告改链。补查 Relay-OPD eab21451、Prefix-RFT 5f5d2a1f、TRL GKD 6d6901c5 与 LUFFY 01f4ae06，8 个源码记录写入 docs/experiment/results/rollout-composition/review.json，区分动态接管、离线前缀、整批随机选择与同批精确配额。

新增 eval/rollout_composition.py 独立生成原型及 eval/probe_vllm_composition.py 原生 GPU 验证入口，尚未接入正式 trainer。CPU 回归涵盖 500 组随机配额、864 组 EOS/边界组合、身份/来源/预算/顺序与并发失败；检查已加入 scripts/checks.json。真实验证仅在 GPU 0 的独立进程组运行两个 0.10 预算的 native vLLM，不改共享环境或现有训练；prefix cache 关闭，固定 checkpoint 只做推理。每次退出均核对 GPU 回到原 48,503 MiB 基线。

三项真实边界促成修正：默认 V1 多进程相同 greedy 请求仅 9/16 条逐 token 一致，官方要求关闭多进程固定调度，因此把 correctness 与默认调度 throughput 模式分开；EOS 恰好达到阶段 cap 时原生优先报告 length，原型改为保留原 reason 并以实际 EOS 终止整条响应；reset_prefix_cache 的公开 bool 注解与 V1 实际 None 返回不一致，原生换权重验证取消 bool 断言，增加 idle 检查和 CPU 回归。APC 关闭的测试不被解释为 APC 开启时的缓存失效验收。

初始尝试 runs/composition-probe-20261018-1350、correctness-v2、correctness-v3 的原始调用、metadata、源码快照与失败证据均保留；v3 已完成两方向分段、比例/嵌套、EOS、固定 seed、4096 长响应、吞吐扫描和 sleep/wake 后，在权重切换的返回值断言处退出。correctness-v4 正在重跑完整矩阵，加入逐 case JSONL 和 checks.json 原子记录，结束后还需跑默认调度吞吐、生成紧凑结果表、补 Note 的实测结论并最终检查提交。运行中的 harness/core 不再编辑；旧生产 Semi 进程与每小时观察不受代码变更影响。

完整实测结束：correctness-v4 完成 71 个场景、774 条场景响应、135,668 个输出 token，35 项契约/恢复检查通过；原生同模型分段与一次到底的位等价负对照为 14/16，严格 all_checks_passed=false 保留。独立 numerics 原生 Top-5 复测重现相同的响应 109/126 位置分歧，同一 token 前缀下候选 log-odds 在并列与 0.125 之间变化；没有将 Top-5 当作完整分布误差上界。新增接受这项观察的显式投影参数，负测试证明不会放过错误段落或其他失败。

默认多进程 throughput 完成 38 场景、316 条场景响应、39,968 个输出 token。B16 的单模型/串行混合/同卡并发混合/分段均值为 2.206/3.400/1.975/3.199 秒；B2/8/16 各 3 次并发相对串行快约 1.64–1.72 倍，限定共享 GPU、128 token 与本轮布局，不外推独占卡或训练吞吐。原生同引擎 q↔p0 两轮每次覆盖 199 参数并匹配独立引擎，缓存热激活约 0.62/0.67 秒；两引擎 sleep/wake 后输出恢复。整卡基线/最高采样为 48,503/66,540 MiB，所有自有进程退出后回到基线，共卡区间归档避免把测试负载算作原长时实验峰值。

汇总与源码/环境/CPU 反例保存在 docs/experiment/results/rollout-composition/，Agent Note 写入实测设置、3×4 吞吐表、否决的位等价假设与后续四卡/FSDP/APC/Graph 验收；没有新增实验正文 Markdown。18 项新增 CPU 回归与包含 12 项原融合检查的 30 项产品 CPU 测试通过，完整 CI 通过。正式 trainer 与在线 actor 参数热同步仍未接入，Note 继续为 proposed；没有改变原训练算法、保存 checkpoint 或原地升级环境。

2026-10-18 14:58（Asia/Shanghai）按用户明确要求停止长时 Semi-OPD。先核对队列、训练、Ray 子进程与观察器的 PID 启动身份，暂停调度父进程防止继续排队，再向训练 driver 与 TaskRunner 发送 SIGINT；其专属 Ray、W&B 子进程随之退出。清理队列与两个专属监控后，原进程树无活跃残留，无需 SIGKILL；物理 GPU 0–3 各剩 1 MiB，GPU 4–7 的其他作业继续运行。原始进程树、停止回执与观察器终止状态保存在忽略的运行目录，不安排自动重启。

停止时已进入第 200 步验证。核对冻结源码的执行顺序，actor 更新在验证之前、指标日志在验证之后，因此完成 200 次更新、四轮缓存消费，但只有第 1–199 步完整训练指标。第 200 步无验证 JSONL；独立采集第 50/100/150 步，每次 570 题、9,120 条响应，macro mean@16 分别为 64.8028%/66.9215%/67.3424%，最后一步 micro 为 82.2697%。三个完整验证点的全部精确指标与 W&B history 一致。远端第 1–198 步 loss/grad_norm 与本地相同，第 199 步仅本地保留；使用 SDK 允许的 running→failed 转换标记人工中断，没有将其标为成功完成。

完整训练 CSV、停止证据、最近平台记录与远端覆盖核验归统一实验结果目录 `results/long-semi-progress/stopped/`，完整三次评测归 `step-150/`。重画长时曲线到第 199 步，完整准确率点到第 150 步，绘图机器审计为空。统一实验报告更新结果表、共同 49 批的四轮 RMS/梯度、阶段时间和停止状态；199 条日志中的训练/完整验证分别为 29.03/296.49 分钟，验证占 91.08%。保留 GPU 0 独立原型并发负载的限制，不据此声称独占吞吐或已达到平台。历史第 50 步快照和原四组结果保持原身份。

停止归档完成后，全量 CI 检查通过，包括 43 项维护测试、18 项 rollout composition CPU 测试、训练源码语法、研究公式/文档、格式和链接检查。另逐项核对三次完整 eval16、199 步连续日志、远端证据输入哈希、绘图输入/输出哈希和题目 CSV 哈希。没有恢复训练或新增 GPU 实验。

按用户澄清，方案解释改为围绕 state 来源展开：定义每个动作之前的 token prefix，分别说明 current/initial/teacher 单模型、两种 logit fusion、四个 teacher/base 分段方向、current/initial batch 混合、分段与 batch 的嵌套，以及 fresh 与固定缓存 replay 的区别。共同训练阶段明确始终只更新当前 student，第一段和 EOS 的有效响应位置均参与训练，切换不重置已有前缀。修正 batch 配额表述，与原型的全局最大余数法和组件顺序平局规则一致；参数 λ、k/ρ、α 分别控制融合、分段和响应配额。

原 Agent Note 的对应小节新增八幅 Mermaid，保留现有总览，共九幅；使用独立忽略目录中的 Mermaid CLI 11.12.0 与已有 Chromium 实际渲染，并查看九图总览及分段边界图。渲染、输入 mmd 和配置保存在 runs/state-variant-diagrams-20261018/，未给项目添加运行依赖。方案继续为 proposed，区分独立原型、现有 HF fusion 和待接入的正式 trainer；实现与研究架构文档均未修改，原训练保持停止。

九幅当前 Mermaid 源码与渲染结果已逐一核对并记录哈希。文档/Agent Note 检查、本地链接检查、Markdown lint、Prettier 和正文间距检查均通过；只提交该方案与既有工作记录，不纳入其他并行修改。

用户要求暂不考虑两类嵌套组合，首轮范围据此收紧为单模型、独立 fusion、单模型间的一次分段和单模型间的简单 batch 混合。移除嵌套配置示意与流程图，正式参数编译拒绝 batch 内使用 switch/fusion 或分段中使用 fusion；原型已做过的嵌套测试与旧测量保留为历史证据，不扩展接入要求，也不改动已运行的训练条件。

按 huggingface-papers 技能获取 2604.24005 的元数据与正文，并核对 arXiv 最新 v3 的 §3–5、附录 C/D 及官方 kokolerk/TCOD 提交 465eef4406ad0cff675b36bd46f37f28b1736ff9。F2B 仅放宽 student 的环境交互轮数，不包含 teacher 后半段续写；B2F 回放 expert 环境动作后让 student 接管，前缀不进入直接蒸馏目标。公开代码与论文概念分开说明：B2F 后段循环可达全局 max_env_steps，ScienceWorld 用环境 gold actions，ALFWorld 接管上下文有结构化历史格式，advantage 使用 sampled log-ratio 加默认 PPO loss，不宣称精确全词表 forward KL。

在忽略目录运行原始 ALFWorld workflow AST 的十个纯 CPU 替身检查，确认 prefix 不产生训练行、student-only 评测、课程进度、缺 expert 回退和提前完成；六轮 expert/三十轮上限的反例实际返回五轮回放与二十五条 student 训练行。没有运行真实环境、模型推理或复现论文成功率。核验输入、源码 SHA 和结果保存为统一实验结果目录下的 rollout-composition/tcod-review.json；测试脚本与外部源码保留在 runs/paper-reviews/2604.24005/。

同一 Agent Note 增加 TCOD 调研、两种 Mermaid 流程与迁移边界；全篇十幅当前 Mermaid 均实际渲染通过并查看新图。该方向仅作为调研候选，未自动加入 math 首轮实验矩阵；原方案的双段全响应训练语义与 TCOD 的前缀条件化、仅后段训练明确分开，后续比较须单列 loss 范围。

TCOD 控制流检查十项通过，当前十幅 Mermaid 均通过真实渲染与输入一致性核对；正式文档、Agent Note、本地链接、Markdown lint、Prettier 和正文间距检查通过。该变更只提交原方案、既有工作记录和新增 TCOD 核验 JSON，其他并行实现与研究修改保持原状。

开始实施可组合原生 rollout 方案，范围以 b6edee4 及用户排除 TCOD/嵌套的指令为准。阶段一统一三种单模型的模型引用、平面策略、轨迹契约与生命周期；阶段二接入单次分段和简单 batch 混合；阶段三单独实现原生双模型 fusion。成功条件包括真实 actor 热同步、固定引用不变、全局配额、EOS/总预算/mask、student-only 评测及真实两步训练；数值与吞吐另建同条件基线，不以 CPU 检查替代 GPU 集成。任何固定权重被覆盖、跨权重复用 KV、策略错配、隐藏 HF 回退、未排空请求就换权重均拒绝进入正式实验。引擎布局、固定模型驻留、批分发与 graph 优化仍由基准决定。

启动现场物理 GPU 0–3 仅占约 4 MiB，4–7 有其他作业。现工作区已有未提交的 KL 对齐与研究正文修改，保持其文件和行为，不将它们纳入重构提交。首个实现提交先处理独立策略模块及 CPU 合约检查，避免把生成重构与损失实现合并。

阶段一契约实现完成：生产 policy.py 与默认关闭的 rollout 配置，明确 current actor 版本与 frozen fingerprint 的不同身份，编译 global exact-count 路由、k/ρ 总预算和 λ 端点，拒绝嵌套、TCOD、未知字段、非有限参数。十三项 CPU 检查与五百组配额属性通过；Hydra 真正带 `_target_` 的配置往返通过。首次直接从空 `_target_` 的默认结构构造 Hydra 测试未通过，改用真实目标配置核验，未据此前失败修改模型行为。CI 增加 native-rollout-tests 门控。

实施前原生基准三十八场景完成，三十六个热调用对照各三次。B16 单模型/串行混合/并发混合/分段均值以正式 JSON 为准；两个引擎冷启动约 21.5/22.3 秒。实际启动时 GPU0 另有 6,921 MiB 占用，因此保留 shared_gpu_load 标记，不解释为独占性能；原始轨迹与采样保存在忽略的 baseline-native-throughput/，压缩设置和耗时归统一实验结果目录 native-rollout/baseline-throughput.json。

下一步原生模型池优先检验独立进程的固定来源引擎与当前 actor 引擎，统一联合显存预算并在评分/更新前休眠；按源聚合请求，避免每 token RPC。固定模型与 actor 的生成存储隔离，teacher/p0 不接受 actor 参数流，验证单独绑定 current。单引擎热换权重保留为基准候选，不以易实现代替联合显存与吞吐比较。工作区另一组 KL 四卡短测已启动，不终止或重启它；自己的真实训练在现场确认空闲后运行。

阶段一已提交 7f4cb7c，隔离 worktree 的全量 CI 与两阶段 hooks 通过；没有暂存或恢复并行 KL 工作区。随后该工作以 60fe096 独立提交，继续基于当前实际源码接入生成接口。生产 native_executor/lifecycle 与异步服务接口新增检查，四十四项 CPU 合约和七项既有 vLLM 生命周期回归通过；包含真实结束原因、EOS=PAD、累计上下文、整批失败排空、冻结写保护及 actor 写入回执。按完整 HF safetensors 名称集合验证 339 个条目，不以 native 打包后的参数个数替代。

补齐同版本恢复的原子性：只比较 actor 版本会误接受 sleep 丢弃权重前的旧回执，改为在 begin 读取实际 worker 写入序号，commit 要求成功完整流推进序号、版本匹配、缓存失效与 CUDA barrier。权重流缺最终 bucket、缺项、重复、未知名称均拒绝发布。新增 preserve_source_weights 以保留后续 paired 引擎的冻结 companion，状态以实际 sleep level 标记；旧配置不进入新协议。

GPU 身份检查首轮在初始 student 上全部功能断言通过，但 in-process CUDA allocator 退出析构失败，整轮仍记为失败。换为生产默认多进程后，发现函数对象 RPC 不可序列化，子进程又在 multiprocessing finalizer 中等待 idle EngineCore，掩盖原始异常。py-spy 定位后仅终止自有 Core 以暴露异常；改用按方法名 RPC、立即打印错误及显式关闭自有 EngineCore。没有设置 insecure serialization 或更改共享环境。

identity-gpu-v3 与最终 v4 均完整通过 p0/q 原生 worker 和生产 executor 检查：冻结来源拒绝真实 actor 写入 RPC，生成前后与 sleep/wake 后 greedy 响应一致，EOS=151643 与 PAD 同 ID 时 mask 仍为一，原生 finish_reason=length 保留。两个引擎各释放 8,090 MiB，进程退出后 GPU0 回到 4 MiB。v3 源码快照留在忽略目录，v4 基于当前写入序号实现重新验收；正式证据归 identity-gpu.json。此时仍未接通原生模型池、协调器、真实 FSDP 更新和 native paired fusion，未将 worker 检查当作 trainer 验收。

临时 trainer 门控对 policy 非空报未接入，防止用户配置 teacher/initial 后实际仍从 actor 生成；下一阶段以真实 coordinator 替换。完整目标和所有待完成门控存于 runs/native-rollout-implementation/state.json，仍包括四卡两步恢复、完整 native fusion 数值/graph/缓存检查、公平吞吐与显存基准、独立数据清单及正式 50 步 math eval16。TCOD 与嵌套保持排除，没有缩小原实施目标。

17:24 完成 trainer 原生协调器与模型池的首个真实四卡验收。当前 student 两步生成使用版本 0/1，最终六题各十六次只用版本 2；独立采集器重算九十六条 verifier 结果并逐项匹配日志，未保存 checkpoint。全程 182.40 秒，两个训练 step 为 22.31/5.71 秒，短验证 11.90 秒；整卡每两秒采样峰值约 17 GiB，进程结束四卡回到 1 MiB。已有四十七项原生 CPU 与五项训练环境桥接检查通过；metadata-only DataProto 的两次接入错误由真实运行和更新后的回归用例定位修复。一次已知错误的自有 canary 在更新前停止，仅清理其已核对的子进程。

新增每次运行独立的 ZMQ 命名空间，区分 job ID 相同的独立 Ray 集群；生成按源/副本批量 RPC，固定来源与 actor 存储隔离，验证同步最新 student。运行源码为 d187a4b 基线加完整文件哈希快照，不能按基线提交冒称全部改动已入库。teacher/initial、分段、batch、graph、原生排空和正式质量矩阵仍未完成。原生双 Qwen2 模型已写出待验证实现，注意力全局层号与两套 KV 分离；其 GPU 数值检查正在独立进行，尚未接入 trainer。

原生协调器阶段提交为 4994d2b，隔离 worktree 的完整 CI 与提交 hooks 均通过。teacher-fresh 随后四卡两步完成并独立核对九十六条最终 student 响应：两步生成固定 q，训练不向生成端同步 actor，最终验证是版本 2 的 student。全程 224.95 秒，第二步生成 0.668 秒、训练 3.539 秒，整卡采样最大 17,889 MiB。initial-student 的独立四卡 canary 已启动，仍待结果。

双原生 Qwen2 的独立 GPU 数值检查通过 λ=0/0.5/1、两条短 prefix 和完整 151,665 个有效 token 的 logits/分布对照，六项误差均为零；λ=0.5 的 56 个 KV 区域互不重叠，两端点各 28 个，sleep/wake 后逐值一致。原始浮点数组留在忽略目录，摘要及数组哈希归统一实验附件。它只覆盖单 token prefill；下一阶段仍须连续 decode、更新时 teacher 不变、缓存失效、graph 与真实 fusion trainer 验收。

initial-student 四卡两步已完成并独立核对最终九十六条响应，固定 p0 与 learner 分离且没有训练同步，验证是最终 student 版本 2。全程 251.83 秒，两步为 22.54/4.85 秒，短验证 16.14 秒，整卡采样最大 18,051 MiB。四种分段方向和两种 batch 混合组成独立串行矩阵，固定源码 4994d2b；切换点为 32/64 token，current batch 的 teacher 占比为 1/8，刻意覆盖部分推理副本无该来源请求。启动器现在支持提交快照，并保存影响策略的环境参数；任意训练或审计失败即停止后续 case。

分段/batch 六项矩阵均已完成并独立核验。current fusion 的单卡调试与两种 fusion 的四卡两步也已通过：current 版本 0/1 生成，最终仅用版本 2 的纯 student；teacher 全参数摘要跨步骤/副本一致，initial fusion 还核对 initial 全参数摘要一致。单卡调试与矩阵第四项启动重叠，干扰 vLLM 显存探测，失败项只到初始化；确认退出后串行原参数重跑通过，后续 GPU 工作改成串行，未修改共享环境。

模型池进一步区分 training 与 evaluation 用途，current 训练也建立专用副本。训练上下文统一为 prompt+response，来源共享预算，所有 case 的最终纯 student 验证使用相同完整预算；actor 导出前核对用途/角色/路径/命名空间/companion。fusion actor loader 与 post-load 变换只处理 base，诊断开关校验完整冻结参数，graph 模式额外核对参数地址；graph 实测仍在进行。

完整 DAPO 题库准备与真实 RLHFDataset/sampler 核验完成：17,917 行，8 行超长，662 条额外重复记录，12 个标签冲突的唯一 prompt 排除后为 17,235 题。batch 64 每 epoch 269 批、drop-last 19 行，前 50 批 3,200 题互异，题库没有按步数截断。访问次数改按内容累计，原始行号另存；正式配方可以绑定 manifest 校验文件/词表/模板/上限。原生路径显式拒绝缺少已验证行为分数的无偏 top-k tail 组合，保留历史入口。

连续解码数值检查覆盖 λ=0/0.5/1、两条 prompt 各八步、完整有效词表及 sleep/wake。current 单模型跨 HF/native 的最大 logits 差为 0.125、TV 为 0.027081，其余两项误差为零，均满足预先采用的 BF16 界限；原生自身恢复前后逐值一致。CUDA Graph 检查只使用已空闲的 GPU 0，等待真实捕获计数和输出验证，不以配置开关当作通过。

CUDA Graph 连续解码随后通过：λ=0.5、两条 prompt 各八步，实际捕获 114 个图，十六个位置的完整 logits 与 HF 参考相同，固定参数和地址跨 sleep/wake 不变。新增重复同一道真实 DAPO 题十六行的专用 fixture，正在执行四卡两步 graph 训练：覆盖相同 prefix 的跨 actor 更新缓存失效，要求 q 完整摘要不变，参数地址不得改变。训练源码冻结在私有快照中，未重启其他作业。

四卡重复题目 graph 训练完整通过并独立采集，最终九十六条 student 响应、current0/1→纯 student2、teacher 全摘要和参数地址检查均通过。实现阶段提交为 7d8e261，完整隔离 CI 与提交 hooks 通过，保留并行研究文档提交。

真实 eager native 检查先暴露 APC 的计算形状导致非逐值 logits 差异，保存全词表数据后按既定 BF16 界限核对：最大差 0.15625、TV 1.293e-5，命中 128/80。随后实际 abort 触发输出协程不唤醒、排空超时，按 vLLM0.10 代码和真实现象修复生成 task 的登记/取消/等待；eager v3 与 graph v1 均在 decode 后成功终止，active/frontend 均清空，p0→q→p0 的完整 IPC、APC 失效和恢复 logits 通过。补充超时覆盖整个 abort/barrier/reset、drain-pending禁止复用及GPU变更前空闲检查，当前57项CPU检查通过。早期GPU检查未冻结源码，附件明确区分采集时源码哈希与后续源码快照。

性能基准采用生产 RPC/executor，固定长度工作负载只在测时屏蔽 EOS，真实 math 样本另保留 EOS。首轮并行冷启动撞到 vLLM TCPStore 临时端口，确认失败并清理后改成串行初始化，稳态请求仍并行。正式 50 步/570 题×16 的完整配置及 17235 题 manifest CPU 预检通过，尚未将预检当作正式训练。

排空修复提交为 46a928f，完整隔离 CI 与提交 hooks 通过。四卡 graph overlap 基准完成 36 次生成，所有固定长度输出数与预算核验通过，B64 单模型约 16.35k token/s，batch约8.91k，fusion约9.36k；分卡、单副本、最新源码冻结后的graph运行检查与完整manifest两步长响应检查进入串行门控。新增十五case正式调度器与完整eval CLI，固定 50 步/B64/4K 训练/570×16 最终评测，任一训练或审计失败则停下后续 case；目前不将该调度器视为已经取得正式结果。

后续串行门控全部通过：分卡四副本、单卡重叠、冻结源码 graph 排空，以及完整 17,235 题清单上的 B64/4K 两步训练。最后一项最终验证为六题各十六次，仅验收链路；结果已独立采集。正式十五组矩阵固定 d59a7c3 启动，首组 student_fresh 的四卡进程实际存活且持续完成更新，W&B 为 0hbo3l5t。矩阵不读取开发中的 verl 工作区，后续优化须单独验证。

根据推理基准补入生产 `generation_layout=split` 和小分组集中：保持默认 overlap、来源 KV 预算和四卡 FSDP 发布，改变推理请求的副本映射；按需恢复冻结副本，未使用的 current 副本休眠后拒绝直接唤醒。每批审计保存实际阶段映射。六十六项原生 CPU 检查与十三项训练环境检查通过，包括五百组随机配额、真实后端分片、顺序恢复、空副本和错误索引；新的四卡布局训练尚未运行，等待正式作业释放资源。预算尚未扩大，不能把此前分卡微基准的速度当作训练提速。

布局阶段提交为 6d079dd，隔离完整 CI 和 hooks 均通过。新增原生轨迹前缀诊断，复用历史精确计数并按 prompt/visit/sample、实际 token 与训练步骤配对；四项检查覆盖显式 state 集合、访问身份、不完整文件、EOS 和分段深度。真实两步 canary 与正式首组前两步共 128 对：第一步 64 条完全相同，第二步无完全相同响应，pooled state Jaccard 为 2.129%。两者源码快照不同，不以此判断分叉原因或来源算法排序；输入和源码哈希已归统一报告附件。正式首组仍在训练，未修改或重启。

前缀分析阶段提交为 41c1f14。随后补入共同评分后的来源分层诊断：紧凑分段元数据随 DataProto 行重排，复用现成 student/teacher log-probability，按来源、段和绝对深度记录 log-ratio、RMS、所选 action 概率及前缀深度，不改 reward。十八项训练环境检查和六十六项原生合约检查通过；真实 B64 长度配合合成概率的单线程 CPU 计量均值 0.0569 s，不能作为真实 GPU 指标。正式首组已写出五十批轨迹并进入 global_steps=50 的最终评测；完整 run 及 W&B 审计仍待结束。

分层诊断提交为 290f822，完整隔离 CI 与 hooks 通过。随后为新布局增加实际 worker 的 CUDA UUID/进程/副本执行记录和独立审计器，拒绝错卡、重复请求、短分片、缺失记录及统计归约错误；六十八项原生、十八项运行和五项审计检查通过。后续计划为六组两步正确性与四组五步 ABBA 布局观测，速度阶段不启用完整冻结摘要，全部等待 d59a7c3 正式矩阵释放 GPU。已运行矩阵继续固定旧源码，不包含后加的执行记录与分层指标。

物理执行与后续验收提交为 7896db4，完整 CI 通过。排队进程 1466692 固定该提交，等待现有十五组正式矩阵全部训练与审计成功后再使用空闲 GPU 0–3。检查独立微基准时补齐来源局部序号与物理卡号的接口适配，并加入可冻结提交、调度容量、chunked prefill 和整卡显存记录；八项 CPU 检查与历史 28 条件/19 个引擎记录回归通过。十五条件扫描器已定义，GPU 扫描和单引擎换权重比较仍未完成，没有把计划或历史结果当作新版本验收。

基准控制提交为 0935f75，完整 CI 通过；扫描排队进程 1528770 等待上述后续验收成功后再执行十五条件扫描。首组正式 student_fresh 完成，完整 9,120 条最终响应、student50、五十次更新、无 checkpoint 和 W&B finished 均已独立核对；macro mean@16 为 64.2965%。总计时 92.16 min，训练 step timer 24.65 min，最终评测 63.74 min；最大整卡采样为 50,759 MiB。

补查精确规范化题面重合：全训练池对 MATH-500/AMC23/AIME25 分别为 4/5/0 题，首组实际消费为 3/0/0。已把 canonical 评测输入/GTS 的 570 个身份与保存结果逐项核对；排除实际消费后 macro 为 64.2989%，排除全池重合后为 64.1634%。数据与完整评测协议没有中途修改；新的独立采集器在后续完整运行附加同一数据定义下的敏感性结果，三项 CPU 检查覆盖规范化、消费身份和 macro 归约。第二组 initial_student_fresh 正在使用原冻结源码运行。

验证指标方案先提交为 4737ade。随后根据用户要求，将主 KL 方案改为全量复用 570×16 响应、对所有有效位置评分；不再为主指标增加 180 条 student rollout。固定 initial/teacher 前缀 bank 保留为共同状态对照，拟从各基准三十题各来源一条响应构造，最多选八个固定位置；该子采样不缩减主验证范围。完整词表为默认口径，top-k 仅作辅助，原准确率采样设置与已冻结矩阵保持原值。

开发源码增加验证长度、终止、解析、多样性和重复统计，DataProto 中保存原始结束原因与 final-student 版本。新增数值函数以 FP32 计算双向完整条件 KL 和 entropy，CPU 小序列枚举核对 sequence KL 的链式分解；三种词表分块与独立 FP64 参考比较。最初逐块串行 logaddexp/累加导致 entropy 误差约 3.5e-5，超过预定 1e-5；改为统一归约块摘要后最大误差降至约 8.35e-6，没有放宽阈值。位置抽样用穷举区间内位置的方式核验加权估计无偏。

七十七项原生 CPU 与二十六项训练环境运行检查通过；实际 `_validate` 和合并方法覆盖不等大 batch、DP padding、响应重排和 EOS=PAD，原准确率返回值与字段仍保留。新检查脚本及证据归统一实验目录，KL 评分 worker 和全量验证 KL hook 尚未接通，GPU 验收未通过前不把数值构造当成真实模型诊断。现有正式第二组、后续布局与扫描 watcher 继续使用原冻结源码。

验证指标核心提交为 a0eb8d3。随后接通可选的完整 KL 评分 worker 和 `_validate` hook，整轮保留有效原始 token，推理池休眠后读取同卡当前 actor 与冻结 teacher，不再额外生成 student 响应。原 colocated worker factory 补齐已有 fused-peer API；评分复用正常 FSDP forward，通过临时 head hook 避免整段词表张量，再按位置计算两个 head，主权重需要转换时每个快照/响应只做一次。保存输入哈希、student 版本、模型绑定的固定 bank、逐响应/分层统计及调用计时。

七十七项原生 CPU 与四十二项训练环境运行检查通过，新增小型 Qwen2 直接前向对照、DP 补齐、原采集器兼容性和异常恢复。两进程 Gloo/FSDP 实验在各 rank 不同长度下完成评分、真实 SGD 更新及再次评分，teacher 本地参数逐项保持，student 结果随更新改变；这不是真实 checkpoint 或 GPU 证明。原准确率采集器将顶层 JSONL stem 解析为 step，故新增诊断归 validation/kl，已用真实 producer/collector 组合回归。GPU 与完整 bank 生成仍待完成，全部正式实验继续使用原冻结源码。

完整 KL 可选评分链路提交为 f1bcea0，所有 CPU、两进程 CPU FSDP、隔离 CI 和 hooks 通过。第二组 initial_student_fresh 也完成正式训练/评测：macro mean@16 为 64.16597%，与在线 student 的差为 −0.13056 个百分点；5,000 次分层配对题目 bootstrap 的 95% 区间为 [−1.69516, 1.33616] 个百分点，不能覆盖训练 seed 不确定性。两组原生生成吞吐分别为 14,978.74/14,977.73 token/s；训练 step 净差 161.73 s，权重发布差 121.30 s，约占 75%，其余评分/更新差异仍待归因。

远端 W&B 核验 pl0yl6nt 与 0hbo3l5t 均 finished、五十条训练记录、仅 step50 验证。原通用 verifier 在缺少 legacy rollout_source 时将原生来源标成 student_online；现改为显式保存原生 policy/backend，并校验远端 policy 与已审计的本地 native manifest 哈希，相关匹配及篡改反例通过。它只修正核验元数据，训练源码仍固定 d59a7c3。第三组 teacher_fresh 已运行，现有调度进程保持原序列。

正式前两组的远端 policy 绑定与质量/计时对照提交为 15765ce。检查排队后续任务时，命令实际传入了完整 570 题文件，与原定六题×16 的工程验收不符；新增确定性两题/基准的 Parquet 准备器和无 PyArrow 的哈希校验器，保留 prompt、答案、schema 及原文件。控制入口固定 n=16，保存准备计划与 manifest 哈希，每个 case 开始前复核文件，最终审计要求恰好九十六条响应。真实源文件的行数为 500/40/30，输出各两行；三项 Parquet/篡改检查与五项布局/计划检查通过。已启动的正式矩阵、训练来源与质量评测协议未修改，后续 GPU 任务仍待运行。

2026-10-19：串行控制器 1762045 已保留 teacher_fresh 与下一组之间的 GPU 验收窗口；只暂停调度器 1317263，当前训练 1664892 持续运行，结束后执行固定 f1bcea0 的两步/九十六条短响应 KL canary，审计并释放 GPU 后自动恢复原调度器。控制状态与当前进程已核验，观察超时不会重启或终止健康训练。

固定前缀 bank 新增真实输入准备、生产原生 RPC 生成入口及组装器。完整 MATH-500/AMC23/AIME25 各选三十题，按身份哈希加 seed42 排序；模型指纹、共同 tokenizer 和原始 token 固定。实际预处理在 token 合法性检查内反复调用 fast tokenizer 的 **len**，py-spy 定位到该热点；终止这个独立 CPU 准备进程并缓存词表大小后，同一九十题在约 15 秒内完成。词表查询次数回归检查、原生 admission 初始化和双来源/原始 prefix/EOS/评分器兼容共六项检查通过。bank 输入与 CPU 合成轨迹不是实际 GPU 生成证据，真实输出和完整 bank KL 仍待完成。

固定 bank 的输入准备、原生生成入口、严格组装及文档日期/引用维护提交为 fd24278。随后新增独立 KL artifact 审计器，流式读取原始 token，避免完整长响应文件在 RAM 中额外复制。审计绑定原准确率 JSONL 与同一 bank，复核哈希/身份/版本/有效位置/深度/worker 记录并独立归约；按 benchmark 分层的题目 bootstrap 保持同题十六个样本相关性，题目均值、token 比值和累计量分别处理。八项检查及 CLI 通过，包含非 macro 指标篡改、缺样本、错误长度/深度、缺 worker 和替换 bank；合成例的 MATH 题目均值区间为 [1,3]，不会误把三十二条响应视为三十二道独立题。GPU canary 已由控制器启动，仍在冷启动阶段，尚未计为通过。

独立 KL artifact/题目 bootstrap 提交为 a6de439。f1bcea0 四卡短 canary 随后成功完成：两次更新、九十六条最终 student2 响应，12,288 个有效位置，原生/独立审计均通过；W&B 12858j44 finished、两条训练 history、仅 step2 验证，110 个 KL 指标与精确最终行相同。评分 wall time 为 5.955 s，worker 内 student forward 约 2.27–2.30 s、teacher forward 约 2.47 s、head/KL 约 0.68–0.72 s；这些分项不把嵌套 callback 重复相加。原运行 status 的 started 字段曾被最终状态覆盖，已保留现场读取的 running-status 时间原点用于短验收阶段采样对齐，并修正未来入口保留 started/finished。

评分阶段 GPU 0/1/2/3 的两秒采样峰值为 8,283/8,263/8,283/8,283 MiB；整轮冷启动/训练/验证峰值为 50,865/50,715/50,715/50,715 MiB，不能混用或当成精确 allocator 峰值。验收完成后全部自有 canary 进程退出，GPU 回到基线，控制器恢复原调度器，当前 student_teacher_fusion 正常运行。原 teacher_fresh 也已生成完整正式结果，待远端核验与后续归档。固定 bank 及长响应/完整数据评分尚未通过，不以短验收代替。

独立四卡短 KL 验收提交为 4ae85dd。后续 0f2c1d8 为 standalone bank 保存 Ray/CUDA 资源归属，完整 CI 通过；bank/long 控制器 1850026 确认存活，只暂停调度器 1317263，current/teacher fusion 1814417 按原设置继续，结束后串行生成 90×2 bank 并运行带 bank 的 31,744-token 上限验证，再独立审计、释放 GPU 并恢复原矩阵。观察超时不终止健康任务，源代码固定提交不随工作树变化。

第三组 teacher_fresh 的 W&B 4skr6m3x finished、五十条训练记录及唯一 step50 验证均通过。Macro mean@16 为 63.80486%，相对 current 为 −0.49167 个百分点，5,000 次题目 bootstrap 区间 [−1.99722,0.90424] 个百分点。训练 step 22.02 min、原生生成 834.31 s，整轮最终验证 68.82 min，四卡峰值最高 55,043 MiB；三来源单 seed 质量结果不构成收敛或优劣结论。

新增 `eval/plot_native_formal.py`，从三组已归档证据重算逐题 mean/pass@k，核对五十步、完整题目身份、W&B 本地文件哈希、模型来源绑定与共同设置，输出训练、准确率、pass@k 和资源四张图及 CSV/审计。准确率/pass 按基准缩放，保留外部模型参考；Initial 标签表示实时冻结初始模型，不是旧 Semi 缓存。四张图的机器审计和目视检查通过，四个手动反例确认未完成运行、训练文件篡改、混合 seed 和错误 policy 绑定均被拒绝。统一报告增加两张内嵌图，其余图与可复现审计直接链接，维持单一实验正文；完整词表 KL 未追溯加入已结束运行。

三来源正式图提交为 c7c6433。随后实现原生单容器按 checkpoint 换权重的实验适配层，仍由现有 token executor/NativeRayBackend 处理路由、连续 prefix、物理副本与排空；逻辑 p0/q 与可写容器版本分开绑定，没有放开冻结 worker 的写入。设备局部 sender 预先准备两套锁页 CPU 权重，按整组经现有 CUDA IPC 并行写入、清理 KV 和提交；大 tensor 的直接 IPC 分支显式等候主机到设备复制结束。新增驻留串行/驻留并发控制，先单卡短测，再做四卡两轮逆序控制。每轮先预热，固定长度与自然 EOS 分开，计量不包括 CPU 缓存准备与引擎冷启动。

八十四项原生 CPU 检查及十二项基准检查通过，其中新增七项协议/生命周期方法，包含实际 NativeRayBackend 的双副本执行记录、双向分段、配额、EOS、错误版本与传输中取消。独立审计除每组绑定和时间组成外，还读取实际保存的完整词表 logits 重算误差/TV；十一种篡改反例覆盖来源、版本、缺失检查、放宽容差、原始数组、伪造误差、词表范围和计时。历史二十八条件、十九引擎记录回归通过。当前只有 CPU 与合成 artifact 证据；新换权重路径尚无 GPU 速度、显存或正确性结论。

换权重控制提交为 67d8b07，串行 watcher 1898531 固定该提交，等待原 scheduler sweep 成功。随后补齐全量 KL gate 与通用归档入口：实际核对三个完整 Parquet 的 500/40/30 行、data_source 与原准备清单哈希，训练只做两步 B8/64 token，验证仍为完整 570×16/31,744 cap，再评估相同 bank。资源归档按评分 epoch 对齐两秒整卡采样，评分期缺卡显式为无覆盖，不用整轮峰值替代；原始报告/记录/资源按字节绑定，native JSON 另用忽略排版和等价整数写法的规范化哈希，避免格式化破坏证据关系。

新增九项 CPU 检查通过，包含真实 producer 归约的完整 9,120+180 条构造记录、主位置全覆盖与 bank 分组、评分区间/整轮峰值区分、缺卡、版本/输入/范围错误、同 bank 的问题/prompt/模型身份以及计划 CLI。该构造只使用两 token/256 词表，不代表完整模型已评分。全量 slot 控制器先等待原 bank/long owner 完成并恢复调度器，再持有一个独立串行窗口；不改变健康训练，排空资源后运行全量 gate，完成或失败后恢复原矩阵。新的 gate、真实长响应和完整规模仍未通过，不将队列或 CPU 数据作为 GPU 结果。

全量 gate 提交为 b528c6d，控制器 1948845 固定该提交等待 bank/long owner。后续按已审阅插件的实际测试版本在独立环境安装 vLLM 0.15.1、Torch 2.9.1/cu129 与 Transformers 4.57.6，锁定一百八十五项依赖及插件 a987755；从锁定 Git 来源同步后，两个独立 CPU 进程的导入/注册和十九项 routing/demo 检查通过，未初始化 CUDA。原训练环境二百二十九个包的版本和 METADATA 校验值一致；这不是对全部文件的完整性声明。CPU 查询还区分了内核驱动 550.163.01 与既有用户态 libcuda 560.35.03/API 12.6，新栈 CUDA 12.9 的 GPU/JIT 仍待验证，没有加载新的兼容库或修改系统驱动。

实际源码核验修正了检查入口：`VLLM_BATCH_INVARIANT` 不在中央 env 注册表，但专用模块会在导入时读取并缓存它；开/关均已在新进程核对，导入后改环境变量不改变该布尔量。GPU kernel override 尚未初始化。另直接调用安装插件的原始 loader，在空 CPU checkpoint 迭代器下不读取给定流、零写入却返回全部参数名，说明不能原样承担 actor 热更新。源文件与官方 v0.15.1 标签的两项校验值一致；GPU/双 KV/graph/热更新仍未验收。

该阶段结束时，current/teacher fusion 的冻结 d59a7c3 运行已成功退出，独立采集核对五十步和完整 9,120 条最终响应，macro mean@16 为 63.88819%。原调度器仍由 bank/long owner 持有，bank 作业 1984062 已启动。其恢复后调度器会再次写入正式结果目录，故最终 W&B 核验、配对比较与图件更新等待调度器完成这次归档，避免格式化后的证据被再次覆盖。

独立环境提交为 5916515。Bank v1 作业随后以退出码 1 结束，失败在生成后的物理 UUID 核验；没有正式 prefix-bank.json，原 driver 未保存该失败的 worker receipt 与轨迹，因此不能将它当成有效 bank 或反推全部 worker 的实际身份。旧 owner 清理完毕并恢复调度器，initial_teacher_fusion 2022419 正常继续；full watcher 因前置失败退出，没有启动长响应/全量 KL。只读 Driver/NVML 枚举发现同一 PCI 对应不同 UUID；CUDA UUID 掩码下 Driver 可见一张、Torch/NVML 计数为零，数字序号掩码下两者均为四。查询没有 CUDA 当前/保留 context，不是并行推理实验。

新增共享 PCI 身份映射、可见性预检与已知别名校验，bank 和纯推理基准按核验后的序号选择物理 0–3；启动后即保存并检查 worker 身份，生成原始轨迹与执行记录在校验之前保存，正式 bank 仍只在全部验收后发布。布局审计将同硬件的 CUDA/NVML 别名统一后再查 rank 唯一性，启动时的身份清单按哈希绑定，旧 UUID 字符串接口保留。三十二项相关维护检查、六项真实训练环境 bank 检查，以及历史二十八条件/十九引擎记录回归通过；新 GPU 路径仍待重测。重测使用新 v2 目录、相同输入哈希与 0f2c1d8 神经网络源码，等待当前健康训练结束后执行，不覆盖失败记录。
