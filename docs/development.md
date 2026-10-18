# 开发与验证

## 开发环境

维护和精确 CPU 实验使用根目录 `.venv`；Python 包归 `requirements-dev.txt` 与带哈希的 `requirements-dev.lock`，npm 工具归 `package.json` 与 `package-lock.json`。安装命令见 [README](../README.md)。Linux x86-64 使用 `.venv/bin/python scripts/install_dev_binaries.py` 将 AutoCorrect 2.16.3 与 gitleaks 8.30.1 下载到本仓虚拟环境，安装前核验固定 SHA-256；其他平台用系统包管理器安装相同版本。CI 复用该安装器和凭据检查。删除 `.venv` 后需重新安装这些工具。

安装 hooks 使用 `.venv/bin/python scripts/setup.py`。`pre-commit` 第一阶段用 git-format-staged 格式化暂存区，第二阶段运行 pre-commit 检查。`pre-push` 读取 Git 提供的全部待推送引用，在临时 Git worktree 中逐个检查不同的提交版本，运行完整 CI 检查并扫描该版本可达的历史凭据；删除引用跳过检查。检查完成或失败后移除临时 worktree，原工作区和暂存区保持原状。临时目录复用当前仓库的 `.venv` 和 `node_modules`，需要先完成维护环境安装。安装器保留外来 hook 和自定义 hooksPath，遇到冲突时退出。不要使用普通 pre-commit install 覆盖两阶段入口。虚拟环境移动或重建后重装 hooks。

GitHub Actions 的 `Checks / check` 对 `main`、`master` 的 push 和所有 pull request 运行，也支持手动触发。CI 从锁定依赖重建维护环境，运行完整检查与历史凭据扫描；令牌权限为 `contents: read`，单任务限时 15 分钟。仓库交付训练源码与研究资料，当前没有部署服务或自动发布目标。

本仓发布地址为 `git@github.com:WncFht/OPD-State.git`，默认开发分支为 `main`；首次推送使用 `git push -u origin main`。推送前确认 `git status`，只有提交中的文件会进入远端。hooks 失败时按输出修复对应提交，再重新推送；CI 结果在仓库的 Actions 页面查看。

## 检查与格式

```bash
.venv/bin/python scripts/run_checks.py --config scripts/checks.json --mode docs
.venv/bin/python scripts/run_checks.py --config scripts/checks.json --mode test
.venv/bin/python scripts/run_checks.py --config scripts/checks.json --mode ci
npm run format
.venv/bin/ruff format .
.venv/bin/python scripts/check_spacing.py --fix
```

`docs` 检查技能、内部链接与 Notes；研究引用的相邻论文和 MirrorOPD 由研究来源清单记录，普通文档门控不要求下载这些外部仓库。`test` 验证维护门控、上游源码语法和 launcher 参数传递，并重算精确数学构造。`ci` 再汇总格式、lint、间距和研究公式检查。检查失败返回非零状态，修复后重跑对应入口。

Python 使用 Ruff；Markdown、JSON、YAML、CJS 使用 Prettier；Markdownlint 与 AutoCorrect 检查文字。根格式检查保留 `README.md`、`data/`、`eval/`、`recipe/` 与 `verl/` 的原有风格，改动上游 Python 时使用其 `verl/pyproject.toml` 的规则并补相关测试。格式化冲突时审阅工作区与暂存区两个 diff，保留未暂存内容；Markdownlint 修复内容后需重新暂存。

```bash
.venv/bin/python scripts/check_source.py
```

源码检查按当前 Git 文件清单编译 Python、检查 Shell 语法。它允许正常修改、增删和重构训练代码；来源清单只记录最初导入版本，不作为日常开发的完整性门控。新增行为仍需运行对应上游测试和训练基线。

## 上游对照

Git remote `upstream` 指向 Thinking-Space/One-Shot-OPD，当前对照版本是导入时的 `6b6b178`。代码直接在当前仓库修改和提交；查看训练实现相对上游的差异：

```bash
git diff upstream/main -- recipe data eval verl
```

需要更新对照版本时先执行 `git fetch upstream` 并审阅差异，再决定合入哪些变更。

## CPU 研究复现

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python docs/research/occupancy/experiments/exact_checks.py
node docs/research/occupancy/experiments/check_document.cjs
```

数学核验只依赖 NumPy；公式语法检查使用本仓锁定的 KaTeX。研究正文的外部本地链接需要相邻 `Awesome-OPD-Paper/` 与 `MirrorOPD/`，检查结果单独列出缺少的外部材料，不将外部缺失视为内部断链。

绘图额外使用工作区 figure-viz 技能的 `assets/mpl`，其资源不随训练代码导入。默认定位当前工作区 `.agents/skills/figure-viz`；跨机器用 `FIGURE_VIZ_HOME` 指定技能根目录。缺少该资源时先准备资源，再运行：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python docs/research/occupancy/experiments/plot_results.py
```

绘图保留 CSV 来源与机器审计；图件更新后查看 PNG，核对图例、尺度和裁切。普通 CI 重算数学结果并检查正文，不依赖外部绘图技能。

## 训练环境与验收

上游声明的运行栈为 Python 3.12、vLLM 0.28、Torch 2.13 和 Transformers 5.x。本机 math 配方使用独立的 CUDA 12.6 环境：Torch 2.7.1+cu126、vLLM 0.10.0+cu126、Transformers 4.57.6，通过 PYTHONPATH 选择本仓源码。针对旧版 vLLM 的导入、CLI 与缓存生命周期做了兼容适配，已用本地 Qwen2 架构 student/teacher 完成四卡两步更新与最终评测。该验证不覆盖其他模型或全部上游功能，也不等于满足上游 setup.py 的全部新版本依赖声明。

工作区 `envs/opd-state` 指向本机缓存中的独立环境；已核对 Torch 实际加载路径与四卡 CUDA 运算，并通过 229 个已安装包的依赖检查。共享 `envs/opd` 保持原样。正式 math 50 步配方为 `bash recipe/math_baseline.sh`，W&B project 固定为 OPD-State，中途不验证且不保存 checkpoint；最终在 MATH-500、AMC23、AIME25 各题采样 16 次。固定来源和 fusion 分别使用 `recipe/math_cached.sh` 与 `recipe/math_fusion.sh`，长时固定来源每 50 步验证；配置、缓存准备、GPU 验收和已完成结果见[统一实验协议](experiment/2026-10-18-rollout来源与logit融合实验.md)。

训练模型、数据准备与完整参数遵循 [README](../README.md) 和 [数据说明](../data/README.md)。准备完成后，从本仓根目录启动的形式为：

```bash
export MODEL_BASE=/absolute/path/to/models
export CONDA_ENV_BIN=/absolute/path/to/training-env/bin
CUDA_VISIBLE_DEVICES=0,1,2,3 N_GPUS_PER_NODE=4 \
    MODE=oneshot bash -o pipefail recipe/opd.sh \
    trainer.total_training_steps=10
```

模型、数据与训练环境需要实际存在；四卡规模、显存预算和 batch 参数仍需实测。使用 `bash -o pipefail` 使训练失败通过上游 tee 管道传回调用方。`OPD_DRY_RUN=1` 仍导入训练依赖，并非无依赖检查；本仓 launcher 测试用替身 Python 核验参数与路径，不代表 Hydra 配置或 GPU 训练通过。

GPU 验收至少记录初始模型身份、权重保存策略、数据哈希、随机种子、有效 token 数、采样配置、时间和峰值显存，先跑原算法小规模基线，再比较候选变体。GPU 限物理编号 0–3；覆盖率与准确率结论必须基于真实结果。三种轨迹来源和 λ=0.5 fusion 已完成四卡 50 步及完整三基准验证；这些单 seed、单预算结果不证明长时收敛。
