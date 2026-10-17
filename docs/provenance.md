# 来源与许可

## 训练源码

One-Shot-OPD[^one-shot] 在 `training/one-shot-opd/` 作为普通文件直接维护。初始快照为 `6b6b1782cc3f04749dd077de19353f839f6bc048`，逐文件 SHA-256 和 Git 文件模式归 [导入清单](../provenance/one-shot-opd.json)。[LICENSE](../training/one-shot-opd/LICENSE)、[NOTICE](../training/one-shot-opd/NOTICE) 和原 README 保留原样；其中 vendored verl 的既有归属继续保留。

导入时的完整性使用 `scripts/check_upstream.py --pristine` 验证。后续修改直接提交到 OPD-State；清单继续代表导入基线，普通检查允许修改并报告差异。当前仓库沿用已有研究提交历史，不改写旧提交。

## 维护规则与工具

agent-maintenance[^maintenance] 的来源提交为 `f2b7cdfe4c6ab96ed779f42c878e84b7042edf43`。[来源元数据](../provenance/agent-maintenance.json)记录采用范围；其 MIT 许可保存为 [agent-maintenance-MIT.txt](../licenses/agent-maintenance-MIT.txt)。采用三个维护技能及其共享引用、模板、Notes 规则、检查调度器和两阶段 hooks；目标路径、检查集合、扫描范围与依赖按 OPD-State 调整。原包过程记录与可选 MPE 资源未纳入。

AutoCorrect 2.16.3 与 gitleaks 8.30.1 沿用维护包的版本基线；Linux 安装器固定发布包 SHA-256，安装到项目虚拟环境。工具链行为测试归 `scripts/tests/`。

## 研究材料

研究正文、精确实验与结果沿用 OPD-State 的原有研究提交；原始需求保存到 notes，README 改为使用入口。所引论文和 MirrorOPD 文件的哈希归 [研究来源清单](../research/occupancy/results/source-manifest.json)。这些相邻仓库不随本仓复制。图件重绘使用外部 figure-viz 技能，依赖位置见[开发指南](development.md#cpu-研究复现)。

### 参考文献

[^one-shot]: Thinking-Space. One-Shot-OPD. GitHub 2026. [Thinking-Space/One-Shot-OPD](https://github.com/Thinking-Space/One-Shot-OPD).

[^maintenance]: WncFht. agent-maintenance. GitHub 2026. [WncFht/agent-maintenance](https://github.com/WncFht/agent-maintenance).
