# 来源与许可

## 训练源码

One-Shot-OPD[^one-shot] 以本仓根目录为开发根目录，直接维护原有源码。初始快照为 `6b6b1782cc3f04749dd077de19353f839f6bc048`，逐文件 SHA-256 和 Git 文件模式归 [导入清单](provenance/one-shot-opd.json)。[LICENSE](../LICENSE)、[NOTICE](../NOTICE) 保留原样，根 [README](../README.md) 说明 OPD-State 的算法、实现范围、训练与验证流程；其中 vendored verl 的既有归属继续保留。

初次导入已核验文件内容与模式。源码现在直接位于仓库根目录，后续修改直接提交到 OPD-State；清单只保存导入基线，不限制后续增删和重构。当前仓库保留已有研究和初始化提交历史。

## 维护规则与工具

agent-maintenance[^maintenance] 的来源提交为 `78f4c630c71f22f4464ec718b7a7803f3d00418a`。[来源元数据](provenance/agent-maintenance.json)记录采用范围；其许可全文见[MIT 许可](#mit-许可全文)。采用三个维护技能及其共享引用、模板、Notes 规则、检查调度器和两阶段 hooks；目标路径、检查集合、扫描范围与依赖按 OPD-State 调整。原包过程记录与可选 MPE 资源未纳入。

AutoCorrect 2.16.3 与 gitleaks 8.30.1 沿用维护包的版本基线；Linux 安装器固定发布包 SHA-256，安装到项目虚拟环境。工具链行为测试归 `scripts/tests/`。

### MIT 许可全文

```text
MIT License

Copyright (c) 2026 DeepSeek

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 研究材料

研究正文、精确实验与结果沿用 OPD-State 的原有研究提交；原始需求保存到 docs/notes，README 改为使用入口。所引论文和 MirrorOPD 文件的哈希归 [研究来源清单](research/occupancy/results/source-manifest.json)。这些相邻仓库不随本仓复制。图件重绘使用外部 figure-viz 技能，依赖位置见[开发指南](development.md#cpu-研究复现)。

### 参考文献

[^one-shot]: Thinking-Space. One-Shot-OPD. GitHub 2026. [Thinking-Space/One-Shot-OPD](https://github.com/Thinking-Space/One-Shot-OPD).

[^maintenance]: WncFht. agent-maintenance. GitHub 2026. [WncFht/agent-maintenance](https://github.com/WncFht/agent-maintenance).
