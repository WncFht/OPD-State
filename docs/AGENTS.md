# AGENTS.md — 文档

本层只补充文档规则；[文档分工](../.agents/references/documentation.md)定义各类事实的归属，[maintain-docs](../.agents/skills/maintain-docs/SKILL.md)负责具体编写与审计。

- 页面的位置与主题决定其范围。上层解释职责、关系和高层行为，细节链接到实际归属；参考文档也只对自己的主题穷尽。
- 区分有顺序的教程与按需查询的参考。教程先给前置条件，再给动作、成功证据与恢复方法。
- 普通文档描述当前状态；决策历史、迁移指南和事故报告保留各自所需的时间语境。
- 事实优先核对源码、测试、生成器与当前记录。操作说明按展示方式实跑；缺凭据或环境时明确未验证及验证责任，不把邻近示例当证据。
- 一项事实一个权威归属；接口使用点保留必要的本地条件、失败和所有权约束，扩展理由链接归属。
- 不手改生成目录，不维护可从源码生成的平行清单。移动页面时同变更修复入链。
- 段落不手工折行。中文使用稳定术语、明确主体、直接动词，保留条件、例外与义务。
- 预算只用于常驻入口和实际超大的检索单元；先归位、压缩重复，再按证据调整预算。中文不照搬英文词数。

Agent Notes 不套普通文档章节，遵循[记录规则](../.agents/notes/README.md)。

本项目实际布局：architecture.md 负责当前组件关系，development.md 负责运行与开发；research/occupancy 负责推导与方法；experiment 的轨迹来源报告是研究问题主入口，support 是独立支撑验证，既有 results/figures 保留原始身份。新证据更新所属报告，CPU/GPU 不决定主次。docs/notes 是历史过程，新执行流水进入 runs。导航见 [README](README.md)，交付遵循[生命周期](../.agents/references/document-lifecycle.md)和[实验规范](../.agents/references/experiment-reports.md)。篇幅清单为 scripts/doc-policy.json，不做重复检测。
