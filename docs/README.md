# OPD-State 文档

从研究问题或当前操作进入。研究报告绑定实际源码、模型、数据和协议；当前实现变化不自动成为旧运行的验证。未完成的配置重构保持 proposed，不用目标配置覆盖现有 recipe 的使用说明。

| 需要回答的问题 | 权威入口 |
| --- | --- |
| 轨迹来源如何影响学习效果与计算成本？ | [问题主报告](experiment/2026-10-19-rollout来源与logit融合实验.md) |
| 条件目标、访问测度与收敛关系是什么？ | [研究导读](research/occupancy/2026-10-17-OPD访问测度与收敛机制研究.md) |
| 当前训练、生成、评分与更新怎样连接？ | [代码结构](architecture.md) |
| 怎样准备环境、运行与验证？ | [开发指南](development.md) |
| 为什么采用当前目录与设计，哪些方案尚待验收？ | [Agent Notes](../.agents/notes/README.md) |
| 上游源码、维护规则与许可证来自哪里？ | [来源](provenance.md) |

## 主报告与支撑材料

问题主报告维护关键实验及综合结论；[准确率](experiment/2026-10-19-rollout准确率与训练指标.md)、[速度与显存](experiment/速度与显存.md)和[KL 诊断](experiment/2026-10-19-rollout验证KL与状态诊断.md)保留所属方法与详细结果。原生链路、恢复和数值检查归 [support](experiment/support/README.md)，均回链主问题。原始产物保留在既有 experiment/results、figures 与 research 证据目录，不为重排正文搬动数据。

`docs/notes/` 是历史需求和工作过程，退出默认检索与常规导航，不作为当前操作或结果权威。新增执行进展留在忽略的 runs；原始证据没有持久归属前不得清理。
