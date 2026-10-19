# OPD-State

OPD-State 基于 One-Shot-OPD 研究轨迹来源如何影响语言模型蒸馏。当前 student、初始 student、teacher 及其混合策略生成训练轨迹，所有变体只更新 student；研究同时区分 prefix 访问、action 采样、条件拟合和任务效果。

原生来源矩阵已有六组五十步训练与完整数学评测，尚不支持来源优劣排序；验证 KL 的工程验收与正式质量运行分别解释。完整条件、负结果和未完成范围见[问题主报告](docs/experiment/2026-10-19-rollout来源与logit融合实验.md)。

## 使用与阅读

- [Rollout 策略与训练语义](docs/reference/rollout.md)：已支持的策略、共同更新、参数与运行入口。
- [开发与验证](docs/development.md)：环境、检查与实际运行前提。
- [代码结构](docs/architecture.md)：当前组件、职责和修改入口。
- [研究导读](docs/research/occupancy/2026-10-17-OPD访问测度与收敛机制研究.md)：测度、目标、估计器和收敛条件。
- [全部文档](docs/README.md)：问题、当前参考和证据入口。
- [来源与许可](docs/provenance.md)：One-Shot-OPD、维护规则与研究材料的归属。

运行前按开发指南建立独立环境并核验已分配资源。默认配置不代表实际运行身份，实验结果必须绑定解析配置、执行源码与原始输出。配置与架构重构仍按各自提案验收，旧实验保留固定协议。

架构重构已暂停，已落地范围与尚未完成的 suite 接线、旧入口清理及 GPU 验收见 [Agent Note](.agents/notes/proposed/architecture/2026-10-19-MirrorOPD配置与架构对齐.md#暂停状态与未完成工作)。
