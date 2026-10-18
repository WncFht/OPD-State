# Sampled-token 与无偏尾部估计

更新时间：2026-10-18。本篇区分 action 采样噪声、稀有 token 的弱平均信号和未来 reward 方差，并推导合并尾桶、无偏 head-tail 数值与梯度。它是比较估计器是否保持同一目标的依据。

[返回研究导读](2026-10-17-OPD访问测度与收敛机制研究.md)。定理与命题在本篇独立编号；跨篇引用直接链接到目标。

相关约定：[条件梯度与停止梯度约定](2026-10-18-局部梯度与长度归约.md)。

本篇用 $p_\theta$ 表示 student、$q$ 表示 teacher；$J$ 为 sequence KL，$M$ 为平均长度，$F=J/M$ 为 pooled token KL。

本篇导航：[1. 噪声与弱信号](#1-噪声与弱信号) · [2. 桶 KL 与无偏 head-tail](#2-桶-kl-与无偏-head-tail)。

## 1. 噪声与弱信号

无偏 head-tail 属于已有估计方法；TT-OPD 给出相关构造[^tt-opd]。本节区分数值估计、条件梯度与未来信用，不将三者混为“完整 KL 无偏”。

### 1.1 定理 1：状态方差与 action 方差可以分别辨识

**定理陈述。** 冻结模型和教师，独立抽取 $B$ 个状态 $s_i\sim\nu$；每个状态独立重采 $m$ 个 action $a_{ij}\sim p_\theta(\cdot\mid s_i)$。令

$$
X(s,a)=e_\theta(s,a)u_\theta(s,a),\qquad
g(s)=\mathbb E[X\mid s],\qquad\Sigma_a(s)=\operatorname{Cov}(X\mid s),
$$

$$
\widehat G=\frac1B\sum_{i=1}^B\frac1m\sum_{j=1}^mX(s_i,a_{ij}).
$$

则 $\mathbb E\widehat G=\nabla L_\nu$，且

$$
\operatorname{Cov}(\widehat G)
=\frac1B\operatorname{Cov}_{s\sim\nu}(g(s))
+\frac1{Bm}\mathbb E_{s\sim\nu}\Sigma_a(s).
$$

增大 $m$ 只缩小第二项，不消除状态间梯度变化，也不消除整体梯度方向不适合任务目标的问题。

**证明。** 对单个状态的平均 $\overline X=m^{-1}\sum_jX(s,a_j)$，条件独立给出

$$
\mathbb E[\overline X\mid s]=g(s),\qquad
\operatorname{Cov}(\overline X\mid s)=\frac1m\Sigma_a(s).
$$

将 $\overline X-\mathbb E\overline X$ 分解为 $\overline X-g(s)$ 与 $g(s)-\mathbb E g(s)$。两个交叉二阶矩为零，因为前一项给定 $s$ 的均值为零。故

$$
\operatorname{Cov}(\overline X)
=\operatorname{Cov}(g(s))+\frac1m\mathbb E\Sigma_a(s).
$$

再利用 $B$ 个状态组独立，将协方差除以 $B$，得到结论。均值由[梯度·定理 1](2026-10-18-局部梯度与长度归约.md#11-定理-1当前-student-action-给出无偏条件-kl-梯度) 得到，证毕。

**适用边界。** 同一原始轨迹中的 token 不独立，且原 action 会决定后续 prefix，不能把真实训练的 token 总数直接当成 $B$。该定理对应冻结 prefix 后的独立 action 重采样诊断；训练日志的不确定性应按轨迹或 prompt 聚类估计。采用 top-$K$ 重归一化会改变均值目标，不能作为只降低第二项的严格干预。

还需同时测量均值梯度强度。方差大、均值小、跨状态梯度抵消，以及梯度与任务改进方向不一致，是不同现象。reward 的方差、正 reward 比例或 top-$K$ overlap 均不能替代梯度协方差。

### 1.2 命题 2：稀有 teacher 候选同时面临弱均值与低信噪比

**命题陈述。** 固定有限词表和严格正的 teacher 分布 $q$。考虑 token $v$，令 $p_v=\varepsilon\to0$。原始无 baseline 的 sampled-token 梯度在 logit $z_v$ 上为 $\widehat g_v=e_A(\mathbf1\{A=v\}-p_v)$，其中 $A\sim p$。则

$$
\mathbb E\widehat g_v
=p_v(e_v-\ell_s)
=\varepsilon\log\varepsilon+O(\varepsilon),
$$

$$
\operatorname{Var}(\widehat g_v)
\sim\varepsilon(\log\varepsilon)^2,
\qquad
\frac{\operatorname{Var}(\widehat g_v)}{(\mathbb E\widehat g_v)^2}
\sim\varepsilon^{-1}.
$$

完整词表求和消除这一条件 action 采样噪声，但保留相同的弱平均梯度。

**证明。** 对 sampled action 求期望：

$$
\sum_ap_ae_a(\mathbf1\{a=v\}-p_v)
=p_ve_v-p_v\sum_ap_ae_a=p_v(e_v-\ell_s).
$$

固定正 teacher 上有 $0\le\ell_s\le\log(1/\min_aq_a)$；又有 $e_v=\log\varepsilon-\log q_v$，得到均值展开。二阶矩为

$$
\mathbb E\widehat g_v^2
=\varepsilon(1-\varepsilon)^2e_v^2
+\varepsilon^2\sum_{a\ne v}p_ae_a^2.
$$

函数 $x\log^2x$ 在 $[0,1]$ 连续延拓且有界，固定 $q_a>0$ 与有限词表保证后一求和有统一上界。故第一项渐近为 $\varepsilon\log^2\varepsilon$，第二项为 $O(\varepsilon^2)$；减去 $O(\varepsilon^2\log^2\varepsilon)$ 的均值平方，得到方差与相对方差，证毕。

**适用边界。** 这是固定 prefix、固定 teacher 和原始无 baseline 估计器的渐近结论，不能套到 teacher 随 prefix 同时变化的极限。Baseline、head 枚举和控制变量会改变方差；完整求和不改变 reverse KL 的均值几何。未采到的 token 也可经 softmax 分母获得间接更新，因此“没有直接采样”不等于该 logit 完全没有梯度。

### 1.3 未来 reward 的方差代价

Revisiting OPD 区分了局部估计器与完整序列估计器的偏差和保守方差上界[^revisiting]。若 $|e_t|\le C_e$、$\|u_t\|\le C_u$，且常数不随分析时域增大，则三角不等式给出

$$
\|\widehat G\|\le HC_eC_u,\qquad
\|\widehat G_{\rm RTG}\|
=\left\|\sum_tu_t\sum_{j\ge t}e_j\right\|
\le\frac{H(H+1)}2C_eC_u.
$$

用 $\mathbb E\|X-\mathbb EX\|^2\le\mathbb E\|X\|^2$ 得到相应的 $O(H^2)$ 和 $O(H^4)$ 上界。它们不是实际方差的等式，也不证明每个问题中两种估计器必按这个顺序表现。双方使用同一固定尺度归约时，方差和平均梯度应同时缩放，不能只比较绝对方差。

给定 $(s_t,a_t)$ 的未来期望失配可以代替该位置的 sampled future return，保持该位置的期望贡献，并消除其后缀采样噪声。不同位置之间仍有协方差；学习近似 value 还引入新的估计偏差。选择局部监督的方差理由与 tabular 收敛定理是两个独立论点。

## 2. 桶 KL 与无偏 head-tail

### 2.1 命题 3：合并尾桶 KL 是完整 KL 的下界

**命题陈述。** 在固定 prefix 上，选定 head 集合 $S$，令 $T=\mathcal V\setminus S$、$\tau=\sum_{a\in T}p_a$、$\upsilon=\sum_{a\in T}q_a$。合并尾部成一个类别的 KL 为

$$
D_{\rm bucket}=\sum_{a\in S}p_a\log(p_a/q_a)+\tau\log(\tau/\upsilon).
$$

若两侧尾部质量为正，则

$$
\operatorname{KL}(p\Vert q)=D_{\rm bucket}
+\tau\operatorname{KL}(p(\cdot\mid T)\Vert q(\cdot\mid T))\ge D_{\rm bucket}\ge0.
$$

**证明。** 对尾部写 $p_a=\tau p(a\mid T)$、$q_a=\upsilon q(a\mid T)$，得到

$$
\sum_{a\in T}p_a\log(p_a/q_a)
=\tau\log(\tau/\upsilon)+\tau\sum_{a\in T}p(a\mid T)\log\frac{p(a\mid T)}{q(a\mid T)}.
$$

加上 head 项即得分解。两个 KL 均非负。空尾部按零贡献处理；student 尾部有质量而 teacher 尾部为零时使用无穷值约定，证毕。

**适用边界。** 桶 KL 为零不意味着尾部内部已经对齐；仅有小尾部质量也不够，仍需控制尾部条件 KL。MirrorOPD 的 `grouped_kl` 实现此粗粒化量，生产 `topj_tail` 用于相应中心或诊断；它本身不是完整尾部误差的无偏补偿。

### 2.2 定理 4：无偏 KL 数值估计与无偏梯度需要分别实现

**定理陈述。** 固定 prefix 与采样前选定的集合 $S$，令 $e_a=\log(p_a/q_a)$、$u_a=\nabla\log p_a$，从当前 student 取 $A\sim p$。对任意不依赖该次 action 的标量 $\beta$，

$$
\widehat D_\beta=\sum_{a\in S}p_ae_a+\tau\beta+\mathbf1\{A\in T\}(e_A-\beta)
$$

无偏估计完整 KL。取 $\beta=\log(\tau/\upsilon)$ 时，前两项是桶 KL，最后一项补回被忽略的尾部条件误差。

训练使用的无偏梯度则可取

$$
\widehat g=\sum_{a\in S}p_ae_au_a+\mathbf1\{A\in T\}e_Au_A.
$$

在冻结当前锚点 $\bar\theta$、只更新一次完整 batch 的条件下，它可以由以下 score surrogate 实现：

$$
\widehat L(\theta;\bar\theta)=
\sum_{a\in S}\operatorname{stopgrad}(p_{\bar\theta,a}e_{\bar\theta,a})\log p_{\theta,a}
+\operatorname{stopgrad}(\mathbf1\{A\in T\}e_{\bar\theta,A})\log p_{\theta,A}.
$$

若实际 action 来自行为分布 $b$，并且 $b_a>0$ 覆盖所有 $p_a>0$ 的尾部 action，令 $w_A=p_A/b_A$。同一定理的修正版为

$$
\widehat D_{\beta,b}=\sum_{a\in S}p_ae_a+\tau\beta+\mathbf1\{A\in T\}w_A(e_A-\beta),
\qquad
\widehat g_b=\sum_{a\in S}p_ae_au_a+\mathbf1\{A\in T\}w_Ae_Au_A.
$$

surrogate 的尾部冻结系数同步乘 $w_A$；这两个估计在固定 prefix 上仍然无偏。

**证明。** 对数值估计，尾部指示项的期望为 $\sum_{a\in T}p_a(e_a-\beta)$。其中 $-\tau\beta$ 与前项相消，留下全词表 $\sum_ap_ae_a$。对梯度估计，同样将尾部指示项按 $p$ 展开，得到全词表 $\sum_ap_ae_au_a$，由[梯度·定理 1](2026-10-18-局部梯度与长度归约.md#11-定理-1当前-student-action-给出无偏条件-kl-梯度) 它等于完整条件 KL 梯度。

surrogate 中所有系数都停止梯度，所以在 $\theta=\bar\theta$ 的导数恰好等于该 $\widehat g$。head 仍使用完整词表概率，未在 $S$ 内重归一化。对行为分布扩展，任意函数 $h$ 满足 $\mathbb E_b[\mathbf1_Tw_Ah(A)]=\sum_{a\in T}b_a(p_a/b_a)h(a)=\sum_{a\in T}p_ah(a)$，分别代入 $h=e-\beta$ 和 $h=eu$ 即可复用前述证明，证毕。

**适用边界。** 不应直接对 $\widehat D_\beta$ 数值反向传播并认为梯度也无偏。特别地，对 live head 项 $\sum_Sp_ae_a$ 直接求导，会得到 $\sum_Sp_a(e_a+1)u_a$；若仅再接 sampled tail 的 $e_Au_A$，就多出 $\sum_Sp_au_a$。使用上面的冻结 score 形式可避免这个遗漏的抵消项。

head 必须只依赖 prefix 和冻结 anchor，不能依赖该次 sampled action；实现可以在 rollout 后重算它，但本次更新必须保持相同 token ID。把 sampled action 事后强行加入 head，再原样使用该公式，会破坏上述固定集合的推导。若复用旧 action 或使用 teacher/fusion action，必须使用行为概率修正版并重新检查支持；权重裁剪会引入偏差。

这个修正没有将 $\nu_b$ 变成 $\nu_p$。校正后的条件 KL 按实际轨迹求和，估计的是 $\mathbb E_{y\sim B}\sum_t\operatorname{KL}(p(\cdot\mid s_t)\Vert q(\cdot\mid s_t))$，只有实际行为与评分策略一致时才等于 $\operatorname{KL}(P\Vert Q)$。另一个可直接估计的量是 $\mathbb E_B\sum_t[\log b(A_t\mid s_t)-\log q(A_t\mid s_t)]=\operatorname{KL}(B\Vert Q)$，其单 action 方差通常更大，但序列解释不同，不能把二者混称。

### 2.3 方差、尾部稀有事件与数值稳定性

对于上述标量估计，令 $m_T=\mathbb E[e_A\mid A\in T]$，直接展开二阶矩可得

$$
\operatorname{Var}(\widehat D_\beta)
=\tau\operatorname{Var}(e_A\mid A\in T)
+\tau(1-\tau)(m_T-\beta)^2.
$$

证明是 head 与 $\tau\beta$ 均为常数，随机项 $I_T(e_A-\beta)$ 的二阶矩为 $\tau\mathbb E_T(e_A-\beta)^2$，均值平方为 $\tau^2(m_T-\beta)^2$。两者相减得到上式。选择桶 log-ratio 作 $\beta$ 是可计算的控制量，不保证它总是方差最优。

无偏梯度的方差也不保证随 $k$ 单调降低，因为 head 和 tail 的梯度方向会影响协方差。应记录 tail 质量、命中数、残差和按轨迹估计的不确定性；少量或零次 tail 命中不能证明尾部已经对齐。每隔若干 checkpoint 需要完整词表校准，不能以桶 KL 的下降替代这一检查。

行为修正的随机尾项二阶矩为 $\sum_{a\in T}p_a^2(e_a-\beta)^2/b_a$；它揭示真正的风险是相关尾部位置上的 $p_a^2/b_a$ 与 log-ratio 共同变大，而不是一个没有指定权重的 top-$k$ overlap 数字。当 $b=p$ 时不存在这个重要性权重风险，仍可能有条件 log-ratio 方差和[定理 1](#11-定理-1状态方差与-action-方差可以分别辨识) 的状态方差。

尾部 log-mass 使用补集的 logsumexp，避免以 FP32 的 $1-\sum_Sp$ 相减产生灾难性消减；head 与 action 使用相同有效词表、相同温度和完整归一化。KL 单次无偏估计可以为负，不做逐 token 截零。EOS 和预算末 action 都参与监督，padding 在概率算术之前排除。

MirrorOPD 的相关依据为[分组 KL](../../../../MirrorOPD/src/mirroropd/core/distributions.py)、[诊断与中心](../../../../MirrorOPD/src/mirroropd/core/diagnostics.py)及[KL 恒等式核验](../../../../MirrorOPD/tests/reference/test_kl_theory.py)。本文采用其已区分的“桶 KL”和“采样尾部补偿”语义，不将生产 `topj_tail` 配置误称为已经训练了完整 KL。

## 延伸阅读

[比较 TIDE 改变的目标](2026-10-18-重复退化与TIDE机制.md)；[运行已实现的 head-tail 配方](2026-10-18-双向实验运行与验证.md)。

### 参考文献

[^tt-opd]: Linjian Meng 等。Unbiased Top-k Estimation for On-Policy Distillation. arXiv 2026，§4.1。 [arXiv:2609.34447v2](https://arxiv.org/html/2609.34447v2).

[^revisiting]: Yuqian Fu 等。Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes. arXiv 2026，§2.1、附录 D 与 Table 1。 [arXiv:2603.25562](https://arxiv.org/html/2603.25562).
