# 抽象代数

本笔记摘自《近世代数讲义》（杨劲根等），加上浙江大学数学科学学院开设课程《抽象代数》（石友晟主讲）中的部分内容整理而成，供复习参考之用。

## Chapter 1: 群的基本知识

### Section 1.1: 定义和例子

> [!definition] Definition 1.1.1: 二元运算
> 设 $S$ 是一个集合，将映射
> 
> $$
> f: S \times S \to S, \quad (a,b) \mapsto ab
> $$
> 
> 称为 $S$ 上的一个二元运算。

这里 $ab$ 是 $f(a, b)$ 的简记，而不是实数乘法。

几个二元运算的例子：

- 向量的和与叉乘（注意内积不是二元运算，因为将向量空间的直积映为实数）
- $n$ 阶方阵的和与左右乘
- 处处有定义的单变量实值函数的复合

这里不对结合律和交换律的定义详述。

> [!definition] Definition 1.1.2: 幺元
> 将满足 $e\in S$ 且：
> 
> $$
> ea=ae=a. \quad \forall a \in S
> $$
> 
> 的 $e$ 称为幺元（恒等元）。

>[!proposition] Proposition 1.1.3
> 幺元存在则唯一


证明可以假设存在两个幺元，证明其相同。

> [!definition] Definition 1.1.4: 逆元
> 设 $a \in S$，$e$ 为 $S$ 中幺元，若存在 $b \in S \quad s.t.$: 
> 
> $$
> ab=ba=e
> $$
> 
> 则称 $b$ 为 $a$ 的逆元。


逆关系显然是相互的。给定一个可逆元，其逆元必唯一（证明略）。

> [!definition] Definition 1.1.5: Group
> 一个带有二元运算的非空集合 $G$ 称为一个群，若下面条件成立：
> 
> 1. 结合律成立
> 2. 幺元存在
> 3. 求逆封闭

另外，只满足 1 的称为半群，只满足 1、2 的称为幺半群。

群中元素的逆可以记为 $a^{-1}$，幺元可以记为 $1_G$，运算可以称为乘法，$n$ 个元素的乘积记为 $a^n$。

如果群中元素均满足交换律，则群可以进一步称为 Abel 群。

有限个元素构成的群称为有限群，元素个数记为 $|G|$，叫做群的阶。$|G|=\infty$ 表示是无限群。对于群中元素，使 $g^n=1$ 成立的最小 $n$ 称为 $g$ 的阶，记为 $o(g)$。

> [!example] Example 1.1.6
> - 任何数域在加法下构成 Abel 群（由于被当成群，只涉及一种运算，故乘法被忽略）
> - 数域 $K$ 上全体 $n$ 阶可逆阵在乘法下构成群，称为一般线性群，记为 $GL_n(K)$
> - 只有在数域 $K$ 去掉 0 后，在乘法下才构成群（也是 Abel 群）
> - 最小的群只含幺元（平凡群）
> - 空集不是群。
> - 三维欧氏空间在向量叉乘下连半群都不是，因为不满足结合律

> [!proposition] Proposition 1.1.7: 消去律
> 群中元素 $a ,b, c$，
> 
> $$
> ab=ac \lor ba=ca \Longrightarrow b=c
> $$

> [!proposition] Proposition 1.1.8
> $(ab)^{-1}=b^{-1}a^{-1}$

### Section 1.2: 子群

> [!definition] Definition 1.2.1: 子群
> 一个群的子集，若满足乘法封闭和求逆封闭，则称为子群

由于求逆封闭，而乘法封闭，可求得子群中幺元存在。

易知 Abel 群的子群仍然是 Abel 群。

> [!example] Example 1.2.2
> - $\left\{1_G\right\}, G$ 为 $G$ 的平凡子群
> - 定义 $n\mathbb{Z} = \left\{a\in \mathbb{Z}|n|a\right\}$，则 $n\mathbb{Z}$ 是 $\mathbb{Z}$ 的子群
> - 定义特殊线性群 $SL_n(K) = \left\{M\in GL_n(K)|det(M)=1\right\}$，是 $GL_n(K)$ 的子群。

> [!proposition] Proposition 1.2.3: 子群判定
> 若 $\forall a,b\in H\subseteq G$，
> 
> $$
> ab^{-1}\in H
> $$
> 
> ，则 $H$ 是 $G$ 的子群

> [!proposition] Proposition 1.2.4
> 任意个子群的交仍是子群

> [!definition] Definition 1.2.5: 中心化子、中心
> 设 $g \in G$，称 $C(g)=\left\{a\in G|ag=ga\right\}$ 为 $g$ 在 $G$ 中的中心化子；进一步，$C(S) = \{a\in G|ag=ga,\quad \forall g \in S\}$；$C(G)$ 称为 $G$ 的中心。

易知 $C(g), C(G)$ 为 $G$ 的子群。

称 $G$ 中所有包含集合 $S$ 的子群的交（也是最小的包含 $S$ 的子群）为 $S$ 生成的群，记为 $\langle S\rangle$.

> [!example] Example 1.2.6
> - $\mathbb{Z} = \langle 1\rangle=\langle -1\rangle$
> - $GL_n(K)$ 有所有 $n$ 阶初等矩阵生成

一个元素生成的群称为循环群。

$$
\langle a\rangle =
\begin{cases}
\{\cdots, a^{-2}, a^{-1},1,a,a^2.\cdots\},\quad o(a)=\infty \\
\{1,a, a^2,\cdots,a^{n-1}\},\quad o(a) =n \\
\end{cases}
$$
> [!proposition] Proposition 1.2.7
> 设 $S$ 是 $G$ 中一个非空子集，则
> 
> $$
> \langle S \rangle = \left\{a_1^{e_1}a_2^{e_2}\cdots a_n^{e_n}|a_i \in S, e_i=\pm 1\right\}
> $$

证明：首先验证其本身是一个子群，然后说明每一个包含 $S$ 的子群都必须含有这些元素。

$\mathbb{Z}$ 除 $0, \mathbb{Z}, 2\mathbb{Z}, \ldots$ 外无其他子群，且这些子群除 0 外（非平凡子群）都是无限循环群。

### Section 1.3: 置换群

> [!definition] Definition 1.3.1: 置换
> 将双射 
> $$
> \sigma: \{1,2,\ldots,n\} \to \{1,2,\ldots,n\}
> $$
> 称为一个 $n$ 个文字的置换


可以用列表法表示一个一般的置换 $\sigma$：

$$
\begin{bmatrix}
1 & 2 & \cdots & n \\
\sigma(1) & \sigma(2) & \cdots & \sigma(n) \\
\end{bmatrix}
$$

将 $n$ 文字置换构成的集合记为 $S_n$。

可以将置换的复合作为一个二元运算。

> [!proposition] Proposition 1.3.2: 对称群、置换群
> $S_n$ 在如上二元运算下构成一个 $n!$ 阶的有限群，称为对称群，其子群称为置换群。

幺元：

$$
id: i \mapsto i
$$

> [!warning] 注意用列表法作运算时，要先看右边的置换。

实际上列表法非常累赘，第一行一点用都没有，第二行有时也可以简化，这启发我们引入轮换的概念：

> [!definition] Definition 1.3.3: 轮换
> 设 $i_1, i_2, \ldots,i_d$ 为文字集中的两两不同的文字，作 $\sigma \in S_n$ 满足:
> 
> $$
> \sigma(i_1)=i_2, \quad \sigma(i_2)=i_3,\quad \ldots,\quad \sigma(i_d)=i_1
> $$
> 且所有其他文字 $i$ 有 $\sigma(i)=i$，则称 $\sigma$ 为一个 d 轮换，记为 $(i_1i_2\ldots i_d)$

直观表示 $(142) \in S_5$：

![[fig-1-3-1.png]]


长度为 2 的轮换也叫==对换==。

任何一个置换都可以写成若干不交轮换之积。

> [!proposition] Proposition 1.3.4
> 任何一个置换可以表示为若干可交对换之积

记 $\text{sgn}(\sigma)=(-1)^r$，其中 $r$ 为有序列 $\{\sigma(i)\}_{i=1}^n$ 的逆序的个数。


对于 $\sigma \in S_n$，若 $\text{sgn}(\sigma)=1$，则 $\sigma$ 称为一个偶置换，否则为奇置换。

> [!lemma] Lemma 1.3.5
> 设 $\sigma \in S_n$，$\tau=(ij)$ 是一个对换，其中 $i<j$。若 $\sigma$ 是偶置换，则 $\sigma\tau$ 是奇置换；反之，$\sigma\tau$ 是偶置换。

> [!corollary] Corollary 1.3.4
> 对称群中奇置换和偶置换各占一半。

> [!corollary] Corollary 1.3.5
> 一个置换是偶置换（奇置换）$\Longleftrightarrow$ 它可表示为偶数（奇数）个对换的乘积。




- 刚性变换：保持长度和角度不变的变换，使得象和原图形相等的，如平移、旋转、对称（虽然在物理世界不能实现，但数学上仍然将其认作一种刚性变换）


- 二面体群：平面上正 $n$ 边形的对称群，记作 $D_n$



### Section 1.4: 陪集

> [!definition] Definition 1.4.1: 陪集
> 设 $H$ 是群 $G$ 的一个子群，$a \in G$，记 $aH = \{ah|h \in H\}$，称为 $H$ 的左陪集；同理，$Ha = \{ha|h \in H\}$ 称为 $H$ 的右陪集。

> [!proposition] Proposition 1.4.2
> 设 $a, b \in G$，则映射 $f: aH \to H, g\mapsto ba^{-1}g$ 是 $aH$ 到 $bH$ 的一个双射。 

> [!proposition] Proposition 1.4.3
> $aH=bH \Longleftrightarrow a^{-1}b \in H$

上面命题写成 $ba^{-1} \in H$ 是等价的。

> [!corollary] Corollary 1.4.4
> 若 $aH \cap bH \neq \emptyset$，则 $aH=bH$。

上面命题说明，群的左陪集要么相等，要么互不相交。这就给出了 $G$ 的一个剖分。

- 子群的指数：设 $G$ 被剖分成 $r$ 个左陪集的并，则称 $H$ 在 $G$ 中的指数为 $r$，记为 $\left(G:H\right)$。这个数可以是自然数，也可以是 $\infty$ 

同理，右陪集也给出群的一个剖分，陪集本身不一定相同，但是所得的指数是相同的。


> [!theorem] Theorem 1.4.5: Lagrange
> 设 $G$ 为有限群，$H$ 为 $G$ 的子群，则 
> 
> $$
> |G| = |H| \cdot (G:H)
> $$


> [!corollary] Corollary 1.4.6
> 设 $G$ 为有限群，$a \in G$，则 $o(a)| |G|$。

这是因为 $\langle a \rangle$ 是 $G$ 的子群，且 $o(a) = |\langle a \rangle|$。


> [!corollary] Corollary 1.4.7
> 素数阶的有限群是循环群。

> [!corollary] Corollary 1.4.8
> 设 $n=|G|<\infty$，则 $\forall a \in G$，有 $a^n=1$。

回忆：

- 等价关系
- 等价类

同样，根据（左）陪集的性质，我们可以定义如下等价关系：

$$
a \sim b \Longleftrightarrow aH = bH \Longleftrightarrow a^{-1}b \in H
$$


### Section 1.5: 正规子群和商群

上节中我们定义了在左陪集上的等价关系，所有等价类的集合为：

$$
\Gamma =G/\sim H
$$

设 $aH \in \Gamma$，则称 $a$ 为该陪集的==代表元==。

很自然地我们想到，能否在 $\Gamma$ 上定义一个群结构，使得陪集之间也可以进行运算？这样，我们首先需要在陪集上给出一个看起来合理的运算：

$$
aH \cdot bH = abH
$$

然而，这个定义有一个问题：它是否是良定义的，即与代表元的选取无关？也就是说，若 $aH=a'H, bH=b'H$，是否总有 $abH=a'b'H$？实际上，这并不总是成立的。为使 $abH=a'b'H$，需要 $b^{-1}a^{-1}a'b' \in H$，而可以将 $a', b'$ 写成 $a'=ah_1, b'=bh_2$，其中 $h_1, h_2 \in H$，则需要 $b^{-1}h_1b \in H$，这就要求对于任意 $h \in H, b \in G$，都有 $b^{-1}hb \in H$。由此可见，要使 $abH=a'b'H$，需要 $H$ 满足一个额外条件（也是充要条件）：

- 对于任意 $h \in H, g \in G$，都有 $g^{-1}hg \in H$。

> [!definition] Definition 1.5.1: 正规子群
> 群 $G$ 的子群 $H$，若满足 $\forall h \in H, g \in G$，都有 $g^{-1}hg \in H$，则称 $H$ 为 $G$ 的==正规子群==，记为 $H \triangleleft G$。

不难验证，在 $H$ 的左陪集的乘法下，定义在 $H$ 上的 $\Gamma$ 就具有群结构，称为==商群==，记为 $G/H$。其幺元为 $H$ 本身，任意 $aH \in G/H$ 的逆元为 $a^{-1}H$。

商群的阶等于 $(G:H)$。

> [!proposition] Proposition 1.5.2
> 设 $H$ 是 $G$ 的一个子群，则 $H\triangleleft G \Longleftrightarrow aH=Ha, \quad \forall a\in G$.

> [!proposition] Propositions 1.5.3
> 1. 平凡子群和群本身都是正规子群
> 2. Abel 群的任意子群都是正规子群
> 3. 设 $H \triangleleft G$，$K$ 是 $G$ 包含 $H$ 的一个子群，则 $H \triangleleft K$。


> [!example] Important Example 1.5.4
> 设 $n \in \mathbb{N}>1$，令 $n\mathbb{Z}$ 为被 $n$ 整除的整数构成的集合，则 $\mathbb{Z}/n\mathbb{Z}$ 是一个 $n$ 阶循环群，记为 $\mathbb{Z}_n$。其元素可列举如下：
> 
> $$
> \mathbb{Z}_n = \{\overline{0}, \overline{1}, \overline{2}, \ldots, \overline{n-1}\}
> $$


- \*射影一般线性群


- 共轭子群：设 $H$ 是 $G$ 的一个子群，$a \in G$，则称 $a^{-1}Ha = \{a^{-1}ha|h \in H\}$ 为 $H$ 的一个共轭子群。

子群的共轭子群和原子群同阶，但不一定相等。容易验证 $H\triangleleft G$ 当且仅当 $H$ 的所有共轭子群都等于 $H$ 本身。

> [!proposition] Propositions 1.5.5
> 1. 群的中心是正规子群
> 2. 正规子群的任意交是正规子群
> 3. 设 $S$ 是 $G$ 的一个子集，若 $g^{-1}sg \in S, \quad \forall s \in S, g \in G$，则 $\langle S \rangle \triangleleft G$。

- 换位子群：特别地，取 $S=\{a^{-1}b^{-1}ab|a,b \in G\}$，显然也满足 Propo. 1.5.5 (3)，则 $\langle S \rangle$ 称为 $G$ 的换位子群，记为 $\left[G,G\right]$。 


> [!definition] Definition 1.5.6: 正规化子
> 设 $H$ 是 $G$ 的一个子群，称 $N_G(H) = \{g \in G|g^{-1}hg \in H, \quad \forall h \in H\}$ 为 $H$ 在 $G$ 中的==正规化子==。

$N_G(H)$ 是 $G$ 的一个包含 $H$ 的子群，且 $H \triangleleft N_G(H)$。若 $H\subseteq K$ 且 $H \triangleleft K$，则 $K \subseteq N_G(H)$。换言之， $N_G(H)$ 是包含 $H$ 的最大的子群，使得 $H$ 在其中是正规子群。

> [!warning] $H_1 \triangleleft H_2 \triangleleft G$ 不能推出 $H_1 \triangleleft G$。


> [!proposition] Proposition 1.5.7
> 设群 $G$，则商群 $G/\left[G,G\right]$ 是一个 Abel 群。若 $H\triangleleft G$ 且 $G/H$ 是 Abel 群，则 $\left[G,G\right] \subseteq H$。

> [!definition] Definition 1.5.8: 单群
> 群 $G$，若其只有平凡子群和群本身两个正规子群，则称 $G$ 为==单群==。




### Section 1.6: 交错群

- 交错群：对称群 $S_n$ 的所有偶置换构成的子群，记为 $A_n$，称为 $n$ 阶交错群。

> [!theorem] Theorem 1.6.1
> $A_n \triangleleft S_n$，且 $\left(S_n:A_n\right)=2, \quad n>1$。

上面的定理可以通过构造一个双射 $\sigma \mapsto (12)\sigma$ 来证明。


> [!proposition] Proposition 1.6.2
> 设 $\sigma=(i_1i_2\ldots i_d)$ 是一个 $r$ 轮换，$\tau \in S_n$，则 $\tau\sigma\tau^{-1} = (\tau(i_1)\tau(i_2)\ldots \tau(i_d))$ 也是一个 $r$ 轮换。

> [!corollary] Corollary 1.6.3
> 设 $\sigma = (a_1a_2\ldots a_r)\ldots(c_1c_2\ldots c_s)$ 是一些互不相交的轮换的乘积，$\tau \in S_n$，则
> $$
> \tau\sigma\tau^{-1} = (\tau(a_1)\tau(a_2)\ldots \tau(a_r))\ldots(\tau(c_1)\tau(c_2)\ldots \tau(c_s))
> $$

> [!proposition] Proposition 1.6.4
> $$
> S_n=\langle (12),(13),\ldots, (1n) \rangle=\langle (12),(23),\ldots,(n-1 n) \rangle,
> $$
> 
> $$
> A_n=\langle (123),(124),\ldots,(12n) \rangle.
> $$

> [!question] Propo. 1.6.4 怎么证明的？


> [!theorem] Theorem 1.6.5
> 设 $n \geq 5$，则 $A_n$ 是单群。

> [!question] Theorem 1.6.5 怎么证明的？







### Section 1.7: 群的同态


> [!definition] Definition 1.7.1: 同态
> 设群 $G_1$，$G_2$，映射 $f: G_1 \to G_2$，若对 $\forall a,b \in G_1$，都有
> $$
> f(ab) = f(a)f(b)
> $$
> 则称 $f$ 为从 $G_1$ 到 $G_2$ 的一个==同态==。

易知 $f(e_1)=e_2$，且 $f(a^{-1}) = f(a)^{-1}$。

- kernel：$G_1$ 的一个正规子群，记为 $\ker (f) = \{a \in G_1|f(a)=e_2\}$。

- 单同态： 若 $\ker (f) = \{e_1\}$，则称 $f$ 为单同态。

- 象： $\text{Im}(f)=\{f(a)|a \in G_1\}$，是 $G_2$ 的一个子群。

- 满同态： 若 $\text{Im}(f)=G_2$，则称 $f$ 为满同态。

- 同构： 若 $f$ 是一个既是单同态又是满同态（双射），则称 $f$ 为同构，记为 $G_1 \cong G_2$。

- 自同构：群 $G$ 到其自身的同构，其全体记为 $\text{Aut}(G)$。

- 平凡同态：将所有元素映为幺元的同态。

- 自然同态：设 $H \triangleleft G$，则映射 $f: G \to G/H, a \mapsto aH$ 是一个满同态，称为从 $G$ 到商群 $G/H$ 的==自然同态==。 


- 内自同构： 设 $G$ 为一个群，$a \in G$，则映射 $f_a: G \to G, g \mapsto a^{-1}ga$ 是 $G$ 的一个自同构，称为由 $a$ 诱导的==内自同构==。所有内自同构构成 $\text{Aut}(G)$ 的一个子群，记为 $\text{Inn}(G)$。

> [!proposition] Proposition 1.7.a
> （这是一个作业题中关于内自同构与中心的结论）
> 设群 $G$，则映射 $\varphi: G \to \text{Inn}(G), a \mapsto f_a$ 是一个满同态，且 $\ker (\varphi) = C(G)$。因此，$G/C(G) \cong \text{Inn}(G)$。

> [!theorem] Theorem 1.7.2: 同态基本定理
> 设 $f: G_1 \to G_2$ 是一个群同态，则：
> 
> 1. $\ker (f) \triangleleft G_1$
> 2. $G_1/\ker (f) \cong \text{Im}(f)$

第二条是非常有用的，当我们需要证明一个商群 $G/H$ 与某个群同构时，可以构造一个从 $G$ 到该群的满同态，验证其 kernel 恰好为 $H$ 即可。

> [!lemma] Lemma 1.7.3
> 设群同态 $f: G_1 \to G_2$，$H_2$ 是 $G_2$ 的一个子群，则 $f^{-1}(H_2) = \{a \in G_1|f(a) \in H_2\}$ 是 $G_1$ 的一个子群且包含 $\ker (f)$。


> [!proposition] Proposition 1.7.4
> 设 $H\triangleleft G$，$f:G \to G/H$ 为自然同态。令 $\Gamma$ 为所有包含 $H$ 的子群构成的集合，$\Gamma'$ 为 $G/H$ 的所有子群构成的集合，则映射 $f^{-1}: \Gamma' \to \Gamma, K \mapsto f^{-1}(K)$ 是 $\Gamma'$ 到 $\Gamma$ 的一个双射。

> [!check] 这里需要仔细拆解一下




> [!theorem] Theorem 1.7.5: 第一同构定理
> 设 $H\triangleleft G$，$N\triangleleft G$ 且 $H \subseteq N$，则 
> 
> $$
> (G/H)/(N/H)\cong G/N
> $$

> [!hint] 记忆：联想分数通分

证明第一同构定理需要用到同态基本定理。



> [!theorem] Theorem 1.7.6: 第二同构定理
> 设 $H\triangleleft G$，$K$ 是 $G$ 的一个子群，令
> 
> $$
> KH = \{kh|k \in K, h \in H\},
> $$
> 
> 则 $KH$ 是 $G$ 的一个子群，且 
> 
> $$
> KH/H \cong K/(K \cap H)
> $$

> [!hint] 记忆：
> ![[fig-1-7-1.png]]

同样地，使用同态基本定理证明。

### Section 1.8: 群的直积

前文子群和商群是“从大到小”的构造，而直积则是“从小到大”的构造。

回忆 $n$ 维欧氏空间 $\mathbb{R}^n$ 是 $n$ 个实数域 $\mathbb{R}$ 的直积，类似地，设 $G_1, G_2, \ldots, G_n$ 是 $n$ 个群，则给出一个笛卡尔积：

$$
G_1 \times G_2 \times \cdots \times G_n = \{(g_1, g_2, \ldots, g_n)|g_i \in G_i\}
$$

并定义如下二元运算：

$$
(g_1, g_2, \ldots, g_n)(h_1, h_2, \ldots, h_n) = (g_1h_1, g_2h_2, \ldots, g_nh_n)
$$

> [!definition] Definition 1.8.1: 直积
> 上述二元运算下的集合称为群 $G_1, G_2, \ldots, G_n$ 的==直积==，记为 $G_1 \times G_2 \times \cdots \times G_n$。

- 直和：组成直积的各个群均为 Abel 群且群运算符为加号时，也称为直和，记为 $G_1 \oplus G_2 \oplus \cdots \oplus G_n$

> [!proposition] Proposition 1.8.2: 两个关键同态
> 设 $G_1, G_2, \ldots, G_n$ 是 $n$ 个群，对每个 $i$，有如下两个同态：
> 
> - $j_i: G_i \to G_1 \times G_2 \times \cdots \times G_n, g_i \mapsto (1_{G_1}, \ldots, 1_{G_{i-1}}, g_i, 1_{G_{i+1}}, \ldots, 1_{G_n})$
> 
> 显然 $j_i$ 是一个单同态，且 $j_i(G_i) \cong G_i$。
> 
> 对任何 $(g_1, g_2, \ldots, g_n) \in G_1 \times G_2 \times \cdots \times G_n$，都有 $(g_1, g_2, \ldots, g_n) = j_1(g_1)j_2(g_2)\cdots j_n(g_n)$，等式右边的乘积可以交换次序。
> 
> - $p_i: G_1 \times G_2 \times \cdots \times G_n \to G_i, (g_1, g_2, \ldots, g_n) \mapsto g_i$
> 
> $p_i$ 是满同态，可以看作投影。

容易看出，$G_i\triangleleft G_1 \times G_2 \times \cdots \times G_n$。

更加困难的任务是将一个群分解为若干群的直积（直和）。最理想的情形是，每一个 $G_i$ 都不能再被分解了。此时如果 $G_i$ 是熟悉的群，则认为 $G$ 的群结构已经搞清楚了。

根据前面提到的两个关键同态，如果同构 $\varphi:  G_1 \times G_2 \times \cdots \times G_n\to G$ 存在，则 $H_i = \varphi \circ j_n(G_i)$ 是 $G$ 的一个正规子群，且 $a_ia_j=a_ja_i$ 对 $\forall a_i \in H_i, a_j \in H_j, i \neq j$ 成立。这启发我们，如果需要对 $G$ 进行直积分解，可以先寻找一些满足上述性质的子群。

> [!theorem] Theorem 1.8.3
> 设 $G,G_1,G_2, \ldots, G_n$ 是群，则
> $$
> G \cong G_1 \times G_2 \times \cdots \times G_n
> $$
> 当且仅当存在 $G$ 的子群 $H_1, H_2, \ldots, H_n$，使得：
> 1. $H_i\cong G_i, \quad i=1,2,\ldots,n$；
> 2. $a_ia_j=a_ja_i, \quad \forall a_i \in H_i, a_j \in H_j, i \neq j$；
> 3. 映射 $\varphi: H_1 \times H_2 \times \cdots \times H_n \to G, (a_1, a_2, \ldots, a_n) \mapsto a_1a_2\cdots a_n$ 是一个双射。


### Section 1.9*: 有限循环群的自同构和 Euler 函数

本章不在考试范围内，暂无记录

### Section 1.10: 群作用

> [!definition] Definition 1.10.1: 群作用
> 设群 $G$ 和集合 $S$， 映射
> 
> $$
> \sigma: G \times S \to S
> $$
> 
> 满足：
> 
> 1. $\sigma(e,x) = x, \quad \forall x \in S$；
> 2. $\sigma(g_1g_2,s) = \sigma(g_1,\sigma(g_2,s)), \quad \forall g_1,g_2 \in G, s \in S$；（结合律）
> 
> 则称 $G$ 在 $S$ 上有一个==（左）作用==。

固定一个 $g\in G$，$x\mapsto\sigma(g,x)$ 是 $S$ 到其自身的一个映射。

用 $gx$ 表示 $\sigma(g,x)$ 更加简明，但要注意这里和群中元素的乘法不同，这里应该理解为 $g$ 是主动去作用的元素，或者干脆直接当成一个算子。

类似地，可以定义右作用，但是需要注意结合律变为：

$$
x(g_1g_2) = (xg_1)g_2
$$

因此，左作用是先 $g_2$ 后 $g_1$，而右作用是先 $g_1$ 后 $g_2$。

> [!proposition] Proposition 1.10.2
> 设群 $G$ 在集合 $S$ 上有一个作用，则对 $\forall g \in G$，映射 $f_g: S \to S, x \mapsto gx$ 是 $S$ 到其自身的一个双射。


- 左平移：$G$ 在自身上的左作用，定义为 $\sigma(g,x) \mapsto gx$。

- 共轭作用： $G$ 在自身上的左作用，定义为 $\sigma(g,x) \mapsto gxg^{-1}$。

注意 $g^{-1}xg$ 是一个右作用，这一点可以通过结合律验证。


> [!definition] Definition 1.10.3: 轨道
> 设群 $G$ 在集合 $S$ 上有一个作用，$x \in S$，则称
> 
> $$
> Gx = \{gx|g \in G\}
> $$
> 
> 为 $x$ 在 $S$ 上的==轨道==。

轨道的元素个数也叫做轨道的长度。

> [!definition] Definition 1.10.4: 稳定子群
> 设群 $G$ 在集合 $S$ 上有一个作用，$x \in S$，则称
> 
> $$
> \text{Stab}(x) = \{g \in G|gx=x\}
> $$
> 
> 为 $x$ 在 $G$ 中的==稳定子群==。

容易验证 $\text{Stab}(x)$ 是 $G$ 的一个子群，可以简记为 $G_x$。

> [!lemma] Lemma 1.10.5
> 设 $x,y\in S$，若 $Gx\cap Gy \neq \emptyset$，则 $Gx=Gy$。

Lemm. 1.10.5 说明，群作用的轨道要么相等，要么互不相交。这就给出了 $S$ 的一个剖分。

> [!definition] Definition 1.10.6: 可迁
> 只有一条轨道的群作用是==可迁==的。

可迁还有一个充要条件：

$$
\forall x,y \in S, \exists g \in G, \text{s.t. } gx=y
$$

轨道和稳定子群之间有密切联系：

> [!theorem] Theorem 1.10.7
> 记 $\Gamma$ 为 $\text{Stab}(x)$ 在 $G$ 中的左陪集全体，则映射
> 
> $$
> \varphi: \Gamma \to Gx, \quad g\text{Stab}(x) \mapsto gx
> $$
> 
> 是双射（一一对应）。

> [!corollary] Corollary 1.10.8
> 设群 $G$ 在集合 $S$ 上有一个作用，$x \in S$，则有
> 
> $$
> |Gx| = (G:\text{Stab}(x))
> $$



> [!theorem] Theorem 1.10.9
> （这个定理在《讲义》中仅用两句话带过，但是课上详细讲解）
> 
> 设 $G$ 在有限集合 $S$ 上有一系列左作用，则这些左作用的集合与群同态
> 
> $$
> \varphi: G \to S_n,\quad n=|S|
> $$
> 
> 是一一对应的。 

换句话说，$F: \{G \text{ 在 } S \text{ 上的左作用}\} \to \{\text{群同态 } G \to S_n\}$ 是双射。


这个定理有一个推论：

> [!corollary] Corollary 1.10.10
> 对任意有限群 $G$，可以看作某个对称群的子群。

用左平移构造一个单同态来证明。



## Chapter 2: 环和域的基本知识

### Section 2.1: 基本定义

> [!definition] Definition 2.1.1: 环
> 集合 $R$ 配备有两个二元运算 $+$ 和 $\cdot$，满足：
> 
> 1. $(R, +)$ 是一个 Abel 群
> 2. $a\cdot(b\cdot c) = (a\cdot b)\cdot c, \quad \forall a,b,c \in R$（乘法结合律）
> 3. $a\cdot(b+c) = ab + ac, \quad (a+b)\cdot c = ac + bc, \quad \forall a,b,c \in R$（分配律）
> 
> 则称 $(R, +, \cdot)$ 为一个==环==。

加法幺元在环论中称为零元，记为 $0$。乘法幺元（如有）称为幺元，记为 $1$。

- 交换环： 若 $\forall a,b \in R$，都有 $ab=ba$，则称 $R$ 为==交换环==。

- 零环：只有一个元素（既是零元也是幺元）的环

> [!proposition] Proposition 2.1.2: 基本性质
> 1. 幺元存在则唯一
> 2. $a0=0a=0, \quad \forall a \in R$
> 3. $(-a)b=a(-b)=-(ab), \quad \forall a,b \in R$
> 4. 若 $ab=ba$，则 $(a+b)^n = \sum_{k=0}^n \binom{n}{k} a^k b^{n-k}, \quad \forall n \in \mathbb{N}$

还有一些容易验证的性质：对于 $\forall n, m\in \mathbb{Z}, a,b \in R$，有

- $na + ma = (n+m)a$
- $n(ma) = (nm)a$
- $n(a+b) = na + nb$

非零环中，幺元不等于零元。

> [!definition] Definitions 2.1.3: 左零因子、右零因子、左逆元、右逆元
> 设 $R$ 是一个环，$a,b \in R$，则：
> 
> - 若 $a \neq 0$ 且 $ab=0$
> 	- $a$ 为 $R$ 中的一个==左零因子==
> 	- $b$ 为 $R$ 中的一个==右零因子==
> 	- $R$ 为交换环时两者无区别
> - 若 $ab=1$
> 	- $a$ 是 $b$ 的==左逆元==
> 	- $b$ 是 $a$ 的==右逆元==
> 	- 若 $ab=ba=1$，则称 $a$ 和 $b$ 互为==乘法逆元==

- 整环：不存在零因子的环

- 整区：交换整环

- 可除环：环中任意非零元的逆元存在

- 域：交换可除环

- 环的直积：与群直积的定义类似

### Section 2.2: 理想和商环

> [!definition] Definition 2.2.1: 子环
> 设 $R$ 是一个环，$S \subseteq R$，若 $S$ 在加法下是是 $R$ 的一个子群，且满足幺元存在、乘法封闭，则称 $S$ 为 $R$ 的一个==子环==。

子环继承了大部分性质，如交换性、整环性（要求子环非零），但不是所有性质都继承，如非交换环的子环可能是交换环、域的子环可能不是域（不可除）。

- 子域：子环如果是域，则称为子域。

现在考虑环 $R$ 的一个加法子群 $I$，由于 $R$ 在加法下是 Abel 群，$I\triangleleft_+R$，于是商群 $R/I$ 有意义。现在，能否利用 $R$ 的乘法在 $R/I$ 上定义一个乘法，使其也成为一个环（这样就定义了商环）？  

为此，我们尝试给出一种看似合理的乘法：

$$
(a+I)(b+I) = ab + I
$$

同样地，这个定义是否良定义？即与代表元的选取无关？设 $a+I=a'+I, b+I=b'+I$，则需要 $a'b' - ab \in I$。将 $a', b'$ 写成 $a'=a+i_1, b'=b+i_2$，其中 $i_1, i_2 \in I$，则需要 $ai_2 + i_1b + i_1i_2 \in I$。因此，为了使乘法良定义，需要 $I$ 满足一个额外条件（也是充要条件）：

- 对于任意 $a \in R, u \in I$，，都有 $au \in I$ 且 $ua \in I$。

> [!definition] Definition 2.2.2: （双边）理想
> 设环 $R$，$I$ 是 $R$ 的一个加法子群，若满足 $\forall a \in R, u \in I$，都有 $au \in I$ 且 $ua \in I$，则称 $I$ 为 $R$ 的一个==（双边）理想==。


> [!proposition] Proposition 2.2.3: 商环
> 设 $I$ 是环 $R$ 的一个双边理想，则商群 $R/I$ 在上述乘法下成为一个环，称为 $R$ 关于理想 $I$ 的==商环==。

只需验证乘法结合律和分配律即可。


类似地有左理想、右理想。

需要注意的是理想不是子环，因为幺元一般不在理想中。（非平凡的理想中幺元一定不在其中，否则理想就是整个环了。）

- 理想的同余： 若 $I$ 是环 $R$ 的一个理想，$a,b \in R$，若 $a - b \in I$，则称 $a$ 与 $b$ 关于理想 $I$ 是==同余==的，记为 $a \equiv b\ (\mod I)$。

- 单环： 环 $R$，若其不可除，且只有平凡理想 （$\{0\}$ 和环本身 $R$） 两个理想，则称 $R$ 为==单环==。

> [!proposition] Proposition 2.2.4
> 1. 设 $\{I_\lambda\}_{\lambda \in \Lambda}$ 是环 $R$ 的一族理想，则 $\bigcap_{\lambda \in \Lambda} I_\lambda$ 也是 $R$ 的一个理想。
> 2. 设 $I,J$ 是环 $R$ 的理想，则 $I+J = \{i+j|i \in I, j \in J\}$ 也是 $R$ 的一个理想。更一般地，对于任意理想族 $\{I_\lambda\}_{\lambda \in \Lambda}$，都有 $$\sum_{\lambda \in \Lambda} I_\lambda = \left\{\sum_{\lambda} a_\lambda|a_\lambda \in I_\lambda, \text{ 和式中只有有限多项不为零}\right\}$$ 是 $R$ 的一个理想。
> 3. 设 $I,J$ 是环 $R$ 的理想，则 $$IJ = \left\{\sum_{i=1}^n a_i b_i|a_i \in I, b_i \in J, n \in \mathbb{N}\right\}$$ 也是 $R$ 的一个理想。

> [!example] Important Example 2.2.5
> 环 $\mathbb{Z}$ 中，令 $I=n\mathbb{Z},J=m\mathbb{Z}$，则 $$ \begin{aligned} I\cap J & =\text{lcm}(m,n)\mathbb{Z} , \\ I+J& =\text{gcd}(m,n)\mathbb{Z}, \\ IJ& =mn\mathbb{Z} \\ \end{aligned}$$

这启发我们给出如下定义：

> [!definition] Definition 2.2.6: 环的互素
> 设环 $R$，$I,J$ 是 $R$ 的两个理想，若 $I+J=R$，则称 $I$ 和 $J$ 在环 $R$ 中是==互素==的。

这样，加上前面同余的定义，在形式上就与数论中的同名概念统一。


- 生成的理想：设环 $R$，$S \subseteq R$，则包含 $S$ 的所有理想的交，记为 $\langle S \rangle$，称为由 $S$ 在 $R$ 中==生成的理想==

与群论类似，有

$$
\langle S \rangle = \left\{\sum_{i=1}^n a_i u_i b_i|a_i,b_i \in R, u_i \in S, n \in \mathbb{N}\right\}
$$

证明也类似，即先验证右边的集合是一个理想，然后再验证它包含于所有包含 $S$ 的理想中。

如果 $S$ 是有限集合，则生成的理想称为==有限生成理想==。

- 主理想：一个元素生成的理想



### Section 2.3: 环的同态

> [!definition] Definition 2.3.1: 同态
> 设环 $R_1, R_2$，映射 $f: R_1 \to R_2$，若对 $\forall a,b \in R_1$，都有
> 
>1. $f(a+b) = f(a) + f(b)$
> 2. $f(ab) = f(a)f(b)$
> 3. $f(1_{R_1}) = 1_{R_2}$
> 
> 则称 $f$ 为从环 $R_1$ 到环 $R_2$ 的一个==同态==。

环同态同时也是加法下的群同态。但与群同态不同的是，由于没有消去律，无法保证将幺元映为幺元，因此需要单独列出第三条。

下文在不引起歧义的情况下，“同态”均指环同态。

显然同态将零元映成零元。

- kernel： $\ker (f) = \{a \in R_1|f(a)=0_{R_2}\}$
- 单同态： 若 $\ker (f) = \{0_{R_1}\}$，则称 $f$ 为单同态。
- 象，满同态，同构，自同构：与群论同态类似。


> [!theorem] Theorem 2.3.2: 同态基本定理
> 设环同态 $f: R_1 \to R_2$，则：
> 1. $\ker (f)$ 是 $R_1$ 的一个理想
> 2. $R_1/\ker (f) \cong \text{Im}(f)$

> [!proposition] Proposition 2.3.3
> 设 $I$ 是环 $R$ 的一个理想，$f:R \to R/I$ 为自然同态。令 $\Gamma$ 为所有包含 $I$ 的理想构成的集合，$\Gamma'$ 为 $R/I$ 的所有理想构成的集合，则映射 $f^{-1}: \Gamma' \to \Gamma, K \mapsto f^{-1}(K)$ 是 $\Gamma'$ 到 $\Gamma$ 的一个双射（一一对应）。


> [!proposition] Proposition 2.3.4
> 设 $S$ 是 $R$ 的一个子环，$I$ 是 $R$ 的一个理想，则 $S+I = \{s+i|s \in S, i \in I\}$ 是 $R$ 的一个子环，$I$ 是 $S+I$ 的一个理想，$S\cap I$ 是 $S$ 的一个理想，且有同构
> 
> $$
> (S+I)/I \cong S/(S \cap I)
> $$
