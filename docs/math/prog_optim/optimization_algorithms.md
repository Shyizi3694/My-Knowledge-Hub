# 优化实用算法

## Chapter 1: Introduction

> [!cite] 笔者的话
> - 教材《最优化理论与方法》这本书编的……一言难尽，遂直接对 Slides 进行记录。

### Section 1.1: Introduction

- Optimization 是在某些约束条件（constrains）下求目标函数（function subject）的最小或最大

- Notation
	- 未知向量：$x$
	- 目标函数：$f(x)$
	- 约束条件：$c_i$（$x$ 的标量函数，定义其满足的方程/不等式）

- Standardized Formulation: 

$$
\begin{array}{cl}
\min & f(x) \\
\text{s.t.} & c_i(x) = 0, \quad i\in \mathcal{E}=\{1, \ldots, m_e\} \\
& c_j(x) \ge 0, \quad j\in \mathcal{J}=\{m_e+1,\ldots,m\}\\
\end{array}
$$

其中 $\mathcal{E}$ 是等式约束集，$\mathcal{J}$ 是不等式约束集。

> [!example] 运输问题


- Classification

优化问题可以按如下方式分类：

> [!note] 
> - 按目标函数和约束函数性质分类：
> 	- Linear, Non-linear, Contex, Constrained, Unconstrained
> - 按未知量性质分类：
> 	- Large, Small, Continuous, Discrete
> - 按函数光滑性分类：
> 	- Differentiable, Non-differentiable
> - 优化的目标
> 	- Global, Local
> - 随机（Stochastic）或确定（Deterministic）的



- Algorithm in Computational Mathematics

指一种产生一个逼近解序列的迭代方法。如果给定初始序列，产生的逼近解序列收敛，则称迭代方法收敛。

一般来说先猜一个 $x$，生成一系列改进的估计（即 iterates），直到终止 ，希望这时改进的结果尽可能接近精确解。

Strategy：区别 algorithm 的核心，决定一个 iterate 如何移动到下一个 iterate

> [!property] 对优化算法的要求
> - 健壮 Robustness
> - 高效 Efficiency
> - 精确 Accuracy

这些目标可能是冲突的，因此需要权衡（trade off）

### Section 1.2: Mathematical Foundation

#### Norm

范数：

$$
\|\cdot\|: \mathbb{R}^n \to \mathbb{R}
$$

且：

- $\|x\|\ge 0, \quad\forall x \in \mathbb{R}^n$
- $\alpha x =|\alpha|\|x\|,\quad \alpha \in \mathbb{R}, x\in \mathbb{R}^n$
- 三角不等式
- $\|x\|=0 \Longleftrightarrow x=0$

常见范数：

- $l_p$ Norm

$$
\|x\|_p=\left(\sum_{i=1}^n|x_i|^p\right)^{\frac{1}{p}}
$$

$p=1,2,\infty$ 是最常见的


- 向量范数诱导矩阵范数

设 $A\in \mathbb{R}^{n\times n}$，向量范数 $\|\cdot \|$ 诱导矩阵范数为：

$$
\|A\|=\max_{x\ne 0}\left\{ \frac{\|Ax\|}{\|x\|} \right\}
$$


特别的，$l_1$ 诱导矩阵列和范数，$l_\infty$ 诱导行和范数。

$l_2$ 诱导谱范数 ：

$$
\|A\|_2=\left(\lambda_{A^TA}\right)^{1/2}
$$

其中 $\lambda_{A^TA}$ 代表 $A^TA$ 的最大特征值。


常用的还有 Frobenius 范数：

$$
\|A\|_F=\left(\sum_{i=1}^n\sum_{j=1}^n|a_{ij}|^2\right)^{1/2} = \left[tr\left(A^TA\right)\right]^{1/2}
$$

- 加权的范数：

$$
\|A\|_M=\|MAM\|
$$

其中 $M$ 为对称正定矩阵。


- 矩阵范数的相容性条件：

$$
\|AB\|\le \|A\|\|B\|
$$


- 正交不变矩阵范数

$$
\|UA\|=\|A\|
$$

显然谱范数和 Frobenius 范数都是正交不变范数


- 范数等价

$\exists \mu_1, \mu_2 >0$: 

$$
\mu_1\|x\|_\alpha \le \|x\|_\beta \le \mu_2\|x\|_\alpha
$$


- 依范数收敛

$$
\lim_{k\to \infty}\|x_k-x^*\|=0
$$


> [!theorem]+ Inequalities on Norms
> - Cauchy-Schwarz
> 
> $$
> |x^Ty| \le \|x\|\|y\|
> $$
> 
> - $|x^TAy|\le \|x\|_A\|y\|_A\Longleftrightarrow x \text{ 和 } y \text{线性相关}$
> - $|x^Ty|\le \|x\|_A\|y\|_{A^{-1}}\Longleftrightarrow x \text{ 和 } A^{-1}y \text{线性相关}$
> - Young's
> $p,q>1, \frac{1}{p}+\frac{1}{q}=1, x,y\in \mathbb{R}$, then
> 
> $$
> xy\le\frac{x^p}{p}+\frac{y^q}{q}\Longleftrightarrow x^p=y^q
> $$
> 
> - Holder's
> $p, q>1, \frac{1}{p}+\frac{1}{q}=1$, then
> 
> $$
> |x^Ty|\le\|x\|_p\|y\|_p \le \left(\sum_{i=1}^n|x_i|^p\right)^{1/p}\left(\sum_{i=1}^n|x_i|^q\right)^{1/q}
> $$
> 
> - Minkowski's
> $p\ge1$, then
> 
> $$
> \|x+y\|_p\le\|x\|_p+\|y\|_p
> $$


#### Inversion of Matrix

> [!lemma] von Neumann Lemma
> 设 $E\in \mathbb{R}^{n\times n}$, $I$ 是单位矩阵, $\|\cdot\|$ 是满足 $\|I\|=1$ 的相容矩阵范数。
> - 若 $\|E\|<1$，则 $(I-E)$ 非奇异，且
> 
> $$
> (I-E)^{-1}=\sum_{k=0}^\infty E^k, \quad \|(I-E)^{-1}\|\le\frac{1}{1-\|E\|}
> $$
> - 若 $A$ 非奇异，$\|A^{-1}(B-A)\|<1$，则 $B$ 也非奇异，且
> 
> $$
> B^{-1}=\sum_{k=0}^\infty (I-A^{-1}B)^kA^{-1}, \quad \|B^{-1}\|\le \frac{\|A^{-1}\|}{1-\|A^{-1}(B-A)\|}
> $$

这说明当 $B$ 充分靠近一个可逆阵 $A$ 时，$B$ 也可逆。

等价形式：

设 $A,B \in \mathbb{R}^{n\times n}$，$A$ 可逆，$\|A^{-1}\|\le\alpha$，如果 $\|A-B\|\le\beta$，$\alpha\beta<1$，则 $B$ 可逆，且

$$
\|B^{-1}\|\le \frac{\alpha}{1-\alpha\beta}
$$

#### Rank One Update






## Chapter 2: Line Search Methods


### Section 2.1: General Description


- 一般迭代格式：$x_{k+1}=x_k+\alpha_k p_k$

因此，关键是构造搜索方向 $p_k$ 和步长 $\alpha_k$。

设 $\phi(\alpha)=f(x_k+\alpha p_k)$，则需要确定 $\alpha_k$ 使得 $\phi(\alpha_k)<\phi(0)$。

$\alpha_k = \arg\min_{\alpha>0} \phi(\alpha)$ 是理想的选择，称为精确线搜索。如果 $\alpha_k$ 使目标函数得到可接受的下降量，即 $\Delta f = f(x_k)-f(x_k+\alpha_k p_k)>0$ 的量可以接受，则称为非精确线搜索。

对于以为搜索，首先需要确定包含最优解的区间作为搜索区间，然后采用某种分割技术或插值来逐步缩小这个区间。

设 $\alpha^*$ 是 $\phi(\alpha)$ 的最优解，则 $\varphi(\alpha^*)=\min_{\alpha \ge 0}\varphi(\alpha)$。若存在 $[a,b]\subset [0,\infty)$，使得 $\alpha^*\in [a,b]$，则称 $[a,b]$ 为一维极小化 $\varphi(\alpha)$ 的搜索区间。

一种简单的确定搜索区间的方法：进退法

> [!definition] Definition: 进退法确定搜索区间
> 1. 选取初始数据：给定 $\alpha_0,h_0>0$，加倍系数 $t>1$，计算 $\varphi(\alpha_0)$，设 $k=0$
> 2. 比较目标函数值：令 $\alpha_{k+1}=\alpha_k + h_k$，计算 $\varphi(\alpha_{k+1})$，并与 $\varphi(\alpha_k)$ 比较，如果 $\varphi(\alpha_{k+1})<\varphi(\alpha_k)$，则转步 3；否则转步 4
> 3. 加大搜索步长：令 $h_{k+1}=t\cdot h_k$，$k=k+1$，$\alpha=\alpha_k$，转步 2
> 4. 反向搜索：若 $k=0$，转换搜索方向，令 $h_k=-h_k$，转步 2；否则，停止迭代，令 $$a=\min\{\alpha, \alpha_{k+1}\},\qquad b=\max\{\alpha, \alpha_{k+1}\}$$



> [!definition]- Definition: 单峰/单谷函数
> 设函数 $\varphi(\alpha)$ 在区间 $[a,b]$ 上连续，若存在唯一的点 $\alpha^*\in (a,b)$，使得 $\varphi(\alpha)$ 在 $[a,\alpha^*)$ 上单调递减，在 $(\alpha^*,b]$ 上单调递增，则称 $\varphi(\alpha)$ 在 $[a,b]$ 上为单峰函数，$\alpha^*$ 为其极小点。

下面给出精确线搜索的算法。在此之前，补充向量之间的夹角的定义：

> [!definition]- Definition: 向量夹角
> 设 $\theta_k=\langle p_k, \nabla f(x_k)\rangle$ 为向量 $p_k$ 和 $\nabla f(x_k)$ 之间的夹角，则有：$$\cos \theta_k = \frac{p_k^T\nabla f(x_k)}{\|p_k\|\|\nabla f(x_k)\|}$$

> [!algorithm] Algorithm: 精确线搜索
> ```cpp
> 给定 x_0 in\mathbb{R}^n，0 \le \varepsilon <1;
> for k=0,1,2,\ldots do
>     计算搜索方向 p_k；
> 	计算步长 \alpha_k=\arg\min_{\alpha>0} f(x_k+\alpha p_k)；
> 	更新 x_{k+1}=x_k+\alpha_k p_k；
> 	if \|\nabla f(x_{k+1})\|<\varepsilon 
> 		stop;
> 	end if
> end for
> ```

> [!theorem] Theorem: 精确线搜索的收敛性
> 设 $\alpha_k$ 是精确线搜索的解，$\| \nabla^2 f(x_k+\alpha_k p_k)\le M$，则有：$$f(x_k)-f(x_k+\alpha_kp_k)\ge \frac{1}{2M} \|\nabla f(x_k)\|^2 \cos^2 \theta_k$$

证明：只需将 $f(x_k+\alpha_k p_k)$ 用 Taylor 展开到二阶项，然后利用 Hessian 的有界性即可。

这个定理表明，若搜索方向与负梯度方向夹角不大，则每次迭代都能获得足够的下降量，从而保证算法的收敛性。