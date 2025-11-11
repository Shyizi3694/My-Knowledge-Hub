
# 大规模数据分析与机器学习

- 不止讲 ai，更关注 optimization & data process


- Algorithm tuning


## Chapter 1(Lecture): AI Technology Trend



Deep Learning 陷入经验主义

中间过程不具有可解释性

End to End Learning：忽略了中间的可解释性，只关注结果上的合理


DL focuses on data rather than features.

Deep learing 学习了层级的特征


深度网络具有语义抽象层次不断提升的感受野



Back Propagation: Loss-driven




## Chapter 1 (online): Game Theory

- Shooter-Goalie Game

Set of action: {L, R}

the shooter and the goalie take the action at the some

payoff matrix

zero-sum game: the sum of payoff for the two players is 0


- Pure & Mixed Strategies

Pure strategy: the player choose a deterministic action

Randomized Strategies: having a distribution over their choices

$$
\sum_{i\in \mathcal{A}} p_i = 1
$$

The distributions $p$ and $q$ are ==mixed strategies==

- expected payoff

$$
\begin{aligned}
V_R & = \sum_{i,j} Pr[A(i,j)] \cdot R_{i,j} \\
V_C & = \sum_{i,j} Pr[A(i,j)] \cdot C_{i,j} \\
\end{aligned}
$$

- Minimax Optimal Strategies

player maximizes his expected payoff over all strategies of his opppnent.

lower bound: $lb = \max_{p}\min_{q}V_R(p,q)$

lower bound 是因为对手采取了他的视角下的最优的策略，这时尽管己方也采取最优策略，也会比对手采取其他策略而己方采取最优策略要来的差。


- Claim: 
$$
\max_{q}\min_{p}V_C(p,q) = -\min_{q}\max_{p}V_R(p,q)
$$

证明是显然的


upper bound: 

$$
up = \min_{q}\max_{p}V_R(p,q)
$$

从对手视角的 lower bound，所以是己方的 upper bound


问题：lower bound 和 upper bound 之间差多少？


Claim: Can assume column player plays a pure strategy *(WHY????)*

需要让 $\sum_{i} p_i R_{i,j}$ 最小化，因此 $q$ 只有一种选择。


$$
lb = \max_{p}\min_{j}\sum_{i}p_i R_{i,j}
$$

$lb = up = 0$

value of the game



## Chapter ?(Lecture)

- Nerf: use multiview images to do 3D reconstruction

用离散的采样、求和逼近连续的积分

