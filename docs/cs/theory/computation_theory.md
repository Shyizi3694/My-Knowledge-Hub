# 计算理论


Computability / Undecidability    借助图灵机


研究计算的工具



检验工具的能力：形式语言



计算的难易（复杂性）


- Halting Problem
- SAT (NPC)
- P ?= NP



10 作业

20 quiz

70 final



## Chapter 1: Introduction to Computing


什么是计算/算法

如何衡量计算的复杂程度


Turing Machine

问题 -> 集合符号 -> 语言 -> 计算模型



语言：
- 正则语言
- 上下文无关语言
- 。。。。。。




### Section 1.1: Sets


Definition



Empty Set



Subset



Proper Subset 



Set Operations



Law



Power Set （不要求一定可数）


Patition （划分）
- 每一个非空
- 互不相交
- 并为全集



### Section 1.2: Relations & Functions


有序对


二元关系 $(a,b) \in R \Rightarrow R: A \to B$  



关系的操作
- Inverse
- Composition


Domain & Range




Functions

Cantor's Theorem
定义了集合的势 $card$

关于实数集 $\mathbb{R}$ 不可数性的证明



#### Closure 闭包

运算的封闭性

==为什么计算理论里面的闭包概念这么奇怪（相当于群）==

关系的封闭性

- 自反
- 对称
- 传递

关系 $R$ 的自反传递闭包 $R^*$

$R^*$ 是包含 $R$ 的最小的满足自反性、传递性的集合

这里的“最小”应该参考点集拓扑中的闭包定义，应该作为自反传递闭包的定义而不是性质（换言之，所有包含 $R$ 的所i有满足自反性、传递性的关系集合的交）。


### Section 1.3: Languages


#### Alphabet

- 任何有限的集合都可以叫做 Alphabet
- 其中元素称为 symbols （符号）


注意空集也是 Alphabet

用 $\Sigma$ 表示

e.g. binary alphabet

#### String 

alphabet 的有限序列 / strings over $\Sigma$

$\Sigma$ 上所有 string 的集合 $\Sigma^*$ (无限但可数)——（如何证明？？？？？

注意空 string 存在，故 $\emptyset$ 上的 $\Sigma^*$ 非空

#### Operations of String
- concatenation 拼接（这是一个群论中的二元运算）
- exponentiation 多个拼接的复合
- reversal （求反）

$$
w = ua \Rightarrow w^R = au^R
$$

#### Languages

string 的集合称为 language

$L \subseteq \Sigma^*$

注意：$\emptyset$、$\Sigma$、$\Sigma^*$ 都是 language

- 有限 language
- infinite language: 通过 ZFC 中的分离公理得出的 language
$$
L = \{w \in \Sigma^* : w \text{ has property } P\}
$$

如何证明：若 $\Sigma$ 有限，则 $\Sigma^*$ 必可数无穷?

任意 alphabet 上的 language 可数

operations of languages

- 交、并、差、补（都是以 $\Sigma^*$ 为全集）
- exponentiation
- concatenation（元素拼接，类似直积）
- Kleene Star
	- $L^* = \{w \in \Sigma^*: w = w_1w_2\cdots w_k, k \ge 0, w_1, w_2, \cdots, w_k \in L\} = L^0 \cup L^1 \cup \cdots$
- $L^+ = L^* \setminus L^0$

$L^+$ 是可以包含空 string 的，因为 $L$ 可能包含空 string

$\emptyset^* = {e}$，$e$ 为空 string

$L^* = LL^+$

$(L^*)^* =L^*$

$\emptyset L = L \emptyset$

#### Finite Representation of Languages

对于有限的 language，可以用有限的形式表达，我们希望能够有限地表示无限的 language

- finite representations

必须是 string

不同的语言有不同的表示

例子：可以用 cup, kleene star & concatenation 有限表示一个 language

给出 regular expression 的定义：一个包含原 alphabet 的新 alphabet ，加入了 cup，kleene star 和 concatenation 三个 symbol


$\emptyset$ & $\{x\}$ 是 regular expression，加上上述三种符号也是，除此之外都不是 regular expressions

为什么这样的 expression is not regular?

$$
L = \{a^nb^n | n \ge 1\}
$$




用 $\mathcal{R}$ 表示 regular expressions 的集合，其中的每一个 expression 都是一个语言的 representation


这样简单的结构能表达的语言是不多的，后续会扩展以支持更多语言的表达


$a \in \mathcal{R}$，则 $\mathcal{L}(a)$ 为 $a$ 表达的 language

严格的 definition: 将 $\mathcal{L}$ 定义为映射：

$$
\begin{aligned}
& \mathcal{L}: \mathcal{R} \to 2^{\Sigma^*} \\
& \mathcal{L}(\emptyset) = \emptyset, \mathcal{L}(a) = a \qquad \forall a \in \Sigma \\
& \mathcal{L} \text{ is closed for } \cup \text{ , } \circ \text{ and Kleene Start(}^*\text{)} \\
\end{aligned}
$$


## Chapter 2: Finite Automata

### Section 2.1: Deterministic Finite Automata

确定性有限自动机可以表示为一个五元组：$M= (K, \Sigma, \delta, s, F)$

- K: 有限的状态集合
- $\Sigma$ : 字母表（用于描述 Action）
- $\delta$: 状态转移函数
- $s \in K$: 初始状态
- $F \subseteq K$: 终结状态集合


$$
\delta: K \times \Sigma \to K
$$
$\delta(q, a) = p$: 当前状态 $q$，动作 $a$ 后状态变为 $p$





为了确保“有限”，所有的状态需要 pre-defined；若不然，实时计算下一个状态的过程中，可能会不断产生新的状态，则无法保证“有限”。

DFA 的缺陷是没有“记忆”，即没有用于存储的组件。


重点：transition function 决定了==唯一的==下一状态，这里“唯一”对应 deterministic



图表示（状态机的图表示）

图中初始状态的节点需要加上一个形如 “>” 的折线，终止状态的节点内部加上一个 “$\circ$”

给定一个 action sequence 和 transition function，可以画出状态转移图。




一个例子：构造一个接受 $\{a, b\}^*$ 的 DFM，使得序列不得包含连续三个 $b$ 
	———— 在第三个 $b$ 时进入一个“否定”的状态，且死循环。（只有这个否定的状态不得是终止状态）




DFM 的 configuration：$(q, w)$

这是一个二元组，其中 $q$ 为当前状态，$w$ 为尚未读取、且将在未来被读取的 action sequence


诱导（yield）下一个 configuration: $(q, w) \vdash_M (q', w')$

其中 $w = aw'$，$\delta(q,a)=q'$

最后一个 configuration: $w = \varepsilon$，表示空串。

$\vdash$ 作为一个二元关系，有其自反传递闭包 $\vdash^*$。

那么，一个字符串 $w$ （对应状态 $s$）可以被 DFM 接受，当且仅当 $\exists q \quad \text{s.t. }(s,w) \vdash_M^* (q, \varepsilon)$


### Section 2.2: Nondeterministic Finite Automata

在一些情况下，NFA 的效率比 DFA 的效率高得多


NFA 与 DFA 的区别是 transition relation 的不同：

$$
\Delta: K\times \{\Sigma \cup\{e\}\}\times K
$$

即：允许无 action 的状态转移；并且对相同的当前状态和 action，输出可能是多个==不同的状态（非确定）==

NFA 的 configuration 和 yield 都与 DFA 类似




NFA 和 DFA 的表达效率比较

命题：若 $\forall L,L = L(M \in NFA)$，$\exists M'\in DFA$ 使得 $L = L(M')$


定理：两台 FA 等价当且仅当 $\mathcal{L}(M_1)  =\mathcal{L}(M_2)$

这个等价中，DFA $\Longrightarrow$ NFA 是显然的。

定理：对任意 NFA，总存在 DFA与之等价


难点：NFA 中是 transition relation 而 DFA 中是 transition function；NFA 允许 empty action 而 DFA 不允许

解决难点 1：将 transition relation 写成：$\Delta: K \times \{\Sigma \cup\{e\}\}\to \mathcal{P}(K)$

在状态图中，如果出现一对多的情况，则加入一个子集节点，将操作结果指向这个子集节点，然后使用响应数量 empty action 转移到原先的几个节点。这里就完成了上面的改写在图上的构造，但是还是没有本质上解决难点 1（引入 empty action 后一对多还是存在与加入的节点和目标节点之间）。但这会在下面解决难点 2 时顺带解决。

解决难点 2：

定义一个 $E(q)$，定义为 状态 $q$ 能==仅通过 empty action== ==多步==跳转到的状态的集合



对于每一个 NFA:$M = (K, \Sigma, \Delta, s, F)$，构造 DFA：$M' = (K', \Sigma, \delta, s', F')$

- $K' = \mathcal{P}(K)$
- $s'=E(s)$
- $F' = \{Q\subseteq K | Q \cap F \ne \emptyset\}$
- $\delta(Q,a) = \bigcup\{E(q)|q\in Q,p \in K, (q,a,p)\in \Delta\}\quad \forall Q\subseteq K, \forall a\in \Sigma$


断言：$(q,w)\vdash_M^* (p,e) \Longleftrightarrow (E(q), w) \vdash_{M'}^*(P,e),\quad p\in P$

> [!faq] 
> 这里关于断言的证明和 NFA $\Longrightarrow$ DFA 的证明还是希望有板书，而不是干讲。


NFA 到 DFA 的流程：

1. 列出所有状态的 $E(q)$
2. 确定 $s'=E(s)$
3. 对于每一个 $q_i$，$\delta(\{q_i\},a)=\bigcup_{j\in K_\alpha} E(q_j)$，此处 $(q_i, a, q_j) \in \Delta$

### Section 2.3: Finite Automata and Regular Expressions

> [!theorem] 
> Language ${L}$ 是正则的 $\Longleftrightarrow$ 它被一个 FA 接收 $\quad \text{i.e. } L = \mathcal{L}(M)$

回顾：REx 在 Union, Concatenation 和 Kleene Star 下封闭。

故只需在 FA 下表示这三个运算，并证明其封闭即可。

- Union

加入一个初始状态，通过 empty step 非确定的跳转到 $M_1$ 和 $M_2$

> [!question] 任意并封闭吗？

- Concatenation

将 $M_1$ 的终止状态通过 empty step 跳转到 $M_2$ 的初始状态

- Kleene Star

首先，为了能接收空串，另加一个状态，既是初始状态也是终止状态；然后将其通过 empty step 跳转到原 $M_1$ 的初始以实现无差别地接收 $M_1$ 的字符串；最后，为了能循环接收字符串，将原 $M_1$ 的所有终止状态通过 empty step 跳转到原 $M_1$ 的初始状态。

> [!question] 为什么不能停在原 $M_1$ 的初始？

- 另外，关于补（Complementation）、交也封闭

这样，对于任何 REx，都可以将其递归地拆解为如上基本操作，然后逐步用 FA 表示。


-  Generalized Finite Automaton Construction

在初始状态前、终止状态后形式化地添加两个状态，用 empty step 实现单向的跳转。




### Section 2.4: Languages that are and are not regular

> [!theorem] Pumping Theorem (Necessary but not Sufficient Condition)
> $L$ 是一个正则语言，则存在正整数 $n \ge 1$ 使得任一字符串 $w \in L$，只要 $|w|>n$，$w$ 就可以写成 $w=xyz$，其中 $y\ne e$，$|xy|\le n$ 且对每一个 $i \ge 0， xy^iz\in L$。

一个有趣的使用方法：想象一个与之博弈的对手，对手坚称 $L$ 是正则的，而己方需要证明其非正则。对手需要给定一个 $n$，然后己方需要在 $L$ 中找到一个长度大于等于 $n$ 的字符串，接着对手将其恰当地分解为 $xyz$，最后己方指出一个 $i$，使得 $xy^iz$ 不在 $L$ 中。无论对手如何高明地拆解，己方总能找到一个 $i$.

## Chapter 3: Context-Free Languages


### Section 3.1: Context-Free Grammars


语言生成器：上下文无关语法


一种映射规则：对于字符串中的某子串，无论其上下文（左边和右边的内容）如何，都可以将其替换为某个不同的子串。

这样可以递归地生成非正则的字符串，如 $a^nb^n$

上下文无关文法 $G = (V, \Sigma, R, S)$，其中 $V$ 是字母表，$\Sigma \subseteq V$ 为终止字符集，$R\subseteq(V\setminus \Sigma)\times V^*$ 为规则集合，$S\subseteq(V\setminus \Sigma)$ 为起始字符

derive

### Section 3.2: Parse Tree

分析树的根是起始符，叶子是终止符或空串

derive 的相似

derive 的 preced （先于）

最左/最右 derivation


存在两个多个不同的 Parse Tree 的语法是 ambiguous （有歧义）的。


### Section 3.3: Push Down Automata

下推自动机，简称 PDA


正则语言对应 DFA，上下文无关语言对应 PDA


某种形式上引入 memory


下推自动机包括一个 input tape，一个 finite control 和一个 stack

定义 PDA：

$$
M = (K, \Sigma, \Gamma, \Delta, s, F)
$$

其中 $K$ 是有限状态集，$\Sigma$ 为字母表，$\Gamma$ 为栈符号集，$\Delta$ 是转移关系，$s=start$，$F=Final$

$$
\Delta \subseteq (K \times (\Sigma \cup \{e\}) \times \Gamma^*)\times(K\times\Gamma^*)
$$

> [!example]+ Example: 转移关系的例子 
> - $((p,u,\alpha),(q,\beta))$
> 
> 接收 $u$，状态从 $p$ 转移到 $q$，栈顶从 $\alpha$ 变为 $\beta$（另一种说法是栈本身从 $\alpha$ 变为 $\beta$，但从可解释性来说，更倾向于前者）
> 
> - $((p,u,a),(p,e))$
> 
> pop $a$


PDA 接受一个字符串：最终栈为空，状态位于终止状态

同样定义 PDA 的 Configuration：$(p, x, \alpha)$

诱导 $\vdash_M$ ：$(p, ux, \beta\alpha) \vdash_M (q, x, \gamma\alpha) \Longleftrightarrow ((p, u, \beta),(q,\gamma)) \in \Delta$

同样有其自反传递闭包 $\vdash_M^*$


PDA $M$ 接收 $w \Longleftrightarrow(s, w, e)\vdash_M^*(f,e,e)$



### Section 3.4: PDA and CFL

> [!theorem]
> 下推自动机接收的语言正好是上下文无关语言类。（充要）

==证明待补全（参考教材）==

