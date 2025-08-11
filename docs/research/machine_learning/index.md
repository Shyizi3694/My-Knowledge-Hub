# 机器学习算法导论


**核心策略：**
- **合并任务**: 将理论上关联或实现上类似的模块合并在同一天或两天内完成。
- **并行处理**: 在等待代码调试或测试时，可以开始下一个模型的理论预习。
- **优先级排序**: 优先保证核心算法的实现和理解，对于某些次要的扩展（如多种剪枝算法）可以先实现一种。

---

### **三周高强度冲刺计划 (21-Day Sprint Plan)**

---

### **第一周：极速入门与核心算法上半场 (Week 1: Rapid Foundations & Core Algorithms Pt. 1)**

**目标：** 以最快速度搭建好环境并掌握线性模型、决策树之前的核心分类器。

- **第1天：项目初始化与线性回归**
    - **上午**: 完成所有工程化配置：Github仓库、项目结构、虚拟环境、安装所有库 (`black`, `ruff`等)。
    - **下午**: **【编码】** 实现 `BaseEstimator` 基类和 `metrics` 评估模块。
    - **晚上**: **【编码/报告】** 实现 `LinearRegression` 并完成Jupyter报告。**目标是在一天内完成从0到第一个完整模型的闭环。**
- **第2天：感知机与K近邻法 (KNN)**
    - **上午**: **【编码/报告】** 实现 `Perceptron`（原始+对偶形式）并完成报告。
    - **下午**: **【编码/报告】** 实现暴力搜索版的 `KNeighborsClassifier` 并完成报告。
    - **晚上**: **【理论】** 预习朴素贝叶斯和逻辑斯蒂回归的数学原理。
- **第3天：朴素贝叶斯与逻辑斯蒂回归**
    - **上午**: **【编码/报告】** 实现带拉普拉斯平滑的 `NaiveBayes` 并完成报告。
    - **下午**: **【编码/报告】** 实现 `LogisticRegression` 并完成报告。
    - **晚上**: **【理论】** （重点）深入预习决策树，包括ID3, C4.5, CART的理论。
- **第4-5天：决策树 (Decision Tree) 攻坚**
    - **第4天**: **【编码】** 实现决策树的核心框架，完成ID3和C4.5的划分逻辑。
    - **第5天**: **【编码】** 实现CART算法（分类+回归）及一个核心的剪枝方法。
    - **产出**: 一个功能相对完整的 `decision_tree.py` 模块。
- **第6-7天：决策树报告与SVM理论周**
    - **第6天**: **【报告】** 专注撰写决策树的Jupyter Notebook报告，要求高质量、图文并茂。
    - **第7天**: **【理论】** 本周的“理论日”。不写新代码，全力预习和理解支持向量机（SVM）的全部理论：最大间隔、拉格朗日对偶、KKT条件、核技巧以及SMO算法的推导。**这是下周攻克难关的必要准备。**

---

### **第二周：高级算法与集成学习之巅 (Week 2: Advanced Algorithms & Ensembles)**

**目标：** 啃下最硬的骨头（SVM），并掌握工业界最有用的集成学习模型。

- **第8-10天：决战支持向量机 (SVM) 与 SMO**
    - **第8天**: **【编码】** 实现核函数模块，并基于之前的理论知识，开始在 `SVM` 类中实现SMO算法的主体逻辑。
    - **第9天**: **【编码】** （核心）专注调试SMO算法，处理各种边界和收敛条件。
    - **第10天**: **【报告】** 完成SVM的训练和测试，撰写最终的Jupyter报告，可视化不同核函数下的分类边界。
- **第11-12天：集成学习双雄 (Ensemble Powerhouses)**
    - **第11天**: **【编码/报告】** 实现 `AdaBoost`（弱学习器直接用我们之前实现的决策树，限制其深度即可）并完成报告。
    - **第12天**: **【编码/报告】** 实现 `RandomForestClassifier` 并完成报告。
- **第13-14天：EM算法与无监督开篇**
    - **第13天**: **【编码/报告】** 实现高斯混合模型（GMM）的EM求解算法并完成报告。
    - **第14天**: **【编码/报告】** 实现 `KMeans` 和 层次聚类（或DBSCAN）。完成一份对比不同聚类算法的报告。**至此，核心内容已大部分完成。**

---

### **第三周：无监督专题与项目升华 (Week 3: Unsupervised Topics & Project Polish)**

**目标：** 快速扫清剩余的无监督主题，并对整个项目进行文档化和最终整理。

- **第15-16天：降维 (Dimensionality Reduction)**
    - **第15天**: **【编码/报告】** 实现 `PCA` 主成分分析并完成报告。
    - **第16天**: **【编码/报告】** 实现 `LDA` 线性判别分析，并与PCA进行对比分析。
- **第17-18天：SVD 与 MCMC**
    - **第17天**: **【应用/报告】** 学习并应用`numpy.linalg.svd`，完成一份SVD应用的报告（如图像压缩）。
    - **第18天**: **【编码/报告】** 实现Metropolis-Hastings或Gibbs抽样，并完成MCMC的报告。
- **第19-21天：项目收尾与升华**
    - **第19天**: **【代码审查】** 回顾你写过的所有代码，进行一次彻底的Code Review，检查是否符合`black`和`ruff`规范，接口是否统一，注释是否清晰。
    - **第20天**: **【文档撰写】** （极其重要）为你的Github仓库撰写一份顶级的`README.md`。内容应包括：项目简介、技术栈、功能列表（实现了哪些算法）、每个模块的简单介绍、如何运行的指南、以及你最满意的一两张结果图。
    - **第21天**: **【总结与复盘】** 撰写一份个人学习总结，归档所有Jupyter报告。回顾这三周的历程，整理你的心得和挑战。准备好如何向别人（面试官、老师）展示和介绍这个项目。

### **高强度计划的特别建议**

- **时间箱 (Time-boxing)**: 对于像SMO这样的难点，给自己设定一个明确的时间上限（比如3天）。如果届时仍未完美解决，先完成一个“能跑”的版本，做好标记，然后继续前进，不要让一个点卡住整个项目。
- **精力管理**: 高强度不等于不休息。保证充足的睡眠，每工作90分钟就起来走动15分钟。 burnout（耗竭）是高强度计划的最大敌人。
- **拥抱不完美**: 在如此紧凑的时间内，某些模型的实现可能无法做到100%的最优化。记住，完成比完美更重要。我们的核心目标是理解和实现算法的主干。

这个三周计划强度非常大，但回报也是巨大的。它将极大地锻炼你的学习能力、编码能力和抗压能力。祝你冲刺成功！

---

-  Git Commit 格式约定

又一个非常棒的专业工程问题！你已经开始思考如何像一个专业的软件工程师一样，不仅仅是“保存代码”，而是“**记录项目的演进历史**”。

清晰的Commit Message是协作开发和项目维护的生命线，即使对于个人项目，它也能在你几个月后回顾代码时，帮你快速记起当时做了什么、为什么这么做。

业界最流行、最受推崇的规范是 **Conventional Commits (约定式提交)**。它格式统一，既方便人阅读，也方便机器（如自动化工具）解析。

---

### **约定式提交的核心格式**

一个标准的 Commit Message 分为三部分：**标题 (Header)**、**正文 (Body)** 和 **页脚 (Footer)**。其中正文和页脚是可选的。

```
<type>(<scope>): <subject>
<-- 空一行 -->
[optional body]
<-- 空一行 -->
[optional footer]
```

#### **1. 标题 (Header)**

这是最重要的一行，由三部分组成：

- **`type` (类型)**: 必填，说明这次提交的性质。常用的类型有：
    
    - `feat`: **新功能** (feature)。你现在完成 `base`, `metrics` 和 `linear_regression` 就是这个类型。
        
    - `fix`: **修复Bug**。
        
    - `docs`: 只修改了**文档**（如README, 注释）。
        
    - `style`: 不影响代码逻辑的修改（如 `black` 格式化、修正拼写错误）。
        
    - `refactor`: **代码重构**，既不是新功能也不是bug修复。
        
    - `test`: 添加或修改**测试**。
        
    - `chore`: **杂务**，修改构建流程、配置文件、依赖等（如修改 `requirements.txt`）。
        
- **`scope` (范围)**: 可选，用括号括起来，表示这次修改影响的范围。对于我们的项目，可以是：
    
    - `base`: 修改了 `base.py`。
        
    - `metrics`: 修改了评估模块。
        
    - `models`: 修改了模型。
        
    - `optim`: 修改了优化器。
        
    - `project`: 项目级别的修改，如更新 `.gitignore`。
        
- **`subject` (主题)**: 必填，对修改的简短描述。
    
    - **使用动词开头的祈使句**，例如 "Add", "Fix", "Refactor"，而不是 "Added", "Fixes"。就像在下达一个命令。
        
    - 首字母通常小写。
        
    - 结尾不加句号。
        

#### **2. 正文 (Body)** - 可选

如果标题无法说清楚，可以在空一行之后添加更详细的描述。说明修改的**动机 (why)** 和 **主要改动 (what)**。

#### **3. 页脚 (Footer)** - 可选

用于标记“重大变更”（BREAKING CHANGE）或关闭某个Issue（如 `Closes #123`）。

---

### **针对你当前情况的实例**

你现在完成了 `base`, `metric`, 和 `linear_regression`，这是一个很大的功能块。你有两种提交策略：

#### **方案A：一次性作为一个完整功能提交**

如果你觉得这些内容共同构成了一个完整的“初始核心功能”，可以合并成一次提交。

**Commit Message 示例:**

```
feat(core): implement foundational modules and linear regression

- Add BaseEstimator, clone, and Mixins to `base.py` to establish a scikit-learn compatible API.
- Implement core evaluation metrics (MSE, RMSE, R2, accuracy) in the `metrics` module for model assessment.
- Implement the first model, LinearRegression, using a reusable Gradient Descent optimizer architecture.
- This commit establishes the core OOP structure and API for all future models.
```

- **优点**: 将一个大的、逻辑上完整的功能块捆绑在一起。
    
- **缺点**: 如果未来发现其中一小部分有问题，回溯起来不如多次提交清晰。
    

#### **方案B：分多次进行“原子性”提交 (更推荐)**

这是更专业的做法。每一次提交都只包含一个独立的、逻辑完整的改动。

**提交顺序和 Message 示例:**

1. **第一次提交：只提交 `base.py` 的内容**
    
    Bash
    
    ```
    git add src/mlfromscratch/base.py
    git commit -m "feat(base): implement BaseEstimator, clone, and Mixins"
    ```
    
2. **第二次提交：只提交 `metrics` 模块**
    
    Bash
    
    ```
    git add src/mlfromscratch/metrics/
    git commit -m "feat(metrics): add regression and classification evaluation functions"
    ```
    
3. **第三次提交：提交优化器和线性回归模型**
    
    Bash
    
    ```
    git add src/mlfromscratch/optim/ src/mlfromscratch/models/linear_regression.py
    git commit -m "feat(models): add LinearRegression with separable GD optimizer"
    ```
    

- **优点**:
    
    - **历史清晰**: Git历史记录非常干净，每一步都清晰可追溯。
        
    - **易于审查 (Code Review)**: 每次改动都很小，方便他人（或你自己）审查。
        
    - **安全**: 如果发现某次提交引入了bug，可以轻松地使用 `git revert` 来撤销它，而不会影响其他功能。
        

### **我的建议**

我个人**强烈推荐方案B（多次提交）**，它能帮助你从一开始就养成“原子性提交”的专业习惯。

不过，考虑到这是项目的第一次大规模代码提交，使用**方案A**将所有初始化工作捆绑在一起，也是完全可以接受的。

你可以自行选择，但从长远来看，努力让每一次 `git commit` 都尽可能小而专注，是一个非常值得培养的习惯。