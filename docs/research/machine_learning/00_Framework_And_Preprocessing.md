# Chapter 0: 项目准备、框架与预处理

模仿 Scikit-Learn 的项目结构，从零开始完成一个机器学习代码库不是一个简单的任务。在对模型的创建、训练和评估之前，我们需要先完成一些准备工作。这个章节将介绍如何设置项目结构、创建数据集类以及实现数据预处理的基本功能。

## Section 0.1: 环境准备

Python 是一个强大的编程语言，拥有丰富的库和工具，适合用于机器学习和数据科学，使用它作为本次项目的语言是很自然的。选择 Python 3.12.3 WSL 作为开发环境，能够充分利用其最新的特性和性能改进。

使用 PyCharm 作为 IDE，在创建工程时能够自动创建虚拟环境和一些必要的配置文件如：

- `.gitignore`：用于指定哪些文件或目录不应该被 Git 跟踪。
- `requirements.txt`：列出项目所需的 Python 包及其版本，便于环境的重现。
- `README.md`：提供项目的基本信息和使用说明。
- `LICENSE`：指定项目的许可证类型，确保代码的使用和分发符合相关法规。
- `pyproject.toml`：用于定义项目的元数据和依赖关系，符合 PEP 518 标准。

以下是 `requirements.txt` 的内容，列出了除了 IDE 自带的库以外，项目仍需的 Python 包及其版本：
```plaintext
# 核心计算与数据科学库
numpy
scikit-learn

# Jupyter 报告与可视化
jupyterlab
matplotlib
seaborn

# 代码质量工具
black
ruff
```

- 在环境准备中遇到的问题：IDE 无法自动创建虚拟环境并激活，需要手动创建和激活虚拟环境。

----


## Section 0.2: 项目结构

以下是项目的目录结构，仿照 Scikit-Learn 的项目结构设计，包含了核心代码、测试代码和文档等部分：
```plaintext
.
├── LICENSE
├── README.md
├── build_docs.sh
├── main.py
├── notebooks
│   ├── 00_Framework_And_Preprocessing.ipynb
│   ├── ...
│   └── notebook_demo.ipynb
├── pyproject.toml
├── requirements.txt
└── src
    └── mlfromscratch
        ├── __init__.py
        ├── compose
        │   ├── __init__.py
        │   ├── _column_transformer.py
        │   └── _pipeline.py
        ├── metrics
        │   ├── __init__.py
        │   ├── classification.py
        │   └── regression.py
        ├── models
        │   ├── __init__.py
        │   ├── linear_regression.py
        │   └── ...
        ├── optim
        │   ├── __init__.py
        │   ├── gradient_descent.py
        │   └── ...
        ├── preprocessing
        │   ├── __init__.py
        │   ├── encoder.py
        │   ├── imputer.py
        │   └── scaler.py
        └── utils
            ├── __init__.py
            ├── base.py
            ├── data_loader.py
            └── validation.py
```

- `src/mlfromscratch/compose`：组合逻辑，包含管道和列转换器的实现。
- `src/mlfromscratch/metrics`：评估指标的实现，包括分类和回归指标。
- `src/mlfromscratch/models`：模型的实现。
- `src/mlfromscratch/optim`：优化算法的实现。
- `src/mlfromscratch/preprocessing`：数据预处理的实现，包括编码器、插补器和缩放器。
- `src/mlfromscratch/utils`：工具类和函数的实现，包括基类、数据加载器和验证器。

----


## Section 0.3: 模型基类的设计

Scikit-Learn 的诸多模型可以抽象出一些通用的接口和方法，这些方法可以在一个基类中实现，以便于其他模型继承和复用。本项目在 `src/mlfromscratch/utils/base.py` 中定义了一个 `BaseEstimator` 类，作为所有模型的基类。这个基类包含了以下方法：

- `@abstractmethod fit(X, y, **fit_params) -> self`：训练模型，其中 `y` 和 `**fit_params` 是可选参数。
- `@abstractmethod predict(X) -> np.ndarray`：使用训练好的模型进行预测。
- `fit_predict(X, y=None, **fit_params) -> np.ndarray`：先训练模型再进行预测，返回预测结果。
- `get_params(deep=True) -> dict` & `set_params(**params) -> self`：模型参数的 getter 和 setter。
- `__repr__()`：返回模型的字符串表示，便于调试和日志记录。

其他方法是上述几个方法的辅助方法。其中 `fit` 和 `predict` 是**抽象方法**，要求派生类必须实现。

----


## Section 0.4: 评估指标函数的设计

分类和回归模型的评估指标有着不同的计算方式，这是分类和回归本身的性质决定的：分类模型根据训练得出的参数，对输入数据进行分类预测，输出类别标签；而回归模型则输出连续值。本项目的评估指标函数设计遵循 Scikit-Learn 的风格，位于 `src/mlfromscratch/metrics` 目录下，包含`classification.py` 和 `regression.py` 两个模块。

### 分类评估指标

前面提到，分类模型输出类别标签的数组，在评估时需要将其与真实标签进行比较。对于分类问题，关注分类是否正确（预测是否严格与实际相等）的意义远远大于计算预测结果与实际的距离。

以二分类问题为例，预测的结果有以下四种情况：

- True Positive (TP)：预测为正类，实际也是正类。
- True Negative (TN)：预测为负类，实际也是负类。
- False Positive (FP)：预测为正类，实际是负类（也称为假阳性）。
- False Negative (FN)：预测为负类，实际是正类（也称为假阴性）。

根据这四个量可以计算出常用的三个评估指标：

- Accuracy（准确率）：正确预测的比例。
$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

- Precision（精确率）：预测为正类的样本中，实际为正类的比例。
$$
\text{Precision} = \frac{TP}{TP + FP}
$$

- Recall（召回率）：实际为正类的样本中，预测为正类的比例。
$$
\text{Recall} = \frac{TP}{TP + FN}
$$

由 Precision 和 Recall 可以进一步计算 F1 Score（F1 分数），它是 Precision 和 Recall 的调和平均数，综合考虑了两者的平衡性：
$$
\text{F1 Score} = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

对于 `classification.py`，实现了以下评估指标函数：

- `accuracy_score(y_true, y_pred) -> float`计算分类准确率。
- `precision_score(y_true, y_pred) -> float`：计算精确率。
- `recall_score(y_true, y_pred) -> float`：计算召回率。
- `f1_score(y_true, y_pred) -> float`：计算 F1 分数。

以及其他辅助函数：

- `_calculate_confusion_matrix_values(y_true, y_pred) -> tuple`：计算混淆矩阵的各个值，返回展平后的数组。
- `_validate_inputs(y_true, y_pred) -> tuple`：验证输入的标签和预测结果是否有效。

### 回归评估指标

对于回归问题，预测的结果是连续值，因此评估指标的计算方式与分类问题有所不同，通常用预测结果与实际值之间的距离来衡量模型的性能。常用的回归评估指标包括：
- Mean Squared Error (MSE)：均方误差，计算预测值与实际值之间的平方差的平均值。
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

其中 $y_i$ 是实际值，$\hat{y}_i$ 是预测值，$n$ 是样本数量。

- Root Mean Squared Error (RMSE)：均方根误差，是 MSE 的平方根，具有与原始数据相同的单位。
$$
\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}
$$

进一步，可以用 R-squared（$R^2$）来衡量模型的拟合优度，它表示模型解释的方差占总方差的比例：
$$
R^2 = 1 - \frac{\text{MSE}}{\text{Var}(y)}
$$
其中 $\text{Var}(y)$ 是实际值的方差。

对于 `regression.py`，实现了以下评估指标函数：

- `mean_squared_error(y_true, y_pred) -> float`：计算均方误差。
- `root_mean_squared_error(y_true, y_pred) -> float`：计算均方根误差。
- `r_squared(y_true, y_pred) -> float`：计算 $R^2$ 值。

### 使用 Mixin 将评估指标添加到模型类

分类问题和回归问题通常各自具有默认的评估指标，如分类问题的默认评估指标是准确率（Accuracy），回归问题的默认评估指标是 $R^2$，因此在模型类中可以通过 Mixin 的方式将这些评估指标方法添加到模型类中。这样，模型类就可以直接调用这些评估指标方法进行评估，而不需要额外的导入或实现。

由此设计出 `src/mlfromscratch/utils/base.py` 中的 `ClassifierMixin` 和 `RegressorMixin` 类，分别包含了分类和回归的评估指标方法，并统一命名为 `score(self, X: np.ndarray, y: np.ndarray) -> float`，一体化训练、预测和评估。这些 Mixin 类可以被模型类继承，从而获得直接获得评估结果的方法。

然而笔者在实际使用过程中发现，为了避免一些不必要的警告信息影响整体美观，需要让 Mixin 类继承 `BaseEstimator` 类，并且如果需要要求派生类必须实现一些方法（如转换器类 Transformer 要求其派生类实现 transform() 方法），应当将对应 Mixin 类设计为抽象类（ABC）。即以如下形式出现：


```python
from abc import ABC, abstractmethod

import numpy as np

from src.mlfromscratch.utils.base import BaseEstimator


class MyMixin(BaseEstimator, ABC):
    @abstractmethod
    def ABCMethod(self):
        """抽象方法，派生类必须实现"""
        pass
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """计算模型的评估指标"""
        pass
```

在设计模型时，由于 Mixin 已经继承了 `BaseEstimator`，因此模型类只需要继承 Mixin 类即可获得评估指标方法，但是为了保持代码的一致性和可读性，笔者仍然建议模型类同时继承 `BaseEstimator` 类。这样可以确保模型类具有统一的接口和方法，同时也能避免一些潜在的错误。

需要注意，模型在同时继承 `BaseEstimator` 和 Mixin 类时，必须将 Mixin 类放在继承列表的前面，否则会导致 `TypeError: Cannot create a consistent method resolution` 错误。这是因为 Python 的方法解析顺序（MRO）决定了在调用方法时，首先查找的是 Mixin 类中的方法，然后才是 `BaseEstimator` 类中的方法。如果顺序颠倒，可能会导致无法找到某些方法，从而引发错误。

事实上，这样的写法与 Mixin 一开始的设计理念有所出入，因为 Mixin 的目的是为了提供额外的功能，而不是作为一个完整的类来使用。因此这种为了避免警告而将 Mixin 设计为抽象类的做法，仍然有待商榷。


```python
# 这是一个使用 Mixin 的示例，假设我们需要设计一个分类模型类和一个回归模型类。
from src.mlfromscratch.utils.base import BaseEstimator
from src.mlfromscratch.utils.base import ClassifierMixin, RegressorMixin

# 同时继承 BaseEstimator 和 ClassifierMixin 的模型类
class MyClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, **fit_params):
        # 模型训练逻辑
        pass

    def predict(self, X):
        # 模型预测逻辑
        return np.argmax(X, axis=1) # 示例返回值，假设 X 是一个二维数组，每行是一个样本的特征向量

# 同时继承 BaseEstimator 和 RegressorMixin 的模型类
class MyRegressor(RegressorMixin, BaseEstimator):
    def fit(self, X, y, **fit_params):
        # 模型训练逻辑
        pass

    def predict(self, X):
        # 模型预测逻辑
        return np.mean(X, axis=1) # 示例返回值，假设 X 是一个二维数组，每行是一个样本的特征向量

```

我们来测试一下这些模型类的评估指标方法是否正常工作：


```python
import numpy as np
# 创建一个分类模型实例
classifier = MyClassifier()
# 训练模型
classifier.fit(np.array([[0, 1], [1, 0], [1, 1], [0, 0]]), np.array([0, 1, 1, 0]))
# 预测结果
y_pred = classifier.predict(np.array([[0, 1], [1, 0],[1, 1], [0, 0]]))
# 计算评估指标
accuracy = classifier.score(np.array([[0, 1], [1, 0],[1, 1], [0, 0]]), np.array([0, 1, 1, 0]))
print(f"分类模型的准确率: {accuracy}")

# 创建一个回归模型实例
regressor = MyRegressor()
# 训练模型
regressor.fit(np.array([[0, 1], [1, 0], [1, 1], [0, 0]]), np.array([0.5, 0.5, 1.0, 0.0]))
# 预测结果
y_pred = regressor.predict(np.array([[0, 1], [1, 0],[1, 1], [0, 0]]))
# 计算评估指标
r2 = regressor.score(np.array([[0, 1], [1, 0],[1, 1], [0, 0]]), np.array([0.5, 0.5, 1.0, 0.0]))
print(f"回归模型的 R^2 值: {r2}")
```

    分类模型的准确率: 0.25
    回归模型的 R^2 值: 1.0


----

## Section 0.5: 数据集导入函数

模型的设计和测试是密不可分的，模型的设计完成后，需要使用合适的数据集进行训练和测试。注意这里我们并不是为了获得一个很好的效果，这取决于模型本身和参数的选择。我们只需要利用数据集来检验模型的逻辑功能是否正常，是否能够正确地进行训练和预测。

所幸，Scikit-Learn 提供了许多常用的数据集，可以直接导入使用。为了方便起见，我们在 `src/mlfromscratch/utils/data_loader.py` 中实现了一个数据集导入函数 `load_dataset(name: str, **kwargs) -> tuple`，可以根据数据集的名称加载相应的��据集，并返回特征矩阵和标签向量。

`data_loader.py` 中实现了以下数据集的自动导入功能：

- `load_regression_data(n_samples, n_features, random_state)`：生成随机回归数据集。
- `load_classification_data(n_samples, n_features, random_state)`：生成随机分类数据集。
- `load_moons_data(n_samples, noise, random_state)`：生成双月形分类数据集。
- `load_circles_data(n_samples, noise, random_state)`：生成同心圆分类数据集。
- `load_blobs_data(n_samples, noise, random_state)`：生成聚类数据集。

以下是一些真实数据集的加载函数：

- `load_diabetes_data()`：加载糖尿病数据集。
- `load_iris_data()`：加载鸢尾花数据集。
- `load_wine_data()`：加载葡萄酒数据集。
- `load_breast_cancer_data()`：加载乳腺癌数据集。

另外还具备一个返回可用数据集名称的函数 `get_available_datasets() -> list`，便于用户查询。

----


## Section 0.6: 数据预处理功能

通过数据加载功能获取的数据集通常需要进行一些预处理操作，以便于模型的训练和测试。Scikit-Learn 提供了许多常用的数据预处理方法，如特征缩放、编码、插补等。本项目在 `src/mlfromscratch/preprocessing` 目录下实现了这些常用的预处理功能。

所有的预处理器除了是一个基估计器外，都必须是一个转换器（Transformer），即除了 `fit()` 方法外，还必须实现 `transform(X: np.ndarray) -> np.ndarray` 方法。这个方法接受一个特征矩阵 `X`，并返回经过预处理后的特征矩阵。

转换器的设计也采用了 Mixin 的方式，将预处理器的通用接口和方法抽象出来，便于其他转换器继承和复用。所有的预处理器都继承自 `BaseEstimator` 类，并额外将 `transform()` 方法设置为抽象方法，要求派生类必须实现，并且直接提供 `fit_transform()` 方法，便于一体化操作。所有的转换器都继承自 `TransformerMixin` 类。

### 插补器

在将原始的 `.csv` 文件转为 Pandas DataFrame 时，可能会遇到缺失值（NaN）。为了处理这些缺失值，我们可以使用插补器（Imputer）来填充缺失值。Scikit-Learn 提供了 `SimpleImputer` 类，可以根据指定的策略（如均值、中位数、众数等）来填充缺失值。

`SimpleImputer` 类的实现位于 `src/mlfromscratch/preprocessing/imputer.py` 中，私有属性如下：

- `strategy`：插补策略，默认为 `'mean'`，可选值包括 `'mean'`、`'median'` 和`'most_frequent'`。
- `statistics_`：存储每列的插补统计量，如均值、中位数或众数。

方法如下：

- `__init__(self, strategy: str = 'mean')`：初始化插补器，指定插补策略。
- `fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> self`：计算插补所需的统计量（如均值、中位数等）。
- `transform(self, X: np.ndarray) -> np.ndarray`：根据计算的统计量填充缺失值。

另外，当策略为众数时，`fit()` 调用辅助函数 `_calculate_most_frequent(X: np.ndarray) -> np.ndarray` 来计算每列的众数。

其他插补器如 `KNNImputer` 和 `IterativeImputer` 也可以在 `src/mlfromscratch/preprocessing/imputer.py` 中实现，前者使用 KNN 算法进行插补，后者使用迭代算法进行插补。

### 编码器

并非所有的属性的值都能使用数值方法处理，分类属性的值通常是字符串或类别标签，这些值需要转换为数值才能用于模型训练，如“猫”“狗”“鸟”等。我们需要对这些非数值的属性值进行合适的编码，以便于模型的训练和预测。

独热编码是一种常用的编码方法，它将每个类别值转换为一个独立的二进制特征。如一个动物类型属性，包含“猫”“狗”“鸟”三个类别，特征矩阵示例如下：

| 样本序号 | 动物类别 |
|------|------|
| 1    | 猫    |
| 2    | 狗    |
| 3    | 鸟    |
| 4    | 猫    |
| 5    | 狗    |
| 6    | 鸟    |

独热编码后的特征矩阵如下：

| 样本序号 | 动物类别（猫） | 动物类别（狗） | 动物类别（鸟） |
|------|---------|---------|---------|
| 1    | 1       | 0       | 0       |
| 2    | 0       | 1       | 0       |
| 3    | 0       | 0       | 1       |
| 4    | 1       | 0       | 0       |
| 5    | 0       | 1       | 0       |
| 6    | 0       | 0       | 1       |

表中的 1 表示该样本属于该类别，0 表示不属于该类别。这样，每个类别值都被转换为一个独立的二进制特征，便于模型的训练和预测。

`OneHotEncoder` 类的实现位于 `src/mlfromscratch/preprocessing/encoder.py` 中，私有属性如下：

- `categories_`：存储每列的唯一类别值。

方法如下：

- `__init__(self)`： 初始化独热编码器。
- `fit(self, X: np.ndarray, _y: np.ndarray = None, **_fit_params) -> self`：计算每列的唯一类别值，存储在 `self.categories_` 中。
- `transform(self, X: np.ndarray) -> np.ndarray`：将类别值转换为独热编码形式。
- `inverse_transform(self, X: np.ndarray) -> np.ndarray`：将独热编码形式转换回原始类别值。

需要注意的是，这里 `OneHotEncoder` 并没有检验输入的待转换特征是否是非数值属性，也没有对输入是否为 NaN 进行前置检验，而是在过程中抛出错误。这是为了让 `OneHotEncoder` 专注于编码的任务，将数据的清洗与分类交给上一层模块来调度。

其他编码器如 `OrdinalEncoder` 和 `LabelEncoder` 也可以在 `src/mlfromscratch/preprocessing/encoder.py` 中实现，前者将类别值转换为整数，后者将类别标签转换为整数。

### 缩放器

缩放器的功能是将特征矩阵中的数值特征进行缩放，使其具有相同的尺度。常用的缩放方法包括标准化（Standardization）和归一化（Normalization）。标准化将特征值转换为均值为 0、方差为 1 的分布，而归一化将特征值缩放到指定的范围内（如 [0, 1]）。

对于某一列特征，通过如下方式计算其均值和标准差：
$$
\text{mean} = \frac{1}{n} \sum_{i=1}^n x_i
$$
$$
\text{std} = \sqrt{\frac{1}{n} \sum_{i=1}^n (x_i - \text{mean})^2}
$$

其中 $x_i$ 是该列特征的第 $i$ 个值，$n$ 是样本数量。标准化后的特征值计算公式为：
$$
\text{standardized\_value} = \frac{x_i - \text{mean}}{\text{std}}
$$

`StandardScaler` 类的实现位于 `src/mlfromscratch/preprocessing/scaler.py` 中，私有属性如下：

- `mean_`：存储每列特征的均值。
- `scale_`：存储每列特征的标准差。

方法如下：

- `__init__(self)`：初始化标准化缩放器。
- `fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> self`：计算每列特征的均值和标准差，存储在 `self.mean_` 和 `self.scale_` 中。
- `transform(self, X: np.ndarray) -> np.ndarray`：将特征值转换为标准化形式。
- `inverse_transform(self, X: np.ndarray) -> np.ndarray`：将标准化形式转换回原始特征值。

property 如下：

- `n_features_(self) -> int`：返回特征矩阵的列数，即特征的数量。

其他缩放器如 `MinMaxScaler` 和 `RobustScaler` 也可以在 `src/mlfromscratch/preprocessing/scaler.py` 中实现，前者将特征值缩放到指定的范围内（如 [0, 1]），后者使用中位数和四分位数进行缩放，适用于存在异常值的情况。

----


## Section 0.7: 更专业的工作流：组合

对于机器学习工作流，通常需要将多个预处理步骤组合在一起，以便于模型的训练和预测。设想一个特征矩阵，可以串行地对其进行插补、编码和缩放等预处理操作，也可以并行地对其中的不同特征列进行不同的预处理操作。从这个角度出发，可以抽象出组合器（Composer）的概念，将多个预处理步骤组合在一起，形成一个完整的工作流。

前述“串行”和“并行”的两种组合方式分别由 `Pipeline` 和 `ColumnTransformer` 类来实现。`Pipeline` 类用于将多个预处理步骤串行地组合在一起，而 `ColumnTransformer` 类用于对特征矩阵的不同列进行不同的预处理操作。

### Pipeline

为了实现所谓“串行”的组合，`Pipeline` 类需要将多个预处理步骤按顺序组合在一起，即需要传入一系列转换器，以及用于最后一个步骤的预测模型。同样地，`Pipeline` 类也需要继承 `BaseEstimator` 类，并实现 `fit()`、`predict()` 方法，来统一地处理多个预处理步骤和模型的训练和预测。

`Pipeline` 类的实现位于 `src/mlfromscratch/compose/_pipeline.py` 中，私有属性如下：

- `steps_`：存储预处理步骤和模型的列表，每个元素是一个元组，包含步骤名称和对应的转换器或模型。

方法如下：

- `__init__(self, steps: list)`：初始化管道，传入预处理步骤和模型的列表。
- `fit(self, X: np.ndarray, y: np.ndarray = None, **fit_params) -> self`：按顺序调用每个预处理步骤的 `fit()` 方法，并将结果传递给下一个步骤，最后调用模型的 `fit()` 方法。
- `predict(self, X: np.ndarray) -> np.ndarray`：按顺序调用每个预处理步骤的 `transform()` 方法，并将结果传递给模型的 `predict()` 方法，返回最终的预测结果。

重载的方法：

- `get_params(self, deep=True) -> dict`：返回管道中所有步骤的参数，便于参数的获取和设置。
- `set_params(self, **params) -> self`：设置管道中所有步骤

property 如下：

- `named_steps(self) -> dict`：返回一个字典，包含每个步骤的名称和对应的转换器或模型。
- `final_estimator(self) -> BaseEstimator`：返回管道中的最后一个模型，即最后一个步骤的转换器或模型。

一个 class-method:

- `from_estimators(cls, estimators: list) -> 'Pipeline'`：从一系列不带步骤名的估计器（转换器或模型）创建一个管道实例，便于快速构建管道。

直观上来看，`fit()` 方法需要对前面所有的转换器调用其 `fit_transform()` 方法，然后单独地对最后一个预测模型调用 `fit()` 方法。`predict()` 方法则需要对前面所有的转换器调用其 `transform()` 方法，然后单独地对最后一个预测模型调用 `predict()` 方法。

然而，`Pipeline` 类有可能是一个完全的转换器，即最后一个步骤是一个转换器而不是预测模型。在这种情况下，`Pipeline` 类仍然可以正常工作，只是最后一步的 `predict()` 方法会被替换为 `transform()` 方法。因此，`transform()` 方法也是需要实现的，但为了防止错误地为预测器类的 `Pipeline` 实例调用 `transform()` 方法，或者反之，`predict()` 和 `transform()` 方法的开头都对最后一个步骤的类型进行了检查，确保其是一个转换器或预测模型。

以下是一个使用 `Pipeline` 类的示例，假设我们需要对特征矩阵进行插补、编码和缩放等预处理操作，然后使用一个分类模型进行训练和预测：


```python
from src.mlfromscratch.compose import Pipeline
from src.mlfromscratch.preprocessing.scaler import StandardScaler
from src.mlfromscratch.preprocessing.imputer import SimpleImputer
from src.mlfromscratch.metrics.classification import accuracy_score
import numpy as np

X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
y_train = np.array([0, 1, 1, 0])
X_test = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
y_test = np.array([0, 1, 1, 0])

classifier = MyClassifier()  # 假设 MyClassifier 是一个分类模型类
pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # 插补缺失值
    ('scaler', StandardScaler()),  # 标准化缩放
    ('classifier', classifier)  # 分类模型
])
# 训练管道
pipeline.fit(X_train, y_train)  # X_train 是特征矩阵，y_train 是标签向量
# 预测结果
y_pred = pipeline.predict(X_test)  # X_test 是测试特征矩阵
# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)  # y_test 是测试标签向量
print(f"管道的准确率: {accuracy}")
```

    管道的准确率: 0.25


### ColumnTransformer

有了 `Pipeline` 类的基础，我们可以进一步实现 `ColumnTransformer` 类，用于对特征矩阵的不同列进行不同的预处理操作。`ColumnTransformer` 类允许我们为每一列指定一个转换器，并将这些转换器组合在一起，形成一个完整的工作流。

直观上看，`ColumnTransformer` 类需要将多个转换器按列组合在一起，即需要传入一系列转换器和对应的列索引或列名称。与 `Pipeline` 不同的是，`ColumnTransformer` 只能是一个转换器，而不能是一个预测模型。因此，`ColumnTransformer` 需要继承 `TransformerMixin` 类，并实现 `fit()`、`transform()` 方法，来统一地处理多个转换器的训练和转换。

另外，`ColumnTransformer` 是完全并行的，即对于每一个列，最多只能对应一个转换器，如果想要对某一列进行多个转换器的处理，可以将这些转换器组合成一个管道，然后再将这个管道作为一个转换器传入 `ColumnTransformer`。这样可以确保每一列只对应一个转换器，避免了多重转换器对同一列的冲突。

`ColumnTransformer` 类的实现位于 `src/mlfromscratch/compose/_column_transformer.py` 中，私有属性如下：

- `transformers`：存储转换器和对应的列索引或列
- `remainder`：指定未被转换器处理的列的处理方式，默认为 `'drop'`，即丢弃未被处理的列。
- `fitted_transformers_`：存储每个转换器的拟合结果，便于后续的转换。

方法如下：

- `__init__(self, transformers: list, remainder: str = 'drop')`：初始化列转换器，传入转换器和对应的列索引或列名称，以及未被处理的列的处理方式。
- `fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params) -> self`：按列调用每个转换器的 `fit()` 方法，并将结果存储在 `self.fitted_transformers_` 中。
- `transform(self, X: np.ndarray) -> np.ndarray`：按列调用每个转换器的 `transform()` 方法，并将结果组合在一起，返回最终的特征矩阵。

以下是一个使用 `ColumnTransformer` 类的示例，假设我们需要对特征矩阵的不同列进行不同的预处理操作，如对数值列进行标准化，对类别列进行独热编码：


```python
# 使用 ColumnTransformer 进行混合数据类型的预处理
from src.mlfromscratch.compose import ColumnTransformer
from src.mlfromscratch.preprocessing.scaler import StandardScaler
from src.mlfromscratch.preprocessing.encoder import OneHotEncoder
import numpy as np

# 创建混合类型的示例数据
# 前两列是数值特征，后两列是类别特征
X_mixed = np.array([
    [1.0, 2.5, 'A', 'X'],
    [2.0, 3.1, 'B', 'Y'],
    [1.5, 2.8, 'A', 'Z'],
    [3.0, 4.0, 'C', 'X'],
    [2.5, 3.5, 'B', 'Y'],
    [1.8, 2.2, 'A', 'Z']
])

print("原始数据:")
print(X_mixed)
print(f"数据形状: {X_mixed.shape}")

# 创建列转换器
# 对列 0,1 进行标准化，对列 2,3 进行独热编码
column_transformer = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), [0, 1]),        # 数值列标准化
        ('encoder', OneHotEncoder(), [2, 3])         # 类别列独热编码
    ],
    remainder='drop'  # 未指定的列将被丢弃
)

# 训练列转换器
column_transformer.fit(X_mixed)
print("\n列转换器训练完成")

# 检查拟合的列转换器
print("\n拟合的转换器信息:")
for name, transformer, columns in column_transformer.fitted_transformers_:
    print(f"  {name}: {type(transformer).__name__} 应用于列 {columns}")
    if hasattr(transformer, 'mean_'):
        print(f"    均值: {transformer.mean_}")
        print(f"    标准差: {transformer.scale_}")
    if hasattr(transformer, 'categories_'):
        print(f"    类别: {transformer.categories_}")

# 转换数据
X_transformed = column_transformer.transform(X_mixed)

print(f"\n转换后的数据:")
print(X_transformed)
print(f"转换后的数据形状: {X_transformed.shape}")

# 验证转换结果
print("\n验证转换结果:")
print("前两列应该是标准化后的数值（均值≈0，标准差≈1）")
print(f"前两列的均值: {np.mean(X_transformed[:, :2], axis=0)}")
print(f"前两列的标准差: {np.std(X_transformed[:, :2], axis=0)}")

print("\n后面的列应该是独热编码的结果（只包含0和1）")
print(f"后面列的唯一值: {np.unique(X_transformed[:, 2:])}")

# 测试 remainder='passthrough' 的情况
print("\n" + "="*50)
print("测试 remainder='passthrough' 的情况")

# 创建一个包含更多列的数据
X_more_cols = np.array([
    [1.0, 2.5, 'A', 'X', 10.0],
    [2.0, 3.1, 'B', 'Y', 15.0],
    [1.5, 2.8, 'A', 'Z', 12.0]
])

print(f"原始数据形状: {X_more_cols.shape}")

# 只对部分列应用转换器，剩余列保持不变
column_transformer_passthrough = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), [0, 1]),        # 只对前两列标准化
    ],
    remainder='passthrough'  # 未指定的列将保持原样
)

# 训练并转换
column_transformer_passthrough.fit(X_more_cols)
X_passthrough = column_transformer_passthrough.transform(X_more_cols)

print(f"转换后的数据形状: {X_passthrough.shape}")
print("转换后的数据:")
print(X_passthrough)
print("前两列被标准化，后三列保持原样")

print("\n" + "="*50)
print("ColumnTransformer 测试完成！")

```

    原始数据:
    [['1.0' '2.5' 'A' 'X']
     ['2.0' '3.1' 'B' 'Y']
     ['1.5' '2.8' 'A' 'Z']
     ['3.0' '4.0' 'C' 'X']
     ['2.5' '3.5' 'B' 'Y']
     ['1.8' '2.2' 'A' 'Z']]
    数据形状: (6, 4)
    
    列转换器训练完成
    
    拟合的转换器信息:
      scaler: StandardScaler 应用于列 [0, 1]
        均值: [1.96666667 3.01666667]
        标准差: [0.64978629 0.60392236]
      encoder: OneHotEncoder 应用于列 [2, 3]
        类别: [array(['A', 'B', 'C'], dtype='<U32'), array(['X', 'Y', 'Z'], dtype='<U32')]
    
    转换后的数据:
    [[-1.48766861 -0.85551835  1.          0.          0.          1.
       0.          0.        ]
     [ 0.05129892  0.13798683  0.          1.          0.          0.
       1.          0.        ]
     [-0.71818485 -0.35876576  1.          0.          0.          0.
       0.          1.        ]
     [ 1.59026645  1.62824461  0.          0.          1.          1.
       0.          0.        ]
     [ 0.82078268  0.80032362  0.          1.          0.          0.
       1.          0.        ]
     [-0.25649459 -1.35227095  1.          0.          0.          0.
       0.          1.        ]]
    转换后的数据形状: (6, 8)
    
    验证转换结果:
    前两列应该是标准化后的数值（均值≈0，标准差≈1）
    前两列的均值: [-1.48029737e-16  8.51170986e-16]
    前两列的标准差: [1. 1.]
    
    后面的列应该是独热编码的结果（只包含0和1）
    后面列的唯一值: [0. 1.]
    
    ==================================================
    测试 remainder='passthrough' 的情况
    原始数据形状: (3, 5)
    转换后的数据形状: (3, 5)
    转换后的数据:
    [['-1.224744871391589' '-1.2247448713915863' 'A' 'X' '10.0']
     ['1.224744871391589' '1.2247448713915918' 'B' 'Y' '15.0']
     ['0.0' '1.8129866073473575e-15' 'A' 'Z' '12.0']]
    前两列被标准化，后三列保持原样
    
    ==================================================
    ColumnTransformer 测试完成！


## Section 0.8: 进一步抽象——输入验证函数

很多情形下我们都需要对输入的数据，无论是特征矩阵还是标签向量，进行验证，以确保它们符合预期的格式和类型。这些验证通常包括：

- 检查输入是否为 NumPy 数组或 Pandas DataFrame。
- 检查输入的维度是否符合预期。
- 检查输入的类型是否符合预期（如数值型、类别型等）。
- 检查输入是否包含缺失值（NaN/Inf）。
- 检查输入的标签是否为一维数组或向量。

因此，设计一个通用的输入验证函数是非常有用的，它可以在模型训练和预测之前对输入数据进行验证，确保数据的质量和格式符合预期。这样可以避免在训练或预测过程中出现错误，提高代码的健壮性和可维护性。根据我们前面的验证内容的分析，单一验证函数 `validate_array` 的参数设计是显然的：

- `X`：被验证内容（可能是特征矩阵或标签向量），类型为 NumPy 数组或 Pandas DataFrame。
- `ensure_2d`：是否确保输入为二维数组，默认为 `True`。
- `allow_nan`：是否允许输入包含 NaN 值，默认为 `False`。
- `allow_inf`：是否允许输入包含无穷大（Inf）值，
- `dtype`：内部需要将输入转换为的 NumPy 数据类型，默认为 `None`，即不进行类型转换。

进一步设计同时检验特征矩阵和标签向量的验证函数 `validate_X_y`，通过调用 `validate_array` 函数来验证特征矩阵和标签向量的格式和类型，并检测 `X` 和 `y` 的维度是否匹配。

-----

## Section 0.9: 报告的编写与部署

使用 Jupyter Notebook 编写报告和内容展示。所有 `.ipynb` 文件都存放在 `/notebooks` 目录下。

通过 Jupyter Notebook 与 Mkdocs 联动可以简单地将报告的内容部署在网页上。对于笔者而言，由于个人知识库项目在另一个仓库中，可以使用 Jupyter 的 `nbconvert` 命令将 `.ipynb` 文件转换为 Markdown 格式，然后使用脚本将 Markdown 文件定向输出到知识库的 `docs` 目录下。这样可以将报告的内容与知识库的内容进行整合，便于后续的维护和更新。

-----
