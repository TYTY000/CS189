# 导入所需的库
import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 读取数据集，这里使用的是鸢尾花数据集
iris = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# 查看数据集的基本信息
print(iris.head())
print(iris.info())
print(iris.describe())

# 将数据集分为特征和标签
X = iris.drop("species", axis=1)
y = iris["species"]

# 将数据集分为训练集和测试集
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树分类器
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=42)

# 训练决策树分类器
dtc.fit(X_train, y_train)

# 预测测试集的结果
y_pred = dtc.predict(X_test)

# 评估决策树分类器的性能
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification report:\n", classification_report(y_test, y_pred))

# 可视化决策树
from sklearn.tree import plot_tree
plt.figure(figsize=(12, 8))
plot_tree(dtc, feature_names=X.columns, class_names=y.unique(), filled=True, rounded=True)
plt.show()

