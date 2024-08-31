# 导入所需的包
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error

# 加载加州房价数据集
housing = fetch_california_housing()

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)

# 创建一个梯度提升回归模型
gb_reg = GradientBoostingRegressor(n_estimators=10, max_depth=3, learning_rate=0.1, random_state=42)

# 训练模型
gb_reg.fit(X_train, y_train)

# 预测测试集
y_pred = gb_reg.predict(X_test)

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)

# 打印模型评估指标
print("Mean Squared Error:", mse)

# 绘制第一棵决策树
plt.figure(figsize=(12, 8))
plot_tree(gb_reg.estimators_[0, 0], feature_names=housing.feature_names, filled=True)
plt.show()

