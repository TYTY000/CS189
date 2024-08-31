# 导入所需的库
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 加载 iris 数据集
iris = load_iris()
X = iris.data # 特征矩阵
y = iris.target # 目标向量

# 创建随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf.fit(X, y)

# 预测新的数据
X_new = [[5.5, 3.5, 1.3, 0.2], # 属于第 0 类
         [6.7, 3.0, 5.2, 2.3]] # 属于第 2 类
y_pred = rf.predict(X_new)

# 输出预测结果
print("Predicted labels:", y_pred)

# 计算模型在整个数据集上的准确率
y_true = y # 真实标签
y_pred = rf.predict(X) # 预测标签
acc = accuracy_score(y_true, y_pred) # 准确率
print("Accuracy on the whole dataset:", acc)
