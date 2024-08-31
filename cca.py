import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA

# 设置随机种子
np.random.seed()

# 生成两组数据，每组有 500 个样本，3 个变量
n = 500
l1 = np.random.normal(size=n)
l2 = np.random.normal(size=n)
latents = np.array([l1, l1, l2, l2]).T
X = latents + np.random.normal(size=4 * n).reshape((n, 4))
Y = latents + np.random.normal(size=4 * n).reshape((n, 4))

# print("l1")
# print(l1)
# print("l2")
# print(l2)
# print("latents")
# print(latents)
# print("X")
# print(X)
# print("Y")
# print(Y)
# 划分训练集和测试集
X_train = X[:n // 2]
Y_train = Y[:n // 2]
X_test = X[n // 2:]
Y_test = Y[n // 2:]

# 打印相关矩阵
print("Corr(X)")
print(np.round(np.corrcoef(X.T), 2))
print("Corr(Y)")
print(np.round(np.corrcoef(Y.T), 2))

# 建立 CCA 模型，设置成分数为 2
cca = CCA(n_components=2)

# 训练数据
cca.fit(X_train, Y_train)

# 降维操作
X_train_r, Y_train_r = cca.transform(X_train, Y_train)
X_test_r, Y_test_r = cca.transform(X_test, Y_test)

# 打印测试集的相关系数
print('test corr = %.2f' % np.corrcoef(X_test_r[:, 0], Y_test_r[:, 0])[0, 1])

# 画散点图
plt.figure(figsize=(8, 6))
plt.scatter(X_train_r[:, 0], Y_train_r[:, 0], label="train data", marker="o", c="dodgerblue", s=25, alpha=0.8)
plt.scatter(X_test_r[:, 0], Y_test_r[:, 0], label="test data", marker="o", c="orangered", s=25, alpha=0.8)
plt.xlabel("x scores")
plt.ylabel("y scores")
plt.title('X vs Y (test corr = %.2f)' % np.corrcoef(X_test_r[:, 0], Y_test_r[:, 0])[0, 1])
plt.xticks(())
plt.yticks(())
plt.legend()
plt.tight_layout()
plt.show()

