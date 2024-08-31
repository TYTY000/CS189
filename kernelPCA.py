from sklearn.decomposition import KernelPCA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 创建并拟合KernelPCA模型
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X)

# 可视化结果
plt.scatter(X_kpca[:, 0], X_kpca[:, 1], c=y)
plt.show()

