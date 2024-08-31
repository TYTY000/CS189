from sklearn.manifold import MDS
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = load_iris()
data = iris.data

# 创建MDS对象
mds = MDS(n_components=2, normalized_stress=False)

# 使用MDS降维
data_2d = mds.fit_transform(data)

# 可视化降维后的数据
plt.scatter(data_2d[:, 0], data_2d[:, 1], c=iris.target)
plt.show()
