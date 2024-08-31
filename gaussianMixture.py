# 导入numpy，sklearn和matplotlib库
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成一个模拟的数据集，包含300个样本，每个样本有2个特征，有3个不同的簇
X, y = make_blobs(n_samples=300, n_features=2, centers=3, random_state=0)

# 创建一个混合高斯模型对象，设置簇的个数为3，协方差类型为full，最大迭代次数为100，随机种子为0
gmm = GaussianMixture(n_components=3, covariance_type='full', max_iter=100, random_state=0)

# 对数据集进行混合高斯模型聚类，返回每个样本的聚类标签
labels = gmm.fit_predict(X)

# 绘制数据集和聚类结果的图，用不同的颜色来表示不同的簇，用黑色的星号来表示质心
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], marker='*', s=200, c='black', label='centroids')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

