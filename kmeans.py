# 导入numpy和matplotlib库
import numpy as np
import matplotlib.pyplot as plt

# 定义一个函数，用于计算两个向量之间的欧氏距离
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# 定义一个函数，用于初始化k个随机质心
def initialize_centroids(X, k):
    # X是一个n*m的矩阵，表示n个样本，每个样本有m个特征
    # k是一个整数，表示要聚类的个数
    # 返回一个k*m的矩阵，表示k个随机质心
    n, m = X.shape
    centroids = np.zeros((k, m))
    for i in range(k):
        # 随机选择一个样本作为质心
        index = np.random.choice(n)
        centroids[i] = X[index]
    return centroids

# 定义一个函数，用于对给定的数据集进行k-means聚类
def kmeans(X, k, max_iter=100):
    # X是一个n*m的矩阵，表示n个样本，每个样本有m个特征
    # k是一个整数，表示要聚类的个数
    # max_iter是一个整数，表示最大迭代次数
    # 返回一个n维的向量，表示每个样本的聚类标签，以及一个k*m的矩阵，表示最终的质心
    n, m = X.shape
    # 初始化k个随机质心
    centroids = initialize_centroids(X, k)
    # 初始化一个n维的向量，表示每个样本的聚类标签
    labels = np.zeros(n)
    # 迭代max_iter次
    for i in range(max_iter):
        # 对每个样本，计算它与每个质心的距离，选择最近的质心作为它的聚类标签
        for j in range(n):
            distances = np.zeros(k)
            for l in range(k):
                distances[l] = euclidean_distance(X[j], centroids[l])
            labels[j] = np.argmin(distances)
        # 对每个聚类，计算它的所有样本的均值，作为新的质心
        for l in range(k):
            # 找出属于该聚类的样本的索引
            indices = np.where(labels == l)[0]
            # 如果该聚类没有样本，随机选择一个样本作为质心
            if len(indices) == 0:
                index = np.random.choice(n)
                centroids[l] = X[index]
            else:
                # 计算该聚类的所有样本的均值
                centroids[l] = np.mean(X[indices], axis=0)
    # 返回聚类标签和质心
    return labels, centroids

# 生成一个模拟的数据集，包含150个样本，每个样本有2个特征
X = np.zeros((200, 2))
# 生成3个不同的高斯分布，每个分布有50个样本
np.random.seed(0)
X[:50, 0] = np.random.normal(0, 0.5, 50)
X[:50, 1] = np.random.normal(0, 0.5, 50)
X[50:100, 0] = np.random.normal(3, 0.5, 50)
X[50:100, 1] = np.random.normal(0, 0.5, 50)
X[100:150, 0] = np.random.normal(0, 0.5, 50)
X[100:150, 1] = np.random.normal(3, 0.5, 50)
X[150:, 0] = np.random.normal(3, 0.5, 50)
X[150:, 1] = np.random.normal(3, 0.5, 50)

# 对数据集进行k-means聚类，设置k为3
labels, centroids = kmeans(X, 4)

# 绘制数据集和聚类结果
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='rainbow')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='black', label='centroids')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

