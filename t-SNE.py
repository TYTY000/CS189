import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载数据
X, y = load_digits(return_X_y=True)

# 创建并拟合TSNE模型
tsne = TSNE(n_components=2, init='random', random_state=501)
X_tsne = tsne.fit_transform(X)

# 可视化结果
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.show()
