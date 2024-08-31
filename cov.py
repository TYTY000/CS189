import numpy as np

# 定义数据集
X = np.array([
[3, 1],
  [7, 2],
  [8, 3],
  [4, 5],
  [3, 6]
    ])

# 计算协方差矩阵
cov_matrix = np.cov(X.T)

print("协方差矩阵为：")
print(cov_matrix)
