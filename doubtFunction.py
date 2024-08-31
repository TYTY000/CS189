# 导入numpy和sklearn库
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 定义一个怀疑函数，用于计算分类器对预测结果的不确定性
def doubt_function(probs, threshold=0.5):
    # probs是一个n*k的矩阵，表示n个样本属于k个类别的概率
    # threshold是一个浮点数，表示判断概率差异是否足够大的阈值，通常取值在[0, 1]之间
    # 返回一个n维的向量，表示每个样本的不确定性，越大表示越怀疑，越小表示越确定
    n, k = probs.shape
    # 初始化一个n维的向量，表示每个样本的不确定性
    doubts = np.zeros(n)
    # 对每个样本，计算它的不确定性
    for i in range(n):
        # 找出该样本属于的最可能的类别的索引
        max_index = np.argmax(probs[i])
        # 找出该样本属于的次可能的类别的索引
        second_index = np.argsort(probs[i])[-2]
        # 计算该样本属于最可能的类别和次可能的类别的概率之差
        diff = probs[i, max_index] - probs[i, second_index]
        # 如果概率之差小于阈值，说明该样本的预测结果不确定，增加它的不确定性
        if diff < threshold:
            doubts[i] = 1 - diff
    # 返回每个样本的不确定性
    return doubts

# 生成一个模拟的数据集，包含100个样本，每个样本有2个特征，有2个不同的类别
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_redundant=0, random_state=0)

# 创建一个逻辑回归分类器对象，设置最大迭代次数为100，随机种子为0
clf = LogisticRegression(max_iter=100, random_state=0)

# 对数据集进行逻辑回归分类，返回每个样本的预测标签
clf.fit(X, y)
labels = clf.predict(X)

# 计算每个样本属于每个类别的概率
probs = clf.predict_proba(X)

# 使用怀疑函数，计算每个样本的不确定性，设置阈值为0.5
doubts = doubt_function(probs, threshold=0.5)

# 打印每个样本的预测标签和不确定性
for i in range(len(labels)):
    print(f"样本{i}的预测标签是{labels[i]}，不确定性是{doubts[i]:.2f}, 概率为{probs[i][0]:.2f} {probs[i][1]:.2f}")

