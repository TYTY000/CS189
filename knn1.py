from sklearn.datasets import load_iris
from sklearn import neighbors
from sklearn.model_selection import train_test_split

iris = load_iris()
x = iris.data
y = iris.target
# 将数据集七三分，70%是用于搭建模型，30%用于测试
x_train , x_test ,y_train , y_test = train_test_split(x, y, test_size=0.3, random_state=100)

# 定义KNN分类器
model = neighbors.KNeighborsClassifier(n_neighbors=5, weights="uniform",p=2, 
                                       algorithm="kd_tree",n_jobs=-1)
"""
n_neighbors=5: K值
weight="uniform", 无加权距离
p=2, 采用欧氏距离，“1”表示曼哈顿距离
algorithm："brute"即遍历所有样本数据，"kd_tree"即KD树
n_jobs, 使用CPU的核心数, -1表示所有
"""
model.fit(x_train, y_train)
print(model.score(x_test, y_test) )
