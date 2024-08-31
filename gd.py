# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt

# 定义目标函数 y = x^2
def f(x):
    return x**2

# 定义目标函数 y = x1^2 + x2^2
def f1(x):
    return np.apply_along_axis(lambda x: x[0]**2 + x[1]**2, 0, x)

# 定义目标函数的导数（梯度） y' = 2x
def f_grad(x):
    return 2*x

# 定义梯度下降法
def gradient_descent(x0, alpha, epsilon, max_iter):
    # x0: 初始点
    # alpha: 学习率
    # epsilon: 容差
    # max_iter: 最大迭代次数
    x = x0 # 初始化 x
    fx = f(x) # 计算初始点的函数值
    history_x = [x] # 记录 x 的历史值
    history_fx = [fx] # 记录 f(x) 的历史值
    iter = 0 # 记录迭代次数
    while iter < max_iter and np.linalg.norm(fx) > epsilon: # 当迭代次数小于最大值且函数值大于容差时
        x = x - alpha * f_grad(x) # 沿着负梯度方向更新 x
        fx = f(x) # 计算更新后的函数值
        history_x.append(x) # 记录 x 的历史值
        history_fx.append(fx) # 记录 f(x) 的历史值
        iter += 1 # 迭代次数加一
    return x, fx, history_x, history_fx # 返回最终的 x, f(x) 和历史值

# 定义随机梯度下降法
def stochastic_gradient_descent(x0, alpha, epsilon, max_iter):
    # x0: 初始点
    # alpha: 学习率
    # epsilon: 容差
    # max_iter: 最大迭代次数
    x = x0 # 初始化 x
    fx = f(x) # 计算初始点的函数值
    history_x = [x] # 记录 x 的历史值
    history_fx = [fx] # 记录 f(x) 的历史值
    iter = 0 # 记录迭代次数
    while iter < max_iter and np.linalg.norm(fx) > epsilon: # 当迭代次数小于最大值且函数值大于容差时
        i = np.random.randint(0, 2) # 随机选择一个维度
        xi = x[i] # 取出该维度的值
        fxi = f(xi) # 计算该维度的函数值
        x[i] = xi - alpha * f_grad(xi) # 沿着负梯度方向更新该维度的值
        fx = f(x) # 计算更新后的函数值
        history_x.append(x.copy()) # 记录 x 的历史值（注意要复制一份，否则会被覆盖）
        history_fx.append(fx) # 记录 f(x) 的历史值
        iter += 1 # 迭代次数加一
    return x, fx, history_x, history_fx # 返回最终的 x, f(x) 和历史值

# 设置参数
x0 = np.array([10, 10]) # 初始点
alpha = 0.1 # 学习率
epsilon = 1e-6 # 容差
max_iter = 100 # 最大迭代次数

# 调用梯度下降法和随机梯度下降法
x_gd, fx_gd, history_x_gd, history_fx_gd = gradient_descent(x0, alpha, epsilon, max_iter)
x_sgd, fx_sgd, history_x_sgd, history_fx_sgd = stochastic_gradient_descent(x0, alpha, epsilon, max_iter)

# 打印结果
print("梯度下降法的结果：")
print("x =", x_gd)
print("f(x) =", fx_gd)
print("迭代次数 =", len(history_x_gd))
print("随机梯度下降法的结果：")
print("x =", x_sgd)
print("f(x) =", fx_sgd)
print("迭代次数 =", len(history_x_sgd))

# 绘制目标函数的等高线图
x1 = np.linspace(-10, 10, 100)
x2 = np.linspace(-10, 10, 100)
X1, X2 = np.meshgrid(x1, x2)
Y = f1(np.array([X1, X2]))
plt.contour(X1, X2, Y, levels=np.logspace(-2, 3, 20))
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Contour plot of f(x)")

# 绘制梯度下降法的下降轨迹
history_x1_gd = np.array(history_x_gd)[:, 0]
history_x2_gd = np.array(history_x_gd)[:, 1]
plt.plot(history_x1_gd, history_x2_gd, "bo-", label="Gradient Descent")

# 绘制随机梯度下降法的下降轨迹
history_x1_sgd = np.array(history_x_sgd)[:, 0]
history_x2_sgd = np.array(history_x_sgd)[:, 1]
plt.plot(history_x1_sgd, history_x2_sgd, "ro-", label="Stochastic Gradient Descent")

# 添加图例和显示图像
plt.legend()
plt.show()

