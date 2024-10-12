import numpy as np

# 数据集
x = np.array([2, 3, -2])  # 特征
y = np.array([1, 0, 2])   # 标签

# 超参数
lambda_reg = 3
learning_rate = 0.01
num_iterations = 1000

# 初始化参数 w0 和 w1
w0, w1 = 0.0, 0.0

# 定义均方误差函数 E_in(w)
def compute_E_in(w0, w1, x, y):
    predictions = w0 + w1 * x
    errors = predictions - y
    return np.mean(errors ** 2)

# 梯度下降循环
for i in range(num_iterations):
    # 计算当前的预测值和误差
    predictions = w0 + w1 * x
    errors = predictions - y

    # 计算 E_in 对 w0 和 w1 的梯度
    grad_w0 = np.mean(2 * errors)
    grad_w1 = np.mean(2 * errors * x)

    # 计算正则化项对 w0 和 w1 的次梯度
    subgrad_w0 = lambda_reg / 3 if w0 > 0 else -lambda_reg / 3 if w0 < 0 else 0
    subgrad_w1 = lambda_reg / 3 if w1 > 0 else -lambda_reg / 3 if w1 < 0 else 0

    # 合併梯度和次梯度，更新参数 w0 和 w1
    w0 -= learning_rate * (grad_w0 + subgrad_w0)
    w1 -= learning_rate * (grad_w1 + subgrad_w1)

    # 打印出每 100 次迭代的损失值
    if i % 100 == 0:
        E_in = compute_E_in(w0, w1, x, y)
        E_aug = E_in + (lambda_reg / 3) * (abs(w0) + abs(w1))
        print(f"Iteration {i}: E_aug = {E_aug:.4f}, w0 = {w0:.4f}, w1 = {w1:.4f}")

# 最终结果
print(f"Final w0: {w0}, w1: {w1}")
E_in_final = compute_E_in(w0, w1, x, y)
E_aug_final = E_in_final + (lambda_reg / 3) * (abs(w0) + abs(w1))
print(f"Final E_aug = {E_aug_final:.4f}")
