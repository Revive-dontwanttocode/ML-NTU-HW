import numpy as np

# 数据集
x = np.array([2, -2])
y = np.array([9, -1])

# 超参数
lambda_reg = 8  # 从这个值开始调整
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

    # 计算正则化项对 w0 和 w1 的梯度
    grad_w0 += lambda_reg * w0
    grad_w1 += lambda_reg * w1

    # 更新参数 w0 和 w1
    w0 -= learning_rate * grad_w0
    w1 -= learning_rate * grad_w1

    # 打印出每 100 次迭代的损失值
    if i % 100 == 0:
        E_in = compute_E_in(w0, w1, x, y)
        E_aug = E_in + (lambda_reg / 2) * (w0 ** 2 + w1 ** 2)
        print(f"Iteration {i}: E_aug = {E_aug:.4f}, w0 = {w0:.4f}, w1 = {w1:.4f}")

print(f"Final w0: {w0}, w1: {w1}")
test_x = 1
y_pred = w0 + w1 * test_x
print(f"Prediction for x = 1: y = {y_pred}")
