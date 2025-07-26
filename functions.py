# coding: utf-8
import numpy as np


def identity_function(x):
    return x


def step_function(x):
    return np.array(x > 0, dtype=np.int)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


def my_softmax(x):
    if x.ndim == 2:
        exp_x = np.exp(x)
        sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)  # 1 为按行，0是跨列计算，这里还要保留维度为(2,1)，避免成(2,)，方便后续广播计算
        return exp_x / sum_exp_x
    x = x - np.max(x) # 多维的，全局求最大
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1 or t.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def my_cross_entropy_error(y, t): # 交叉熵一般用于计算分类问题的损失函数，-tk*logyk，其中tk是对应正确解标签的值1，这一项就是-logyk
    if y.ndim == 1: # 一个样本的情况下，y=[0.3,0.4] t=[0,1] or t=[1] 这里reshape套一个括号，作用是后续能统一按多个样本进行处理
        y = y.reshape(1, y.size)
        t = t.reshape(1, t.size)
    if y.size == t.size: # 需要正确的下标去y里取yk
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    # np的高级索引：各个索引位都是数组，轮番取值，作为索引
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X, t):
    y = softmax(X)
    return cross_entropy_error(y, t)

if __name__ == '__main__':
    x = np.array([[1,4,6],[3,5,7]])
    y1 = softmax(x)
    y2 = my_softmax(x)

    print(y1)
    print(y2)