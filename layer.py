import numpy as np

from functions import softmax, cross_entropy_error, my_softmax

"""
    神经网络由各种类型的层构成
    每个layer要做：
    1.forward：前向传播，predict进行函数计算，保留反向传播时必要的依赖值，可以是x，w，y
    2.backward：反向传播，从Loss标量开始，对所有参数进行求导（求梯度），并保存梯度的同时，将自变量参数X被Loss的微分往下传
"""


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 监督数据

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 监督数据是one-hot-vector的情况
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx

# 这里的y，是一个(N,10)的结果矩阵
class MySoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y_softmax = None
        self.t = None
    def forward(self, y, t):
        self.t = t
        self.y_softmax = my_softmax(y)
        # 似乎没有必要一定算一下loss？有必要，为了计算结果统一，一次前向传播过程，就是遵循网络结构进行计算
        self.loss = cross_entropy_error(self.y_softmax, t)
        return self.loss

    def backward(self, dout=1):
        # 交叉熵损失函数，即Loss，对第二层输出y怎么求导？返回一个y同形状向量
        batch_size = self.t.shape[0] # 这里Loss是有一个平均的动作1/N在的，因此求导需要乘上
        if self.t.size == self.y_softmax.size: # 01编码
            dx = (self.y_softmax - self.t) / batch_size
            return dx
        else:
            dx = self.y_softmax.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
            return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None

        # 权重和偏置参数的导数
        self.dW = None
        self.db = None
        return
    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b
    def backward(self, dout):
        # 求参数梯度
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        # 返回自变量X的梯度，因为X还可能是关联了其他变量的函数
        return np.dot(dout, self.W.T)
class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
class Sigmoid:
    def __init__(self):
        self.y = None
        return
    def forward(self, x):
        self.y = 1/(1+np.exp(-x))
        return self.y
    def backward(self, dout):
        return dout*(self.y*(1-self.y))

if __name__ == '__main__':
    y = np.array([[0.9,0.05,0.05],
                  [0.3,0.7,0.00]])
    t = np.array([[1,0,0],
                  [0,1,0]])
    softmax_layer1 = SoftmaxWithLoss()
    softmax_layer2 = MySoftmaxWithLoss()
    f1 = softmax_layer1.forward(y,t)
    f2 = softmax_layer2.forward(y,t)
    print(f1)
    print(f2)
    dx1 = softmax_layer1.backward(dout=1)
    dx2 = softmax_layer2.backward(dout=1)
    print(dx1)
    print(dx2)