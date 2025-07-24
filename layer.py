import numpy as np

from functions import softmax, cross_entropy_error


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
    # 初始化 ReLU 层
    relu = Relu()

    # 正向传播
    x = np.array([[1.0, -0.5, 2.0], [-1.0, 3.0, 0.0]])  # 形状 (2, 3)
    out = relu.forward(x)
    print("正向传播结果：")
    print(out)
    # 输出：
    # [[1.  0.  2. ]
    #  [0.  3.  0. ]]

    # 假设上游传来的梯度
    dout = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])  # 形状 (2, 3)
    dx = relu.backward(dout)
    print("\n反向传播结果：")
    print(dx)
    # 输出：
    # [[0.1 0.  0.3]
    #  [0.  0.5 0. ]]