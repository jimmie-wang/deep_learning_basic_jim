from collections import OrderedDict

import numpy as np

from layer import SoftmaxWithLoss, Affine, Relu, MySoftmaxWithLoss


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # 784*50
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)  # 50*10
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()

        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()

    # x(100,784) t(100,10)
    def gradient(self, x, t):
        y = self.forward(x) # 此时y经过了最后一层Affine
        # 由于梯度下降是针对Loss求导，这里还需要加一层softmax，并连接到交叉熵Loss，这里需要计算一下才能反向传播
        loss = self.loss(y, t)
        # 先对softmaxWithLoss做反向传播
        dout = 1
        dout = self.lastLayer.backward(dout)
        # 网络backward
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 根据反向传播更新的参数梯度值，返回给上层用于更新参数
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
    # 根据y和t计算标量loss值
    def loss(self, y, t):
        return self.lastLayer.forward(y, t)


    # 从输入层开始，往前计算，这里相当于计算完了第二层Affine，差最后一步softmax
    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def accuracy(self, x, t):
        y = self.forward(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = np.sum(y == t) / x.shape[0]
        return acc


class MyTwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)  # 784*50
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)  # 50*10
        self.params['b2'] = np.zeros(output_size)

        # 生成层
        self.layers = OrderedDict()

        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = MySoftmaxWithLoss()

    # x(100,784) t(100,10)
    def gradient(self, x, t):
        y = self.forward(x) # 此时y经过了最后一层Affine
        # 由于梯度下降是针对Loss求导，这里还需要加一层softmax，并连接到交叉熵Loss，这里需要计算一下才能反向传播
        loss = self.loss(y, t)
        # 先对softmaxWithLoss做反向传播
        dout = 1
        dout = self.lastLayer.backward(dout)
        # 网络backward
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 根据反向传播更新的参数梯度值，返回给上层用于更新参数
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
    # 根据y和t计算标量loss值
    def loss(self, y, t):
        return self.lastLayer.forward(y, t)


    # 从输入层开始，往前计算，这里相当于计算完了第二层Affine，差最后一步softmax
    def forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def accuracy(self, x, t):
        y = self.forward(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        acc = np.sum(y == t) / x.shape[0]
        return acc

