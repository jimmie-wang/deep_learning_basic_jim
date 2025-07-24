# This is a sample Python script.
import numpy as np

from dataset.dataset import load_mnist
from two_layer_net import TwoLayerNet


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# 实现一个两层的全连接线性神经网络，训练mini set
# 输入-Affine1（ReLU）-Affine2（sigmoid）-输出

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi(100 / 6)

    # 读入数据
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    iter_nums = 10000
    batch_size = 100
    lr = 0.1
    train_size = x_train.shape[0]

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    iter_per_epoch = max(train_size / batch_size, 1)

    network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

    cnt = 0

    for i in range(iter_nums):
        # 1.抽取随机样本
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 2.求梯度，包括前向传播，反向传播
        grad = network.gradient(x_batch, t_batch)

        # 3.梯度更新
        for key in ('W1', 'W2', 'b1', 'b2'):
            network.params[key] -= lr * grad[key]

        # 4.计算loss，记录
        y = network.forward(x_batch)
        loss = network.loss(y, t_batch)
        train_loss_list.append(loss)

        # op：统计一个epoch的训练数据
        if i % iter_per_epoch == 0:
            cnt += 1
            train_acc = network.accuracy(x_train, t_train)
            test_acc = network.accuracy(x_test, t_test)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
            print(train_acc, test_acc)

    print('Finished, all times for calculate accuracy:'+ str(cnt))
