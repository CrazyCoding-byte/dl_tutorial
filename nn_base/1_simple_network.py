import numpy as np

import common.functions as com


# 初始化神经网络


def init_network():
    network = {}
    # 第一层参数
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])  # 权重w=输入层也就是x1,x2*隐藏层3个 也就是2乘3 b1是隐藏层个数
    network['b1'] = np.array([0.1, 0.2, 0.3])
    # 第二层参数
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    # 第三层参数
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network


# 前向传播
def forward_propagate(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    # 逐层进行计算传递
    a1 = np.dot(x, w1) + b1
    z1 = com.sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = com.sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = com.sigmoid(a3)
