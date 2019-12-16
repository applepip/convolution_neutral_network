# 当输入数据含多个通道时，我们需要构造一个输入通道数与输入数据的通道数相同的卷积核，
# 从而能够与含多通道的输入数据做互相关运算。假设输入数据的通道数为 ci ，那么卷积核
# 的输入通道数同样为 ci 。设卷积核窗口形状为 kh×kw 。当 ci=1 时，我们知道卷积核只
# 包含一个形状为 kh×kw 的二维数组。当 ci>1 时，我们将会为每个输入通道各分配一个形
# 状为 kh×kw 的核数组。把这 ci 个数组在输入通道维上连结，即得到一个形状为 ci×kh×kw
# 的卷积核。由于输入和卷积核各有 ci 个通道，我们可以在各个通道上对输入的二维数组和
# 卷积核的二维核数组做互相关运算，再将这 ci 个互相关运算的二维输出按通道相加，得到
# 一个二维数组。这就是含多个通道的输入数据与多输入通道的卷积核做二维互相关运算的输出。

from mxnet import nd

def corr2d(X, K):
    """Compute 2D cross-correlation."""
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

def corr2d_multi_in(X, K):
    # 首先沿着X和K的第0维（通道维）遍历。然后使用*将结果列表变成add_n函数的位置参数
    # （positional argument）来进行相加
    return nd.add_n(*[corr2d(x, k) for x, k in zip(X, K)])

X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

C = corr2d_multi_in(X, K)

print(C)
