# 当输入通道有多个时，因为我们对各个通道的结果做了累加，所以不论输入通道数
# 是多少，输出通道数总是为1。设卷积核输入通道数和输出通道数分别为 ci 和 co，
# 高和宽分别为 kh 和 kw 。如果希望得到含多个通道的输出，我们可以为每个输出
# 通道分别创建形状为 ci×kh×kw 的核数组。

from multi_input import *


def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])

K = nd.stack(K, K + 1, K + 2)

print(K.shape)

out = corr2d_multi_in_out(X, K)

print(out)



