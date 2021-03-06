# 检测图像中物体的边缘，即找到像素变化的位置。首先我们构造一张 6×8 的图像（即高
# # 和宽分别为6像素和8像素的图像）。它中间4列为黑（0），其余为白（1）。

from  Conv2D import *

X = nd.ones((6, 8))
X[:, 2:6] = 0
print(X)

# 然后我们构造一个高和宽分别为1和2的卷积核K。当它与输入做互相关运算时，如
# 果横向相邻元素相同，输出为0；否则输出为非0。

K = nd.array([[1, -1]])

# 和我们设计的卷积核K做互相关运算。可以看出，我们将从白到黑的边缘和从黑到
# 白的边缘分别检测成了1和-1。其余部分的输出全是0。由此，我们可以看出，卷积
# 层可通过重复使用卷积核有效地表征局部空间。

Y = corr2d(X, K)
print(Y)