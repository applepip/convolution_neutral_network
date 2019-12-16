from mxnet import gluon, init, nd
from mxnet.gluon import nn

# VGG块的组成规律是：连续使用数个相同的填充为1、窗口形状为 3×3 的卷积层后接上
# 一个步幅为2、窗口形状为 2×2 的最大池化层。卷积层保持输入的高和宽不变，而池
# 化层则对其减半。我们使用vgg_block函数来实现这个基础的VGG块，它可以指定卷积层
# 的数量num_convs和输出通道数num_channels。

def vgg_block(num_convs, num_channels):
    blk = nn.Sequential()
    for _ in range(num_convs):
        blk.add(nn.Conv2D(num_channels, kernel_size=3,
                          padding=1, activation='relu'))
    blk.add(nn.MaxPool2D(pool_size=2, strides=2))
    return blk

# 现在我们构造一个VGG网络。它有5个卷积块，前2块使用单卷积层，而后3块使用双卷积层。
# 第一块的输出通道是64，之后每次对输出通道数翻倍，直到变为512。因为这个网络使用了
# 8个卷积层和3个全连接层，所以经常被称为VGG-11。

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

def vgg_11(conv_arch):
    net = nn.Sequential()
    # 卷积层部分
    for (num_convs, num_channels) in conv_arch:
        net.add(vgg_block(num_convs, num_channels))
    # 全连接层部分
    net.add(nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(4096, activation='relu'), nn.Dropout(0.5),
            nn.Dense(10))
    return net

net = vgg_11(conv_arch)

net.initialize()
X = nd.random.uniform(shape=(1, 1, 224, 224))
for blk in net:
    X = blk(X)
    print(blk.name, 'output shape:\t', X.shape)

# 可以看到，每次我们将输入的高和宽减半，直到最终高和宽变成7后传入全连接层。与此
# 同时，输出通道数每次翻倍，直到变成512。因为每个卷积层的窗口大小一样，所以每层
# 的模型参数尺寸和计算复杂度与输入高、输入宽、输入通道数和输出通道数的乘积成正
# 比。VGG这种高和宽减半以及通道翻倍的设计使得多数卷积层都有相同的模型参数尺寸和
# 计算复杂度。


ratio = 4
small_conv_arch = [(pair[0], pair[1] // ratio) for pair in conv_arch]
net = vgg_11(small_conv_arch)

from vgg_prepare import *

lr, num_epochs, batch_size, ctx = 0.05, 5, 128, try_gpu()
net.initialize(ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=224)
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
              num_epochs)