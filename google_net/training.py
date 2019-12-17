from function_tools import *
from google_net import *

lr, num_epochs, batch_size, ctx = 0.1, 5, 128, try_gpu()
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
              num_epochs)