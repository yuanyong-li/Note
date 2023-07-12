# pytorch

```
# 远程服务器
10.99.4.2:8090
1131272948
```

## 1. 基础

### Tensor

#### 创建tensor

`torch.empty(5,3)`

`torch.rand(5,3)`

`torch.zeros(5,3,dtype=torch.long)`

`torch.tensor([5.5,3],dtype=torch.float64)`

`torch.new_ones(x,5,3,dtype=torch.float64)`: 返回的tensor默认具有相同的torch.dtype和torch.device

`torch.randn_like(x, dtype=torch.float)`: 返回的默认有相同形状

`x.size()`
`x.shape`

|               函数                |           功能            |
| :-------------------------------: | :-----------------------: |
|          Tensor(*sizes)           |       基础构造函数        |
|           tensor(data,)           |  类似np.array的构造函数   |
|           ones(*sizes)            |         全1Tensor         |
|           zeros(*sizes)           |         全0Tensor         |
|            eye(*sizes)            |    对角线为1，其他为0     |
|         arange(s,e,step)          |    从s到e，步长为step     |
|        linspace(s,e,steps)        | 从s到e，均匀切分成steps份 |
|        rand/randn(*sizes)         |       均匀/标准分布       |
| normal(mean,std)/uniform(from,to) |     正态分布/均匀分布     |
|            randperm(m)            |         随机排列          |

#### 操作tensor

`a+b`
`torch.add(a, b, out=result)`
`result = y.add_(x)`: inplace操作，直接修改原内存（别用）

`y = x.view(-1, 10)`: -1所指的维度可以根据其他维度推出来，view函数修改新变量原变量也改变
`y = x.reshape(-1, 10)`: 不保证返回的是拷贝
`y = x.clone().view(-1, 10)`

|     函数     |               功能                |
| :----------: | :-------------------------------: |
|    trace     |     对角线元素之和(矩阵的迹)      |
|     diag     |            对角线元素             |
|  triu/tril   | 矩阵的上三角/下三角，可指定偏移量 |
|    mm/bmm    |     矩阵乘法，batch的矩阵乘法     |
| addmm/addbmm |             矩阵运算              |
|      t       |               转置                |
|  dot/cross   |             内积/外积             |
|   inverse    |             求逆矩阵              |
|     svd      |            奇异值分解             |

#### Tensor on GPU

```python
if torch.cuda.is_available():
    x = torch.rand(3,3)
    y = torch.ones_like(x, device="cuda")
    x = x.to("cuda")
    print(x.to("cpu",torch.double))
```

### Autograd

```python
x = torch.ones(2,2,requires_grad=True)
y = x + 2
print(y.grad_fn) # =<AddBackward0 object at 0x7f765c5eb880>
print(x.is_leaf, y.is_leaf) #叶子节点

y.requires_grad_(True) #采用in-place方法改变属性

out = (y*y*3).mean()
out.backward()
print(x.grad)
```

```python
x = torch.tensor(1.0, requires_grad=True)
y1 = x ** 2 
with torch.no_grad():
    y2 = x ** 3
y3 = y1 + y2
    
print(x.requires_grad)
print(y1, y1.requires_grad) # True
print(y2, y2.requires_grad) # False
print(y3, y3.requires_grad) # True

y3.backward()
print(x.grad)

# 重置grad
x.grad.data.zero_()
```

## 2. DL_basics

### 线性回归

预测表达式：$\hat{y} = x_1 w_1 + x_2 w_2 + b$

损失函数：$\ell(w_1, w_2, b) =\frac{1}{n} \sum_{i=1}^n \frac{1}{2}\left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right)^2$

梯度更新：$|\mathcal{B}|$ 代表每个小批量中的样本个数, $\eta$ 学习率

$\begin{aligned}
w_1 &\leftarrow w_1 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{ \partial \ell^{(i)}(w_1, w_2, b)  }{\partial w_1} = w_1 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}x_1^{(i)} \left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right),\\
w_2 &\leftarrow w_2 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{ \partial \ell^{(i)}(w_1, w_2, b)  }{\partial w_2} = w_2 -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}x_2^{(i)} \left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right),\\
b &\leftarrow b -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{ \partial \ell^{(i)}(w_1, w_2, b)  }{\partial b} = b -   \frac{\eta}{|\mathcal{B}|} \sum_{i \in \mathcal{B}}\left(x_1^{(i)} w_1 + x_2^{(i)} w_2 + b - y^{(i)}\right).
\end{aligned}$

```python
%matplotlib inline
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
```

#### 构造数据集

构造数据集：$\boldsymbol{y} = \boldsymbol{X}\boldsymbol{w} + b + \epsilon$

使用线性回归模型真实权重 $\boldsymbol{w} = [2, -3.4]^\top$ 和偏差 $b = 4.2$，以及一个随机噪声项 $\epsilon$ 来生成标签

```python
from d2lzh_pytorch import *

set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
```

```python
for X, y in data_iter(batch_size=10, features, labels):
    print(X, y)
    break
```

#### 初始化权重

```python
w = torch.tensor(np.random.normal(0, 0.01, (nums_inputs,1)),dtype=torch.float32,requires_grad=True)
b = torch.zeros(1, dtype=torch.float32,requires_grad=True)
```

也可以单独设置requires_grad

```python
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
```

```python
# 线性回归函数
def linreg(X, w, b)
	return torch.mm(X, w) + b
# 损失函数 返回值是向量，没有除以2
def squared_loss(y_hat, y):
  return (y_hat - y.view(y_hat.size())) ** 2 / 2
# sgd
def sgd(params, lr, batch_size):
  for param in params:
    # 希望修改param的值，但不影响反向传播，需要用到.data
    param.data -= lr * param.grad / batch_size
```



### 线性回归简易实现

#### 生成数据

```python
num_inputs = 2
num_examples = 1000
true_w = [2, -3.4]
true_b = 4.2
features = torch.tensor(np.random.normal(0, 1, (num_examples, num_inputs)), dtype=torch.float)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float)
```

#### 读取数据（有用）

```python
import torch.utils.data as Data

batch_size = 10
# 将训练数据的特征和标签组合
dataset = Data.TensorDataset(features, labels)
# 随机读取小批量
data_iter = Data.DataLoader(dataset, batch_size, shuffle=True)
```



#### 定义模型

```python
# 继承nn.Module
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        super(LinearNet, self).__init__()
        self.liear = nn.Linear(n_feature, 1)
    
    def forward(self, x):
        y = self.linear(x)
        return y
    
net = LinearNet(num_inputs)
print(net)
```

```python
# 使用nn.Sequential
# 写法一
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # 此处还可以传入其他层
    )

# 写法二
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))
# net.add_module ......

# 写法三
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
          ('linear', nn.Linear(num_inputs, 1))
          # ......
        ]))

print(net)
print(net[0])

# 查看可学习参数
for param in net.parameters():
	print(param)
```

>   注意：`torch.nn`仅支持输入一个batch的样本不支持单个样本输入，如果只有单个样本，可使用`input.unsqueeze(0)`来添加一维。

#### 初始化模型参数

```python
from torch.nn import init

init.normal_(net[0].weight, mean=0, std=0.01)
init.constant_(net[0].bias, val=0)
#net[0].bias.data.fill_(0)
```

#### 定义损失函数

```python
loss = nn.MSELoss()
```

#### 定义优化算法

```python
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)
```

对不同子网络设置不同的学习率

```python
optimizer =optim.SGD([
                # 如果对某个参数不指定学习率，就使用最外层的默认学习率
                {'params': net.subnet1.parameters()}, # lr=0.03
                {'params': net.subnet2.parameters(), 'lr': 0.01}
            ], lr=0.03)
```

```python
# 调整学习率
for param_group in optimizer.param_groups:
    param_group['lr'] *= 0.1 # 学习率为之前的0.1倍
```

#### 训练模型

在使用Gluon训练模型时，我们通过调用`optimizer`实例的`step`函数来迭代模型参数。按照小批量随机梯度下降的定义，我们在`step`函数中指明批量大小，从而对批量中样本梯度求平均。

```python
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
```

#### 实例

```python
%matplotlib inline
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch.nn

input_features = 3
X = torch.tensor(np.random.normal(0,1,(1000, input_features)), dtype=torch.float)
true_w = torch.tensor([1, -1, 2], dtype=torch.float)
true_b = 5
y = torch.mm(X, true_w.view(-1,1))+true_b
y += torch.tensor(np.random.normal(0, 0.01, size=y.size()),dtype=torch.float)
print(X[0:5])
loss = torch.nn.MSELoss()

net = torch.nn.Sequential(torch.nn.Linear(input_features,1))
optimizer = torch.optim.SGD(net.parameters() ,lr=0.03, weight_decay=0.1)

batch_size = 50
dataset = torch.utils.data.TensorDataset(X, y)
data_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

torch.nn.init.normal_(net[0].weight, mean=0, std=0.01)
torch.nn.init.constant_(net[0].bias, val=0)
[i for i in net[0].parameters()]

num_epochs = 30
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        optimizer.zero_grad() # 梯度清零，等价于net.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))
print([i for i in net.parameters()])
```



### 图像分类数据集(Fashion-MNIST)

1. `torchvision.datasets`: 一些加载数据的函数及常用的数据集接口；
2. `torchvision.models`: 包含常用的模型结构（含预训练模型），例如AlexNet、VGG、ResNet等；
3. `torchvision.transforms`: 常用的图片变换，例如裁剪、旋转等；
4. `torchvision.utils`: 其他的一些有用的方法。

```python
import torch
import torchvision
import matplotlib.pyplot as plt
import time
import sys
import d2lzh_pytorch as d2l

mnist_train = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='./Datasets/FashionMNIST', train=False, download=True, transform=transforms.ToTensor())

print(len(mnist_train), len(mnist_test))

X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
d2l.show_fashion_mnist(X, d2l.get_fashion_mnist_labels(y))
```

多进程加速读取

```python
batch_size = 256
if sys.platform.startswith('win'):
    num_workers = 0  # 0表示不用额外的进程来加速读取数据
else:
    num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))
```



### Softmax-regression

```python
import torch
import torchvision
import numpy as np
import sys
import d2lzh_pytorch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size,root='./Datasets/FashionMNIST')

num_inputs = 784 #28*28
num_outputs = 10 #10个最终分类

# 初始化参数
W = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),dtype=torch.float,requires_grad=True)
b = torch.zeros(num_outputs, dtype=torch.float,requires_grad=True)
print(b)

def softmax(X):
    X_exp = X.exp()
    X_rowsum = X_exp.sum(dim=1, keepdim=True)
    return X_exp / X_rowsum

def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)

def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
		# net(X).argmax(dim=1) dim: the dimension to reduce
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n

# 训练模型
num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh包中方便以后使用
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()
            
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            
            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到
                
            # 统计结果
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)
```

### MLP

```python
%matplotlib inline
import torch
import numpy as np
import matplotlib.pyplot as plt
import d2lzh_pytorch as d2l

num_inputs, num_outputs, num_hiddens = 784, 10, 256

from collections import OrderedDict
# 定义模型
net = torch.nn.Sequential(OrderedDict([
    ('flatten', d2l.FlattenLayer()),
    ('Linear1', torch.nn.Linear(num_inputs, num_hiddens)),
    ('ReLU1', torch.nn.ReLU()),
    ('Linear2', torch.nn.Linear(num_hiddens, num_outputs))
]))

# 初始化参数
for params in net.parameters():
    torch.nn.init.normal_(params, mean=0, std=0.01)
torch.nn.init.constant_(net[1].bias, val=0)
torch.nn.init.constant_(net[3].bias, val=0)

# 初始化数据loader以及优化器
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
```

### Weight Decay

```python
%matplotlib inline
import torch
import torch.nn as nn
import numpy as np
import d2lzh_pytorch as d2l

num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
drop_prob1, drop_prob2 = 0.2, 0.5
net = nn.Sequential(
        d2l.FlattenLayer(),
        nn.Linear(num_inputs, num_hiddens1),
        nn.ReLU(),
        nn.Dropout(drop_prob1),
        nn.Linear(num_hiddens1, num_hiddens2), 
        nn.ReLU(),
        nn.Dropout(drop_prob2),
        nn.Linear(num_hiddens2, 10)
        )
for param in net.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.5)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)
```

## 3. DL_computation

### model_construction

#### 继承Module类

```python
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)
net = MLP()
```

### Parameters

```python
import torch
from torch import nn
from torch.nn import init

net = nn.Sequential(nn.Linear(4, 3), nn.ReLU(), nn.Linear(3, 1))  # pytorch已进行默认初始化

print(net)
# 查看某一层的参数
print(net[2].state_dict())

X = torch.rand(2, 4)
Y = net(X).sum()

# 访问模型参数
print(type(net.named_parameters()))
for name, param in net.named_parameters():
    print(name, param.size())
    
print(net[0])

# 初始化权重
# 这里*表示解构，将原本的list的各个组成部分拆分
print(*[(name, param.shape) for name, param in net[0].named_parameters()])
for name, param in net.named_parameters():
    if 'weight' in name:
        init.normal_(param, mean=0, std=0.01)
        print(name, param.data)
    if 'bias' in name:
        init.constant_(param, val=0)
        print(name, param.data)
        
def init_normal_(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, mean=0, std=0.01)
        nn.init.zeros_(m.bias)
# 对每一层都应用init_normal_函数
net[0].apply(init_normal_)
net.apply(init_normal_)

net[0].weight.data[:] += 1
```

```python
# 定义自定义初始化参数的方法
def init_weight_(tensor):
    with torch.no_grad():
        tensor.uniform_(-10, 10)
        tensor *= (tensor.abs() >= 5).float()

for name, param in net.named_parameters():
    if 'weight' in name:
        init_weight_(param)
        print(name, param.data)

# 不影响梯度的修改方法
for name, param in net.named_parameters():
    if 'bias' in name:
        param.data += 1
        print(name, param.data)
        
# 同一对象共享参数
linear = nn.Linear(1, 1, bias=False)
net = nn.Sequential(linear, linear) 
print(net)
for name, param in net.named_parameters():
    init.constant_(param, val=3)
    print(name, param.data)
```

### 自定义层

#### 不含模型参数的自定义层

```python
import torch
from torch import nn

class CenteredLayer(nn.Module):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()
    
layer = CenteredLayer()
layer(torch.tensor([1, 2, 3, 4, 5], dtype=torch.float))

net = nn.Sequential(nn.Linear(8, 128), CenteredLayer())
y = net(torch.rand(4, 8))
y.mean().item()
```

#### 含模型参数的自定义层

```python
class MyDense(nn.Module):
    def __init__(self):
        super(MyDense, self).__init__()
        self.params = nn.ParameterList([nn.Parameter(torch.randn(4, 4)) for i in range(3)])
        self.params.append(nn.Parameter(torch.randn(4, 1)))
        self.weight = nn.Parameter(torch.randn(4,4))
        self.bias = nn.Parameter(torch.randn(4))

    def forward(self, x):
        for i in range(len(self.params)):
            x = torch.mm(x, self.params[i])
        return x
net = MyDense()
print(net)
```

### 读写

存储单个tensor或list

```python
import torch
from torch import nn
from torch.nn import functional as F

x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
```

存储MLP

```python
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden = nn.Linear(784, 256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256, 10)
    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)
net = MLP()

# 保存模型
torch.save(net.state_dict(), 'mlp.params')
clone = MLP()
clone.load_state_dict(torch.load("mlp.params"))
clone.eval()
y_clone = clone(net)
y_clone == y
```

### GPU

```python
import torch
from torch import nn

torch.device('cpu')
torch.cuda.device('cuda')
torch.cuda.device('cuda:1')
torch.cuda.device_count()

x = torch.rand(3,3, device='cuda:3')
x.to(device = 'cuda:3')
```

## 4. CNN

### 4.1 卷积神经网络

```python
import torch
from torch import nn


# 为了方便起见，我们定义了一个计算卷积层的函数。
# 此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数
def comp_conv2d(conv2d, X):
    # 这里的（1，1）表示批量大小和通道数都是1
    print(X.shape)
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    # 省略前两个维度：批量大小和通道
    print(Y.shape)
    return Y.reshape(Y.shape[2:])

# 请注意，这里每边都填充了1行或1列，因此总共添加了2行或2列

conv2d = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=2)
comp_conv2d(conv2d, X).shape
```

### 4.2 现代卷积神经网络

#### 4.2.1 LeNet

```python
import os
import time
import torch
from torch import nn, optim

import sys
sys.path.append("..") 
import d2lzh_pytorch as d2l

# control which GPUs are visible to an application.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 检测cuda是否能用，否则用cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.__version__)
print(device)
```

定义模型

```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, 5), # in_channels, out_channels, kernel_size
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2), # kernel_size, stride
            nn.Conv2d(6, 16, 5),
            nn.Sigmoid(),
            nn.MaxPool2d(2, 2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16*4*4, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
```

