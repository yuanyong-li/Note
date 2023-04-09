# 2023-04-09

# 2023-04-06

出门办事

下午晚上读论文BERT以及GLM

BERT论文

# 2023-04-02

GRU：门控循环单元
LSTM：长短期记忆

Bidirectional RNN：双向RNN

## NLP

词嵌入、矩阵嵌入
Word2Vec、Skip-Gram、GloVe（没太懂。有些复杂）

*突然有的想法。既然可以训练word-embedding矩阵，且是不可解释的。能否转换为可解释的字段。*

词嵌入偏差去除（深度学习中的政治正确）

## seq2seq

Beam search algorithm
Refinements to beam search 细化集束搜索

# 2023-04-01

Semantic Segmentation（语义分割）：精准判断每个像素属于哪个类别

U-Net、转置卷积

Face verification vs. Face recognition:

* Face verification:1v1
* Face recognition:1vn

单样本学习 (One-shot learning): 输入一对数据，输出两者的差异
孪生神经网络 (Siamese network): 输入的图像输出为128维的向量，同一对象输出结果应尽可能相似

### Triplet Loss (三元组损失) :

Anchor, Positive, Negative
损失函数：$L(A,P,N)= max(\|f(A)-f(P)\|^{2}-\|f(A)-f(N)\|+\alpha,0)$

期望$\|f(A)-f(P)\|^{2}-\|f(A)-f(N)\|+\alpha\le0$

### 图像风格迁移

内容代价函数
风格代价函数
有些复杂，没有仔细听

### 循环神经网络Recurrent Nueral Networks

对后面的词进行预测时，会考虑前面词。前一个激活结果$a^i$作为$a^{i+1}$的输入

RNN容易出现梯度消失和梯度爆炸问题
梯度爆炸采用Gradient clipping

梯度消失：门控循环单元 (GRU)

# 2023-03-31

### 迁移学习

计算机视觉很有用

### 计算机视觉目标检测

输出值包括x，y，w，h

**目标检测**包含**目标识别**和**目标定位**

#### 基于滑动窗口的目标检测

用滑动窗口去截取目标对象，再进行卷积判断，存在计算量过大的问题，以及窗口大小

解决办法：将FC视为$1 \times1$核的卷积层，有多少个滑动窗口，最后输出就是几乘几

#### YOLO (You Only Look Once)

把画面切成$n\times n$个小窗口，对每个窗口进行 目标检测，采用滑动窗口卷积思路

并交比（Intersection over union）：$0\le\frac{相交面积}{相并面积}\le1 $
	"Correct" if $IoU \ge 0.8$

#### Non-max suppression Example (非极大值抑制)

目的：确保对一个目标只得到一个检测结果

流程：

1. 对每个框内的输出结果，输入网络，输出结果
2. 去除掉所有$p_c \lt 0.6$的框
3. 在剩余框中取p最高的一个确定为一个输出
4. 弃掉所有$IoU\gt0.5$的框（和输出预测结果有冲突的框）
5. 重复3，4直到没有剩下的框

#### Anchor box（锚框）

预设不同形状，输出多组形状，取Iou高的，可以识别一个框内出现两种锚框的情况

无法针对一个框内有多个不同对象，或有多个相同对象

# 2023-03-30

### 表白日，虽然不顺利

# 2023-03-29

### 卷积神经网络

1. 卷积层conv
2. 池化层pooling
3. 全连接fully-connected
4. softmax

### 残差网络

ResNet：$a{[l+2]}=g\left(z^{[l+2]}+a^{[l]}\right)$

### 初始网络

Inception network：可以混合1v1卷积、nvn卷卷积、池化堆叠在一起

### MobileNet

Depthwise Separable Convolution深度可分离卷积：Depthwise Convolution + Pointwise Convolution

Depthwise Convolution: 用$n \times n \times 1$的一个单通道卷积核复制拓展到多通道，去对输入$n \times n \times n_c$卷积，输出仍为$n \times n \times n_c$

Pointwise Convolution: 用$1 \times 1 \times n_c$的卷积核  

# 2023-03-21

异常分析（地铁上看的，速过）

终于开始计算机视觉了

# 2023-03-20

dropout

归一化输入

Mini-batch gradient descent

指数加权平均

指数加权平均的偏差修正

动量梯度下降

RMSprop

Adam optimization algorithm: 动量梯度下降+RMSprop

![image-20230320213928381](/Users/liyuanyong/files/temp/note/note.assets/image-20230320213928381.png)

Batch-Normalization: 批量归一化。对隐藏层进行均值归一化

soft-max

Tensorflow

![image-20230321021353821](/Users/liyuanyong/files/temp/note/note.assets/image-20230321021353821.png)




# 2023-03-17

### Activation function

1. sigmoid function
2. tanh function
3. Relu
4. leaky relu

### Dimension

```python
np.sum(dZ, axis=1, keepdims=True)
```

```python
X = (n0, m)
w1.shape = (n1, n0)
b1.shape = (n1, 0)

Z1 = w1 * X + b
Z1.shape = (n1, m)
A1.shape = (n1, m)
w2.shape = (n2, n1)
b2.shape = (n2, 0)

Z2 = w2 * A1 + b
Z2.shape = (n2, m)
A2.shape = (n2, m)
```

### HyperParameters
