## Logistic Regression

逻辑回归是分类算法

Sigmoid function:
$$
h_{\theta}(x)=\frac{1}{1+e^{-\theta^{\top} x}}
$$


decision boundary

Cost Function：
$$
J\left( \theta  \right)=\frac{1}{m}\sum\limits_{i=1}^{m}{{Cost}\left( {h_\theta}\left( {x}^{\left( i \right)} \right),{y}^{\left( i \right)} \right)}
$$

$$
\operatorname{Cost}\left(h_{\theta}(x), y\right)=\left\{\begin{aligned}
-\log \left(h_{\theta}(x)\right) & \text { if } y=1 \\
-\log \left(1-h_{\theta}(x)\right) & \text { if } y=0
\end{aligned}\right.
$$
Gradient Descent:
$$
\theta_j := \theta_j - \alpha \frac{1}{m}\sum\limits_{i=1}^{m}{{\left( {h_\theta}\left( \mathop{x}^{\left( i \right)} \right)-\mathop{y}^{\left( i \right)} \right)}}\mathop{x}_{j}^{(i)}
$$


## 配套笔记对应内容

week1: 单变量线性回归、线性代数回顾

week2: 多变量线性回归、Octave教程

week3: 逻辑回归、正则化

