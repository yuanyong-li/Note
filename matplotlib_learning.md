# matplotlib.pylot

```python 
from matplotlib import pyplot as plt
```

$ 标注涉及到python实用语法

## 基础知识

```python
x = [1,2,3,4,5]
y = [2,4,5,6,2]
# 设置画布大小
plt.figure(figsize=(120,80), dpi=100)
# 画点
plt.plot(x,y)
# 显示
plt.show()
# 保存
plt.savefig("./temp.png")
# 保存为矢量图（放大不失真）
plt.savefig("./temp.svg")

# 设置x轴刻度
plt.xticks(x) #plt.xticks([i/2 for i in range(2,11)][::3]) 列表生成式+设置步长
# [random.randint(20,30) for i in range(10)] 生成10个20~30的随机数
```



## 实例

```python
# 将数据的最小值到最大值切分100份，用来画直线
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0,0] + (g[0,1] * x)

fig, ax = plt.subplots(figsize=(12,8))
# 可选kind = line[bar, scatter]
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
# 1右上 2左上 3左下 4右下
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()
```

