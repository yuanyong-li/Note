# matplotlib.pylot_learning

涉及python基础知识$

## matplotlib基本知识

```python
from matplotlib import pyplot as plt

x = range(2,26,2)
y = [i+1 for i in x] # $
# 设置图片大小
plt.figure(figsize=(20,8),dpi=80)
# 绘图
plt.plot(x,y)
# 展现图形
plt.show()
# 保存(svg格式为矢量图)
plt.savefig('./temp.png')
# 设置x,y轴刻度
plt.xticks(x)
plt.yticks(y)
# 设置非传统刻度
_xtick_labels = ['{}'.format(i) for i in range(2,26,2)] # $
# plt.xticks(x,_xtick_labels,rotation=90) 逆时针旋转90度

# 添加描述信息
plt.xlabel('x axis')
plt.ylabel('y axis')
plt.title('title of figure')

# 绘制网格
plt.grid(alpha=0.5) # 可选参数 不透明度alpha=[0,1]
```

```python
from matplotlib import pyplot as plt

x = range(2,26,2)
y1 = [i+1 for i in x]
y2 = [i-1 for i in x]
plt.plot(x,y1,
         label='y1',
		#color='r',[#F08080]
        #linestyle='--'
        #linewidth=5
)
plt.plot(x,y2,label='y2')

# 图例
plt.legend(loc=2) #loc=1[2,3,4] 
```

```python
# 散点图
from matplotlib import pyplot as plt
import random

x = [i for i in range(100)]
y1 = [random.randint(1,10) for i in range(100)]
y2 = [random.randint(10,20) for i in range(100)]
plt.figure(figsize=(20,8),dpi=80)

plt.scatter(x,y1)
plt.scatter(x,y2)

plt.show()
```

```python
# 条形图
from matplotlib import pyplot as plt
import random

bar_width = 0.3
x1 = [i for i in range(10)]
y1 = [random.randint(1,10) for i in range(10)]
x2 = [i+bar_width for i in x1]
y2 = [random.randint(10,20) for i in range(10)]
x3 = [i+2*bar_width for i in x1]
y3 = [random.randint(10,20) for i in range(10)]

plt.figure(figsize=(20,12),dpi=80)
plt.bar(x1,y1,width=bar_width,label='x1')
plt.bar(x2,y2,width=bar_width,label='x2')
plt.bar(x3,y3,width=bar_width,label='x3')
plt.xticks(x2,x1) # 让x轴刻度标在中间位置
plt.legend(loc=2)
plt.show()

# 横向条形图
plt.barh(x,y1,height=0.3)
```

```python
# 绘制直方图
from matplotlib import pyplot as plt
import random

data = [random.randint(1,100) for i in range(100)]
#计算组数
d = 5 #组距
nums_bins = (max(data)-min(data))//d
plt.figure(figsize=(12,8))
plt.hist(data,nums_bins) #d为一组长度，nums_bins为共多少组
plt.xticks(range(min(a),max(a)+d,d)
plt.grid()
plt.show()
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

```python
plt.figure(figsize=(12,8))
positive = data[data['Admitted'].isin([1])]
negative = data[data['Admitted'].isin([0])]
plt.scatter(positive['Exam1'],positive['Exam2'],marker='x',label='Admitted')
plt.scatter(negative['Exam1'],negative['Exam2'],marker='o',label='Not Admitted')
plt.show()
```



# numpy_learning

## 基础

```python
import numpy as np
import random

# numpy数组几种生成方式
print(np.array([1,2,3]))
print(np.array(range(3)))
print(np.arange(3))

print(np.array([1,2,3]).dtype) #内部数据的数据类型 
print(np.array([1,2,3],dtype='float32').dtype) #手动指定数据类型
temp_t = np.array([1,2,3],dtype='float32').astype('int8') #更改数据类型

temp_t = np.array([random.random() for i in range(10)])
print(np.round(temp_t,2))

print(np.array([[1,2,3],[4,5,6]]).shape) #(2,3)
print(np.arange(12).reshape(3,4).reshape(4,3).reshape(12,)) #更改形状 变成一维不是(12,1)而是(12,)
print(np.array([[1,2,3],[4,5,6]]).flatten())

# 运算
print(np.arange(3)+2) #[2,3,4] 可以理解为低纬度的扩展维度对应一一运算
print(np.arange(3)*np.array([[0,1,2],[0,1,2]]))
```

## Further

轴axis，对于shape(2,2,2)，对应的分别为0，1，2轴

```python
### numpy读数据
np.loadtxt(frame,dtype=np.float, delimiter=None, skiprows=0, usecols=None, unpack=False)
frame:文件、字符串或产生器
dtype:数据类型
delimiter:分割字符串，默认为空格
skiprows:跳过前x行
usecols:读取指定的列，索引，元组类型
unpack:若为True，读入属性写入不同数组变量(效果为转置)，False只写入一个数组变量
###
import numpy as np
file_path='./temp.csv'
print(np.loadtxt(file_path, delimiter=','))

# 转置
print(np.arange(10).reshape(5,2).T)
print(np.arange(10).reshape(5,2).transpose())
print(np.arange(10).reshape(5,2).swapaxes(1,0))
```

## numpy的索引和切片

```python
import numpy as np

temp = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(temp[[0,1],:]) #取多行
print(temp[:,[0,1]]) #取多列
print(temp[1,:]) #[4,5,6]
print(temp[[0,1],[0,1]]) #取temp[0][0]和temp[1][1]，列表相对应

temp[temp<4] = 0
print(np.where(temp<4,0,10))
print(temp.clip(3,5)) #小于3的赋值3，大于5的赋值5

# 数组的拼接
t1 = np.arange(10)
t2 = np.arange(10)
print(np.vstack((t1,t2))) #竖直拼接，竖直分割同理
print(np.hstack((t1,t2))) #水平拼接

print(np.zeros((2,2)).astype(int))
print(np.ones((2,2)).astype(int))
print(np.eye(4)) #对角线全为1，单位矩阵
print(np.argmax(temp,axis=0)) #axis为0上最大值的索引
print(np.argmin(temp,axis=1)) #axis为1上最小值的索引

np.random.seed(10) #输入一样的seed值时，随机生成结果相同
print(np.random.rand(3,3,3)) #随机生成维度为(3,3,3)的均匀分布浮点数矩阵0-1
print(np.random.randn(3,3,3)) #随机标准正态分布
print(np.random.randint(2,5,(3,3,3))) #随机生成(3,3,3)的2-5的随机整数
print(np.random.uniform(2,5,(3,3,3))) #随机生成(3,3,3)的均匀分布浮点数矩阵
print(np.sum(np.random.randint(2,5,(4,2)),axis=0)) #结果应为(1,2)矩阵

print(temp.sum(axis=0))
print(temp.mean(axis=1))
print(np.median(temp,axis=0)) #中值
print(temp.max(axis=0))
print(temp.min(axis=0))
print(np.ptp(temp,axis=1)) #极值
print(temp.std(axis=0)) #标准差
new_temp = temp.copy() #深拷贝，temp和new_temp互不影响
```

## numpy的nan和inf

**nan**：not a number

**inf**：infinity

**nan**性质：

 1. 两个nan不相等

    用处：np.count_nonzero(t!=t) 统计出nan个数

	2. 用np.isnan()判断nan

    用处：np.count_nonzero(np.isnan(t))

2. nan和任何值计算都为nan
## 一些常用函数

1. isin

   ```python
   a = np.array([
       [1,2,3],
       [4,5,6],
       [7,8,9]
   ])
   b = [3,4,5,6,7]
   print(np.isin(a,b))
   ```

## 实例

$$
J(\theta)=\frac{1}{m} \sum_{i=1}^{m}\left[-y^{(i)} \log \left(h_{\theta}\left(x^{(i)}\right)\right)-\left(1-y^{(i)}\right) \log \left(1-h_{\theta}\left(x^{(i)}\right)\right)\right]
$$



```python
def cost(theta, X, y):
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)
    first = np.multiply(-y, np.log(sigmoid(X * theta.T)))
    second = np.multiply((1 - y), np.log(1 - sigmoid(X * theta.T)))
    return np.sum(first - second) / (len(X))
```

# pandas_learning

```python
import numpy as np
import pandas as pd
```

$ 标注涉及到python实用语法

## 读数据

pd.read_csv

pd.read_excel

```python
fpath = './ex1data1.txt'
# 设置参数
data = pd.read_csv(
    fpath,
    sep=',',
    header=None,
    names=['x', 'y']
)
data.head()
```

## 数据结构DataFrame和Series

```python
# 三种创建series的方法
s1 = pd.Series([1,'a',5,2,7])
# 查询series的索引和值
# s1.index或s1.values
s2 = pd.Series([1,'a',5,2,7], index=['a','b','c','d'])
s3 = pd.Series({'a':1,'b':2,'c':3,'d':4})
s4 = s3[['a','b']] # 多个索引值返回结果也是一个Series，返回的是两列
s5 = s3.loc[1:4] # 按index返回
s6 = s3.iloc[1:4] # 按行索引返回
```

## Pandas使用loc查询数据

```python
# 1.使用单个label值查询数据
df.loc['2018-01-01', 'bWendu']
# 2.使用列表批量查询
df.loc['2018-01-01', ['bWendu','yWendu']]
# 3.使用数值区间进行查询
df.loc['2018-01-01', 'bWendu':'yWendu']
# 4.使用条件表达式（多个条件要加括号）
df.loc[(df['bWendu']<=30) & (df['yWendu']>=10), :]
# 5.调用函数查询(两种函数定义方式，涉及到python内容$)
df.loc[lambda df:df['bWendu']<=30 & df['yWendu']>=10, :]

def query_my_data(df):
    # 查询2018-09月所有aqiLevel为1的值。行索引为日期形式
    return df.index.str.startswith('2018-09') & df['aqiLevel']==1

df.loc[query_my_data, :]
```

## Pandas新增数据列

```python
# 1.直接赋值(字符串替换 + python变量类型转换)
df.loc[:, 'bWendu'] = df['bWendu'].str.replace('℃', '').astype(int32)
df.loc[:, 'wencha'] = df['bWendu']-df['yWendu'] #创建新列

# 2.apply方法
# Objects passed to the function are Series objects whose index is either the DataFrame's index(axis=0) or the DataFrame's columns(axis=1)
def get_wendu_type(x):
    if x['bWendu'] > 33: return '高温'
    if x['yWendu'] < -10: return ‘低温'
    return ’常温'
df.loc[:, 'wendu_type'] = df.apply(get_wendu_type, axis=1) #使用的是其他列不是index索引列，所以axis设为1
df['wendu_type'].value_counts() #统计

# 3.assign方法
# Returns a new object with all original columns in addition to new ones. 在原对象基础上添加新列，生成一个新的DataFrame对象
df.assign(yWendu_haushidu = lambda x: x['yWendu'] * 9 / 5 + 32)

# 4.按条件选择分组分别赋值(使用loc进行查询 + 赋值)
df['wencha_type'] = ''
df.loc[df['bWendu']-df['yWendu']>10, 'wencha_type'] = '温差大'
df.loc[df['bWendu']-df['yWendu']<=10, 'wencha_type'] = '温差小'
df.value_counts()
```

## Pandas数据统计函数

```python
# 1.一次获取所有数字列统计结果
df.describe()
df['bWendu'].mean()

# 2.唯一去重和按值计数
df['fengxiang'].unique()
df['fengxiang'].value_counts()

# 3.相关系数和协方差
df.cov() #协方差
df.corr() #相关系数
df['aqi'].corr(df['bWendu'])
df['aqi'].corr(df['bWendu']-'df['yWendu']')
```

## Pandas缺失值处理

包括检测、丢弃、填充三种处理

* isnull和notnull：检测是否是空值，可用于df和series
* dropna：丢弃、删除缺失值
  * axis：删除行或者列，{0 or 'index'. 1 or 'columns'}, default 0
  * how：如果等于any则任何值为空都删除，如果等于all则所有值为空才删除
  * inplace：如果为True则修改当前df，否则返回新df
* fillna：填充空值
  * value：用于填充的值，可以是单个值，或者字典（key是列名，value是值）
  * method：等于ffill则使用前一个不为空的值填充forword fill；等于bfill则使用后一个不为空的值填充backword fill
  * axis：按行还是按列填充，{0 or 'index', 1 or 'columns'}
  * inplace：如果为True则修改当前df，否则返回新df

```python
# 读取跳过前两行
studf = pd.read_excel('./student_excel.xlsx', skiprows=2)
# 删除全是空值的列
studf.dropna(axis='columns', how='all', inplace=True)
# 删除全是空值的行
studf.dropna(axis='index', how='all', inplace=True)
# 将分数列为空的填充为0分
studf.fillna({'分数':0}) #等价于studf.loc[:, '分数'] = studf['分数'].fillna(0)
# 使用前一个有效值填充
studf.loc[:, '姓名']=studf['姓名'].fillna(method='ffill')

studf.to_excel('./student_excel_clean.xlsx', index=False)
```



## Pandas的SettingWithCopyWarning

pandas的dataframe的修改写操作，只允许在源dataframe上进行

```python
condition = df['ymd'].str.startswith('2018-03')
# 错误代码：df[condition]['wen_cha'] = 1
# 上面语句等价于df.get(condition).set('wen_wencha')
# get后的值不确定为copy还是原dataframe
# 正确写法
df.loc[condition, 'wencha'] = 1

df_temp = df[condition].copy()
df_temp['wencha'] = df['bWendu']-df['yWendu']
```

## Pandas数据排序

```python
###
Series.sort_values(ascending=True, inplace=False)
* ascending: 默认为True升序，为False降序
* inplace: 是否修改原始Series
DataFrame.sort_alues(by, ascending=True, inplace=False)
* by: 字符串或者list<字符串>，单列排序或者多列排序
* ascending: bool或者List，List对应by的多列
* inplace: 是否修改原始Series
###
```

## Pandas字符串处理

1. 先获取Series的str属性，再调用函数
2. 只能在字符串列上使用，不能数字列上使用
3. Dataframe上没有str属性和处理方法
4. Series.str不是Python的原生字符串，而是独立

https://pandas.pydata.org/pandas-docs/stable/reference/series.html#string-handling

```python
# 1.使用str的startswith和contains
condition = df['ymd'].str.contains('2018')
df.loc[condition,:]

# 2.链式操作
df['yml'].str.replace('-','').str.slice(0,6) #slice是切片语法

# 3.正则表达式($python语法)
def get_nianyueri(x):
    year,month,day = x['ymd'].split('-')
    return f'{year}年{month}月{day}日'
df['中文日期'] = df.apply(get_nianyueri, axis=1)
# 去除年月日
df['中文日期'].str.replace('年','').str.replace('月','').str.replace('日','')
df['中文日期'].str.replace(['年月日'],'')
```

## Pandas的axis参数

单行处理：axis=0代表行，axis=1代表列

多行处理：axis=0代表跨行（逐行遍历），axis=1代表跨列

```python
# 3×4的矩阵Matrix
Matrix.mean(axis=0) #得到的结果为1×4
Matrix.mean(axis=1) #得到的结果为3×1
```

## Pandas的索引index

1. 方便查询
2. 提升性能
   1. index若为唯一：使用哈希表存储，时间复杂度为O(1)
   2. index若非唯一但有序，使用二分查找，O(logN)
   3. 完全随机，O(N)
3. 数据对齐
4. 更多更强大数据结构支持

```python
# 1.方便查询
# drop=False，让索引列还保持在column中，默认是会删除
# inplace=True，让函数返回结果替代原df
df.set_index('userId', inplace=True, drop=False)
df.loc[500].head(5)
# 等价于
df.loc[df['userId']==500].head(5)

# 2.提升性能

```

## Pandas数据的Merge

pd.merge(left,right,how='inner',on=None,left_on=None,right_on=None,left_index=False,right_index=False,sort=True,suffixes=('_x','_y'),copy=True,indicator=False,valiade=None)

* left,right：要merge的df或series
* how：join的类型，'left','right','outer','inner'
* on：join的key，left和right都需要有这个key
* left_on：left的df或者series的key
* right_on：right的df或者series的key
* left_index,right_index：使用index而不是普通的column做join
* suffixes：非key字段列有重名，自动添加后缀，默认是('_x','_y')

不同join区别：

1. Left Join：左边都会出现在结果中，右边只保留key重叠的行
2. Right Join：和Left Join同理相反
3. Inner Join：左边右边都有打的才保留
4. Outer Join：取最大并集

## Pandas是实现数据合并concat

concat语法：pandas.concat(objs, axis=0, join='outer', ignore_index=False)

* objs：一个列表，可以是DataFrame或者Series混合
* axis：默认是0代表按行合并，1代表按列合并
* join：合并时候索引的对齐方式，默认outer join，也可以是inner join（例如左有col1，右没有，结果就没有col1列）
* ignore_index：是否忽略掉原来的数据索引（自动建立新索引）

append语法：DataFrame.append(other, ignore_index=False)
只能按行合并，没有按列合并

* other：单个dataframe、series或者列表
* ignore_index：~

```python
# $列表生成式[i*3 for i in range(5)]
pd.concat(
	[pd.DataFrame([i], columns=['A']) for i in range(5)],
    ignore_index=True
)
```

## Pandas批量拆分Excel与合并Excel

将一个source excel文件拆分

```python
work_dir='./datas'
# $格式化字符串
splits_dir=f'{work_dir}/splits'

import os
if not os.path.exists(splits_dir): os.mkdir(splits_dir)
    
df_source = pd.read_excel(f'{work_dir}/data_source.xlsx')
total_count = df_source.shape[0]

# 使用iloc数字索引拆分
user_names = ['A', 'B', 'C', 'D']
split_size = total_count // user_names # 假设可以整除

df_subs = []
# $enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
for idx, user_name in enumerate(user_names):
    begin = idx * split_size
    end = begin + split_size
    df_sub = df_source.iloc[begin:end]
    df_subs.append((idx, user_name, df_sub))
for idx, user_name, df_sub in  df_subs:
	file_name = f'{splits_dir}/data_{idx}_{user_name}.xlsx'
    df.sub.to_excel(file_name, index=False)
```

将多个合并

```python
......
import os
excel_names = []
# $listdir返回文件夹下所有文件名的列表
for excel_name in os.listdir(splits_dir)
	excel_names.append(excel_name)
df_list = []
for excel_name in excel_names:
    excel_path = f'{splits_dir}/{excel_name}'
    df_split = pd.read_excel(excel_path)
    username = excel_name.replace(f'{splits_dir}/data_','').replace('.xlsx','')[2:]
    # 添加用户名新列
    df_split['username'] = username
    
    df_list.append(df_split)
df_merged = pd.concat(df_list)
df_merged['username'].value_counts()
df_merged.to_excel(f'{work_dir}/data_merged.xlsx', index=False)
```

## Pandas实现groupby分组统计

```python
# 对A列分组统计，分别计算每组的各列sum
df.groupby('A').sum()
df.groupby(['A','B']).mean() #结果中'A'和'B'成为了索引
df.groupby(['A','B'], as_index=False).mean()
# 同时查看多种数据统计
df.groupby('A').agg([np.sum, np.mean, np.std])
# 对不同的列获得不同的数据统计
df.groupby('A').agg([col1:np.sum, col2:np.mean, col3:np.std])
# 获得某个单独分组
df.get_group('A')

# 实例：查看每个月最高温度
data = df.groupby('month')['bWendu'].max()
data.plot()	

data = df.groupby('month').agg({'bWendu':np.max, 'yWendu':np.min, 'aqi':np.mean})
data.plot()
```

## Pandas.DataFrame.plot

Parameters:

 * data: Series or DataFrame

 * x: label or position, default None

 * y: label, position or list of label, positions, default None

 * kind: The kind of plot to produce

    * **line**: default
    * **bar**: vertical bar plot
    * **scatter**: scatter plot (DataFrame only)
   * hexbin: hexbin plot (DataFrame only)
    * barh: horizontal bar plot
    * hist: histogram
    * box: boxplot????
    * kde: Kernel Density Estimation plot

   * area: area plot
   * pie: pie plot

* figsize: a tuple (width, height) in inches. Size of a figure object

```python
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
plt.show()
```

## Pandas的一些实例

```python
# recommendation_predict

# 使用到groupby和transform，计算每个标签一段时间内的百分比
df = pandas.DataFrame(data, columns=['value', 'date', 'label_type'])
df['percent'] = df.groupby(['date'])['value'].transform(lambda x: round(x / sum(x) * 100, 4))
df['change_percent'] = df.groupby(['label_type'])['percent'].transform(lambda x: round((x - np.mean(x)) * 100 / np.mean(x), 4))
```

