# Python Learning

## 基础知识

### 转义字符

```python
# \n	new line
# \t     tab
# \r     return    print('111\r2') 输出结果为'211'
# \b    backward    print('111\b2') 输出结果为'112'
# \\\\     转义字符
# 在字符串前用r，字符串内转义字符不起作用    print(r'abc\nab') 输出结果'abc\nabc'
```

### 数据类型

输出数据类型

```python
val=1
print(type(val))
```

#### int

二进制		0b
八进制		0o
十六进制	0x

#### float

针对浮点数计算结果不精确，解决办法

```python
from decimal import Decimal
print(Decimal('1.1')+Decimal('2.2')) # 3.3
```

#### str

单引号和双引号在一行，三引号可以多行

#### 类型转化

```python
val1=1
val3=1.1
val4='11'
print(str(val1)) # 使用str()转换数据类型
print(int(val2)) # str(整数字符串),float(截取整数部分)
print(float(val1)) # int(1->1.0),str(仅数字)
```

### 注释

```python
# 单行注释
'''多行注释'''
```

### 运算符

```python
# 算数运算符
print(11//2) # 整除运算
print(11%2) # 取余运算
print(2**2) # 幂运算
# 赋值运算符
a=b=c=2 # a,b,c为同一地址
a**=b # 4
a,b,c=1,2,3 # 解包赋值
a,b=b,a # 交换操作
# 比较运算符
list1=[11,22,33,44]
list2=[11,22,33,44]
print(a==b) # True
print(a is b) # False
print(a is not b) # True
# 布尔运算符 and,or,not,in,not in
pirnt('w' not in 'hello world') # False
# 位运算符 转为二进制计算
print(4<<2) # 16 循环左移
print(4>>1) # 2
# 运算符优先级
# 幂运算 > 算数运算 > 位运算(循环移位>与>或) > 比较运算
```

---

## 数据类型

#### 列表List

创建方式：[ ] 或 list()

```python
L=list('a,b,c,d,e') # ['a','b','c','d','e']
# L[-1]为L的最后一个元素 L[-N]为L倒数第N个元素
print(L.index('b')) # 返回L中第一个b的索引值，没找到则会ValueError
print(L.index('b',1,3)) # 从L[1]到L[2]（小于3）中查找
```

获取列表多个元素：切片 lst[start : stop : step]

```python
lst1 = list('abcdef')
lst2 = lst1[1:4] # 赋值结果为['b','c',d']
lst3 = lst1[0:5:2] # 赋值结果为['a','c','e']
lst4 = lst1[::-1] # 赋值结果为['f','e','d','c','b','a']
lst5 = lst1[4:0:-1] #赋值结果为['e','d','c','b','a']]
```

```python
lst = [1,2,3,4]
if 2 in lst: pass
if 2 not in lst: pass
```

```python
lst = [1,2,3]
lst.append(4) # 在尾部添加一个元素
# lst.append([4.5]) 添加结果为[1,2,3,[4,5]]
# lst.extend([4,5]) 添加结果为[1,2,3,4,5]
lst.insert(1,100) # 在位置1插入100 结果为[1,100,2,3,4]
lst[1:]=[2,3] # 结果为[1,2,3]
```

```python
lst = [1,2,3,4,5,6,7,8,9]
lst.remove(1) # 删除列表中第一个1
lst.pop(1) # 删除索引为1的元素，没参数则默认最后一个元素
lst[0:4]=[] # 删除索引0到索引3的元素
lst[0:2]=[1,2] # 索引0到索引1的元素分别改为1，2
lst.clear() # 清空lst
```

```python
lst = [2,1,4,3,5]
lst.sort() # 升序（默认，等价于lst.sort(reverse=False)
lst.sort(reverse=True) # 降序
sorted(lst, reverse=True) # 使用方法和sort()一样，但返回类型为一个新list
```

```python
lst = [i*i for i in range(1,10)] # 输出1~9的每个数的平方
```




#### 字典dict

内部存储无序，存储位置要经过一次hash计算
key->value

```python
D = {1:0, 2:1, 3:2}
D2 = dict(name='jack',age=20)
```

```python
scores={'张三',100}
print(scores['张三'])
print(scores.get('张三')) # 若不存在，则返回None，而[]会报错
print(scores.get('不存在的人',99)) # 若查找不存在，则返回99
```

```python
scores={'张三':100}
print('张三' in scores)

del scores['张三']
# scores.clear()
```

```python
D = {1:0, 2:1, 3:2}
print(D.keys()) # dict_keys([1, 2, 3])
print(D.values()) # dict_values([0,1,2])
print(D.items()) # dict_items([(1, 0), (2, 1), (3, 2)])
```

```python
# 字典的遍历
D = {'a':1, 'b':2}
for item in scores:
	print(item, scores[item], scores.get(item))
```

```python
# 字典生成式
items = ['Fruits','Books','Others']
prices = [96,78,25]
D = {item:price for item, price in zip(items, prices)} # zip以短的可遍历对象长度为标准
```

列表、字典为可变序列，修改内容后id位置不变
元组、字符串为不可变序列，修改后id改变

#### 元组Tuple

不可修改的List

```python
tuple(list)或list(tuple)
t1 = ('a', 2, 'hello')
t2 = 'a', 2, 'hello'
t3 = ('abc') # 数据类型为str
t4 = ('abc',) # 数据类型为tuple
```

```python
t = (10, [20, 30], 9)
# 不允许修改元组的元素，但可以修改元组内的可变元素
t[1].append(10) # t = (10, [20, 30, 10], 9)
```

#### 集合set

没有value的字典

```python
s={1,2,3,4,5} # 自动去重，无序存储
# s={i for i in range(6)}
set(range(6)) # 赋值为0~5
set([1,2,3,4]) # list to set
set((1,2,3)) # tuple to set
print(type({})) # dict
print(type(set())) # set
```

```python
s = {1,2,3}
print(2 in s)
s.add(4)
s.update({5,6,7}) # 一次更新至少一个；参数可以是列表、元组、集合

s.remove(1) # 移除不存在元素时，报KyError异常
s.discard(300) # 不存在时不报错
s.pop() # 删除s的第一个元素，但因为s时无序存储，第一个元素随机，返回值为删除元素值
```

```python
s1={1,2,3,4,5}
s1 == {2,1,3,4,5} # True
{1}.issubset(s1) # True
{1,2,3,4,5,6}.issuperset(s1) # True
{2,6,7,8}.isdisjoint(s1) # 有交集为False，没交集为True
```

```python
s1 = {1,2,3,4}
s2 = {1,2,3,4,5}
s1.intersection(s2) # 取s1和s2的交集，等价于s1 & s2
s1.union(s2) # 取s1和s2的并集操作，等价于s1 | s2
s1.difference(s2) # 取差集操作，s1有s2没有，等价于s1 - s2
s1.symmetric_difference(s2) # 对称差集，等价于s1|s2 - s1&s2
```

#### 字符串

```python
# 驻留机制 仅保存一个不重复的字符串 具体处理过程较复杂

# 查询操作
str = 'hello, hello'
s.index('lo') # 3 没查到返回ValuesError
s.rindex('lo') # 10
s.find('lo') # 3 没查到返回-1
s.rfind('lo') # 10
```

```python
s = 'Hello'
# s.upper() s.lower()
# s.swapcase() 大写变小写 
# s.title() 首字母大写
print(s.center(9, '*')) # **Hello** 第二个参数默认是空格
print(s.ljust(9, '*')) # Hello****
# s.rjust(9) 
s.zfill(10) # 右对齐，左边用0填充 00000Hello
'-123'.zfill(6) # -00123
```

```python
s = 'Hello World'
print(s.split()) # 默认以空格作为分割符 ['Hello', 'World']
print('Hello|World|Python'.splist(sep='|', maxsplit=1)) # 使用sep来设置分割符，maxsplit设置最大分割数量 输出结果为 ['Hello', 'World|Python'] 
# 'Hello|World|Python'.rsplit() 和 split()使用方法一样，逆向
```

```python
s = 'hello, python'
s.isidentifier() # False
'\t'.isspace() # True ' '.isspace()也是True
'a'.isalpha() # True中文也算
'1'.isdecimal() # True
'123四'.isnumeric() # True 而isdecimal为false
'123abc'.isalnum() # True 是否仅有数字和字母组成
```

```python
s = 'Hello World'
print(s.replace('Hello','a')) # a World
print('aaaa'.replace('a','b',2)) # bbaa
print('|'.join(['a','b','c'])) # a|b|c 元组、列表、序列、字符串(当作序列)
# 小技巧，输出字符的原始数值
print(ord('a')) # 97
print(ord('b')) # 98
```

```python
'''
 == 比较的是value
 is 比较的是id是否相等
'''
# 字符串>, <, <=, >=, ==
```

```python
s = 'Hello World'
s[:5] + ',' + s[6:] # Hello,World
# [start, end, step]
s[::-1] # 倒置
```

```python
# 格式化字符串
'我的名字叫%s, 今年%d岁了' % ('Liyy', '22')
# 第一种
print('我叫{0}, 今年{1}岁了, 我叫{0}'.format('liyy', 22))
# 第二种
for i in range(10):
    print(f'{i}')
name = 'liyy'
age = 22
'我的名字叫{name}, 今年{age}岁了'
'''
%s	字符串
%d	数字
%f	浮点数
----------
print'%10.3f' % 3.1415926)
# 输出结果为 '     3.1415', 10表示宽度, '.3'表示精度
print('{0:.3}'.format(3.1415)) 结果为3.14, 表示3位数字
print('{0:.3f}'.format(3.1415)) 结果位3.141, 表示3位小数
'''
```



---

## 函数

#### 基本用法

```python
def fun1(a,b):
    return a+b

# 限制仅可关键字传参
def fun2(a, b, *, c, d) # c, d只能用关键词传参赋值
def fun3(a, b, *, **args)
def fun4(*args1, *, **args2)

fun1(10, 20)	# 位置传参
fun1(b=10, a=20)	# 关键字传参(不可修改)
'''
return 可以不写
return 多个值时返回一个元组()
'''
```

```python
# 不确定位置参数个数
def fun1(*args):
	print(args)
   	print(args[0])
# 不确定关键字参数个数
def fun2(**args):
    print(args)

fun1(1,2,3)
fun2(a=10, b=20, c=30)
fun2({'a':1, 'b':2, 'c':3})
```

```python
# 局部变量 转 全局变量
def fun():
    global a
    a = 1
print(a)
```



```python
if __name__=='__main__':
    main()
```

#### lambda

```python
f = lambda x: x*2+1
f(3) # 7
```

#### 内置函数

##### range()

range(stop)
range(start, stop)
range(start, stop, step)

##### bool()

##### for _ in

```python
for _ in range(5):
    pass # 不适用变量，且想循环五次
```



##### input()

```python
val=input('input tip') # input()参数为输入时的提示信息
# input()输入结果为一个str类型，需要进行类型转换
num=int(input('input a number'))
```

##### print()

```python
for _ in range(3):
    print('*', end='\t') # 不以'\t'作为输出结束
```



##### else少见用法

```python
for i in range(3):
    if i == 2: break
else:
    print('非正常结束')
```

##### id()

python内部list采用链式存储

```python
i = 10
print(id(i))
```

##### zip()

**zip()** 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。

```python
a = np.arange(4)
b = np.arange(4)
for (i,j) in zip(a,b)
```

##### map

**map()** 会根据提供的函数对指定序列做映射。

```python
map(lambda x: x ** 2, [1, 2, 3, 4, 5])  # 使用 lambda 匿名函数
```



## Bug调试

### Try Catch

```python
try:
    pass
except Exception1:
    pass
except Exception2 as e:
    print(e)
else:
    pass
finally: # 无论是否产生异常都会执行
```

### 常见Error

```python
'''
一些常见的异常
ZeroDivisionError	除零异常
IndexError	索引异常
KeyError	字典当中，没有关键字
NameError	变量未声明
SyntaxError	语法错误
ValueError	传入无效参数 例如: int('hello')
'''
```

```python
import traceback
try:
    print(1/0)
except:
    traceback.print_exc()
```

## 面向对象

```python
class Student: # 首字母大写
    id = '1234' # 类属性 类似于C++中的 静态变量
    def __init__(self, name, age):
        self.name = name
        self.age = age
        id = '2022111099'
	def learn(self): # 类内叫方法，类外叫函数，习惯在参数内写self
        pass
    @staticmethod
    def method():	# 静态方法，不能写self 调用方法： Student.method() 与类本身无关，只是用到了类的命名空间
        pass
    @classmethod
    def cm(cls):	# 类方法，写cls	调用方法： Student.cm()
        pass
stu1 = Student('Liyy', 22)
Student.learn(stu1) # 等价于stu1.learn()
```

__init__方法中定义的属性是 ，该类所有对象共有属性

### 封装

```python
# 不希望外部访问的私有属性，在前面加 __
clss Student:
    def __init__(self,name,age):
        self.name = name
        self.__age = age # 外部不能访问
	def show(self):
        print(self.name, self,__name)
stu = Student('liyy', 22)
print(stu._Student__age) # 也可以访问
```

### 继承

```python
# 支持多继承
# 子类中必须调用父类的构造函数
class Person(object):
    def __init__(self):
        pass
class Student(Person):
    def __init__(self):
        super().__init__()
        pass
class Primary_Student(Person, Student):
# 有点复杂，没太搞懂 TODO
```

```python
# object类是所有类的父类
# dir(stu):全属性加方法
# 例如
import math
print(dir(math))
```

### 特殊属性和方法

```python
# 接上面代码
stu = Student()
stu.__dict__ # 输出stu的属性字典
Student.__dict__ # 输出Student类的属性字典
stu.__class__ # 输出stu所属类
Student.__mro__ # 查看类的继承关系
Student.subclasses__ # 查看类的子类

a = 10
b = 20
c = a.__add__(b) # 等价于 a+b
class C1:
    def __init__(self, id):
        self.id = id
    def __add__(self, other): # 特殊方法
        return self.id + other.id
    def __len__(self):
        return len(self.id)
```

```python
# __new__() 用于创建对象
# __init__() 对创建的对象初始化
class Person(object):
    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        return obj
    def __init__(self, name, age):
        pass
p = Person() 
'''
Person() 调用__new__()，创建Person对象，返回值obj传给self，进行__init__()初始化，返回值赋值给p
'''
```

```python
# 变量赋值操作：两个变量指向同一内存地址
# 类的浅拷贝
import copy
p1 = Person()
p2 = copy.copy(p1) # p1和p2的内存地址不同
# 但p1和p2的内部变量内容指向的仍是一个id
# id(p1.name) is id(p2.name)
p3 = copy.deepcopy(p2) # p2和p3的内存地址不同，且包含所有对象的内存地址也不同
```

## 模块化编程

### 导入模块

```python
from math import pi # 仅导入模块中的一个对象
import math as m
```

想导入自己写的模块的时候，在pycharm里右键选择Mark Directory as -> Sources Root 就导入自定义模块

### 主程序 main

```python
if __name__ == '__main__':
# 在pycharm中输入main + 回车
```

主程序仅在运行所属模块时会运行，被import到其他包内时不运行

## Others

```python
# 查看当前python路径
import sys
print(sys.executable)
```

```python
# 查看列表字典长度
print(len(list))
```

```python
temp_data = 2
print(temp_data.isin([1,2,3])) #判断某一元素是否在一个列表内出现
```



## 编码

```python
#encoding=gbk
```

## 文件流

```python
file = open(filename[,mode,endcoding])
file.readlines()
```



# Pycharm

## 快捷键

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210719215529773.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl81NjE4NzAxNg==,size_16,color_FFFFFF,t_70)

# Python + Mysql

## 环境配置

pip install pymysql

```python
import pymysql
```

## 基本操作

```python
conn = pymysql.connect(host='127.0.0.1', post=3306, user='root', passwd='123456', db='liyy')
cursor = conn.cursor()
# cursor = conn.cursor(cursor=pymysql.cursors.DicCursor) 结果为字典
ret = conn.execute("select * from TEST")
print(cursor.fetchone()) # 每执行一次游标都会移动到下一行
print(cursor.fetchmany(3))
print(cursor.fetchall())
# cursor.scroll(-1, mode = 'relative') 向上滚动一行
# cursor.scroll(1, mode = 'relative') 向下滚动一行（相对移动）
# cursor.scroll(1, mode = 'aboslute') 移动到第一行（绝对位置）

conn.execute("insert into resource(id, name) values(%s, %s)", (12, 'liyy'))
```



# PIP

## 常用命令

# Anaconda

## 常用命令

### 环境操作

查看已创建环境: conda env list 或 conda info -e

创建环境: conda create -n env_name(name) python=3.7(python version)
启动环境: conda activate env_name(name)
退出环境: deactivate env_name
删除环境: conda remove -n env_name --all

### 库文件操作

查看已经安装的库文件: conda list -n env_name
安装某个版本库文件: conda install / pip install package_name(=version)
搜索某个包，查看包版本: conda search pack_name 、

### 安装库

```
清华：https://pypi.tuna.tsinghua.edu.cn/simple
中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/
华中理工大学：http://pypi.hustunique.com/
山东理工大学：http://pypi.sdutlinux.org/
豆瓣：http://pypi.douban.com/simple/
```

pip3 install 包名==版本 -i 镜像源url

# 库文件

## 系统模块

### sys

```python
import sys
print(sys.getsizeof(24))
```

### time

```python
import time
print(time.time())
print(time.localtime(time.time()))
```

### os

```python
# 提供了访问操作系统服务功能标准库
```

### urllib

```python
import urllib
print(urllib.request.urlopen('http://www.baidu.com'))
```



## os

```python
import os
os.path.abspath(__file__) # 当前文件绝对路径
os.path.basename(__file__) # 当前文件文件名
os.path.dirname(__file__) # 当前文件路径
os.path.dirname(os.path.dirname(__file__)) # 获得上上级文件路径
os.path.join(os.path.dirname(__file__), "test")

# 当前文件夹
'./'
os.path.dirname(__file__)

# 上一级文件夹
'../'
os.path.dirname(os.path.dirname(__file__))

# 上一级文件夹下的兄弟文件夹
'../other'
os.path.join(os.path.dirname(os.path.dirname(__file__)), 'other')
```

## logging

**日志级别等级排序**：critical > error > warning > info > debug

### 设置日志显示级别

```python
import logging

logging.basicConfig(level=logging.DEBUG)
```

### 日志信息写到文件

```python
import logging
logging.basicConfig(filename='os.getcwd'+'example.log', level=logging.DEBUG)
logging.debug('This message should go to the log file')
```

### 显示日期及更改显示消息格式

```python
import logging
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
```

```python
import logging
logging.basicConfig(format=%(levelname)s:%(message)s', level=logging.DEBUG)
```

