# tmux

`tmux`: 新建session
`tmux new -s <session-name>`

`tmux detach`: 离开session

`tmux ls`: 查看session列表
`ctrl + b s`

`tmux attach -t <session-name>`: 进入session

`tmux kill-session -t <session-name>`: 关闭session
`ctrl+b d`: 关闭当前session

`tmux switch -t <session-name>`: 切换session

`tmux rename-session -t <old-session-name> <new-session-name>`: 重命名session

`tmux kill-session -t session_name`

# Anaconda

`conda list`

`conda create -n env_name python=x.x`

`conda remove -n env_name --all`: 删除虚拟环境

`conda install -n env_name package_name`

`conda env list`

## 配置远程连接

`jupyter notebook password`

`jupyter notebook --generate-config`

`vim ～/.jupyter/jupyter_notebook_config.py`

```python
c.NotebookApp.open_browser = False#不需要自动打开火狐浏览器
c.NotebookApp.port = 8890 #需要改
c.NotebookApp.ip = '10.99.4.2' #需要改
c.NotebookApp.notebook_dir = '/home/liyuanyong2022'
#设定默认打开的目录。
c.NotebookApp.allow_remote_access = True
#如今需要你之前设置的密码登录，填入下面的即可免密登录。(需要改)
c.NotebookApp.token = 'argon2:$argon2id$v=19$m=10240,t=10,p=8$/s9zNT3+VpBX0M1aDwn+6g$jEYL//Lkmq2c0B5VmpwWL8ftxisfK001/GUBVJbIvdw'
```

`jupyter notebook &`

### 已有的session

```
大模型演示平台 http://10.99.4.2:8544 密码为空
jupyter https://10.99.4.2:8890 密码为空
```



# Ubuntu Learning

## 基本操作命令

* 查看文件权限：

    * `ls -l filename`
    * `ls -ld folder`

* 更改文件权限：

    * `chmod u+x filename`

        u 代表所有者（user）

        g 代表所有者所在的组群（group）

        o 代表其他人，但不是u和g （other）

        a 代表全部的人，也就是包括u，g和o

        r w x 

        `sudo chmod 777` ：所有者、群组、其他用户都全权限

* 下载命令：

    * `wget url`

* 解压命令：

    - `tar -zxvf filename.tar.gz`

### nohup

```
nohup /home/scandb/code/predict/label_predict_main.py > temp.log 2>&1 &
# 将标准错误 2 重定向到标准输出 &1 ，标准输出 &1 再被重定向输入到 runoob.log 文件中
# https://www.runoob.com/linux/linux-comm-nohup.html 
nohup /home/scandb/anaconda3/envs/pytorch/bin/python3.7 /home/scandb/code/predict/label_predict_main.py
ps -aux | grep "label_predict_main.py"
kill -9
```

# github learning

## git配置

`git config --global user.name "Your Name"`
`git config --global user.email "email@example.com"`

* `git config`命令的`--global`参数，表明这台机器上的所有Git仓库都会使用这个配置，也可以对某个仓库指定不同的用户名和邮箱地址。

`git config --list`

* 查询配置

## 基本操作

### 1. 连接仓库

`git init`

* You've just told your computer that you want git to watch the `myProject` folder and to keep track of any changes. This also allows us to run git commands inside of the folder. (Warning: Be very careful to make sure you're in the right directory when you run `git init`!)

`git remote add origin [Repository URL goes here]`

* Basically, we tell our computer "Hey, I created this repo on GitHub, so when I push, I want my code to go to this GitHub repo." Now whenever you run `git push origin master` your computer knows that origin is pointing to your repo you made on GitHub and it pushes your changes there.

    ( If you accidentally DID initialize your repository with a README, you must do a `git pull origin master` first - to get the README file on your computer - before you'll be able to push. )

* *failed to push some refs to 'temp.txt'*:

    * `git remote -v`
    * `git remote remove [前一条查询的结果] `

### 2. 文件上传

`git add nameOfMyFile.fileExtension`

* This adds our file(s) to the 'staging area'. This is basically a fail safe if you accidentially add something you don't want. You can view items that our staged by running `git status`.
* `git add -A`表示添加所有内容
* `git add .` 表示添加新文件和编辑过的文件不包括删除的文件
* `git add -u` 表示添加编辑或者删除的文件，不包括新添加的文件。

`git commit -m "the sentence I want associated with this commit message"`

* This tells your computer: 'Hey, the next time code is pushed to GitHub, take all of this code with it.' The message also specifies what GitHub will display in relation to this commit.

`git push origin main`

* Your code is now pushed to GitHub. Be sure to include `origin master`, as this tells GitHub which branch you want to push to, and creates the branch if it doesn't exist yet.

* `git push -f origin master`: 强制上传

### 3. fork and clone

`git clone [url]`
`git pull`

* This takes what's on GitHub and essentially downloads it so you can now make changes to it on your local computer.

### 4. 分支

`git branch`

* 查看分支
* `-r`: 查看远程分支
* `-a`: 查看所有分支

`git branch <name>`

* 创建分支

`git checkout <name>`

* 切换分支

`git checkout -b <name>`

* 创建+切换分支
* `git checkout -b branch-name origin/branch-name`: 同时建立关联

`git merge <name>`

* 合并某分支到当前分支

`git branch -d <name>`

* 删除分支
* `git branch -D <name>`: 强制删除

`git log --graph`

* 查看分支合并图
* `git log --graph --pretty=oneline --abbrev-commit`: 简版分支树

`git branch --set-upstram branch-name origin/branch-name`

* 设置为跟踪来自origin的远程分支branch-name
* 使用`--track`或`--set-upstram-to`

### 5. log

`git log`

* `git log --pretty=oneline`: 简化版日志
* `git log -n`: 查看前n条
* `git log --stat -n`: 查看前n条变更日志

`git show <commit-hash-id>`

* 查看某次commit做了哪些修改

### 6. 版本管理

`git show`

* 查看本次修改内容

`git checkout -- <file>`

* 回到最近一次git commit前状态
    * 一种是file自修改后还没有被放到暂存区，现在，撤销修改就回到和版本库一模一样的状态；
    * 一种是file已经添加到暂存区后，又作了修改，现在，撤销修改就回到添加到暂存区后的状态。

`git reset HEAD <file>`

* 该命令将已经add到暂存区的修改撤销掉（unstage），重新放回工作区

`git rm <file>`

* 等同于`rm <file>`和`git add <file>`
* `rm <file>`可以`git checkout -- <file>`
* `git rm <file>`可以`git reset HEAD <file>`以及`git checkout -- <file>`
* `git rm -r --cached file`: 删除暂存区内某个文件，再push

`git status`

* This will show what files have been changed. This also helps us determine what files we want to add to GitHub and what files we don't want to add to GitHub. 
* 查看工作区状态

`git diff`

* This will show the actual code that has been changed. Again, we want to make sure we don't push anything to GitHub that shouldn't be there.
* Your code is now pushed to GitHub. Be sure to include `origin master`, as this tells GitHub which branch you want to push to, and creates the branch if it doesn't exist yet. 
* 查看工作区（work dict）和暂存区（stage）的区别
* `git diff --cached`查看暂存区（stage）和分支（master）的区别
* `git diff HEAD -- <file>`查看工作区和版本库里面最新版本的区别

`git reset --hard commit_id`

* 回退到指定版本号

### 7. 小操作

`git ls-files`:

* 查看暂存区内的文件
* `git ls-files -c`: 默认参数，-cached
* `-deleted(-d)`
* `-modified(-m)`
* `-other(-o)`

## 小知识

- 工作区：在电脑里能看到的目录；
- 版本库：在工作区有一个隐藏目录`.git`，是Git的版本库。
- `git add`实际上是把文件添加到暂存区
- `git commit`实际上是把暂存区的所有内容提交到当前分支
- 手动创建 .gitignore文件，里面写要忽略的文件

# Streamlit

## 安装

`conda create -n stenv python=3.9`

`pip install streamlit`
`streamlit run demo.py`
`streamlit run web_demo2.py --server.port 6008`
`python -m streamlit run your_script.py`

`streamlit run https://raw.githubusercontent.com/streamlit/demo-uber-nyc-pickups/master/streamlit_app.py`

## 常用方法

```
st.set_page_config(layout="wide")
```

### display

```python
import streamlit as st
import pandas as pd
import numpy as np
st.write()
st.dataframe()
st.table()
st.line_chart()
st.map()
```

### Wigets

```python
import streamlit as st
import pandas as pd
import numpy as np
# Widgets

# slider
st.write(st.slider('x', key='x'), 'squared is', pow(st.session_state.x,2)) # 用key访问

# text_input
st.text_input('your name', key='name')
st.session_state.name

# checkbox
if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])
    chart_data

# selectbox
df = pd.DataFrame({
    'first column': [1, 2, 3, 4],
    'second column': [10, 20, 30, 40]
    })
option = st.selectbox(
    'Which number do you like best?',
     df['first column'])
'You selected: ', option
```

### Layout

```python
import streamlit as st
import pandas as pd
import numpy as np
# Layout

# Add a selectbox to the sidebar:
add_selectbox = st.sidebar.selectbox(
    'How would you like to be contacted?',
    ('Email', 'Home phone', 'Mobile phone')
)

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'Select a range of values',
    0.0, 100.0, (25.0, 75.0)
)
```

```python
# 显示进度
import streamlit as st
import time

'Starting a long computation...'

# Add a placeholder
latest_iteration = st.empty()
bar = st.progress(0)

for i in range(100):
  # Update the progress bar with each iteration.
  latest_iteration.text(f'Iteration {i+1}')
  bar.progress(i + 1)
  time.sleep(0.1)

'...and now we\'re done!'
```

### Caching

为避免大型变量函数重复加载，使用`@st.cache_data`或者`@st.cache_resource`。一般读数据或者常规函数用前者，读取模型用后者

```python
DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')
@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state.text("Done! (using st.cache_data)")
```

### Pages

多页面：在主py的同路径下建立pages，里面文件会在左侧显示

## 实例

`st.title`

`st.header`

`st.subheader`

### Day3 button

```python
import streamlit as st

st.header('st.button')

if st.button('Say hello'):
     st.write('Why hello there')
else:
     st.write('Goodbye')
```

### Day5 write

```python
import numpy as np
import altair as alt
import pandas as pd
import streamlit as st

st.header('st.write')

# Example 1

st.write('Hello, *World!* :sunglasses:')

# Example 2

st.write(1234)

# Example 3

df = pd.DataFrame({
     'first column': [1, 2, 3, 4],
     'second column': [10, 20, 30, 40]
     })
st.write(df)

# Example 4

st.write('Below is a DataFrame:', df, 'Above is a dataframe.')

# Example 5

df2 = pd.DataFrame(
     np.random.randn(200, 3),
     columns=['a', 'b', 'c'])
c = alt.Chart(df2).mark_circle().encode(
     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
st.write(c)
```

### Day8 slider

```python
import streamlit as st
from datetime import time, datetime

st.header('st.slider')

# Example 1

st.subheader('Slider')

age = st.slider('How old are you?', 0, 130, 25)
st.write("I'm ", age, 'years old')

# Example 2

st.subheader('Range slider')

values = st.slider(
     'Select a range of values',
     0.0, 100.0, (25.0, 75.0))
st.write('Values:', values)

# Example 3

st.subheader('Range time slider')

appointment = st.slider(
     "Schedule your appointment:",
     value=(time(11, 30), time(12, 45)))
st.write("You're scheduled for:", appointment)

# Example 4

st.subheader('Datetime slider')

start_time = st.slider(
     "When do you start?",
     value=datetime(2020, 1, 1, 9, 30),
     format="MM/DD/YY - hh:mm")
st.write("Start time:", start_time)
```

### Day9 line_chart

```python
import streamlit as st
import pandas as pd
import numpy as np

st.header('Line chart')

chat_data = pd.DataFrame(
	np.random.randn(20, 3),
    columns=['a', 'b', 'c']
)

st.line_chart(chart_data)
```

### Day10 selectbox

```python
import streamlit as st

st.header('st.selectbox')

option = st.selectbox(
     'What is your favorite color?',
     ('Blue', 'Red', 'Green'), index=2)
# 默认选择index=2项
st.write('Your favorite color is ', option)
```

index选择默认值
disabled为True则无法选择
label_visibility选择label是否显示

```python
import streamlit as st

# Store the initial value of widgets in session state
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

col1, col2 = st.columns(2)

with col1:
    st.checkbox("Disable selectbox widget", key="disabled")
    st.radio(
        "Set selectbox label visibility 👉",
        key="visibility",
        options=["visible", "hidden", "collapsed"],
    )

with col2:
    option = st.selectbox(
        "How would you like to be contacted?",
        ("Email", "Home phone", "Mobile phone"),
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
    )
```

### Day11 multiselect

```python
import streamlit as st

st.header('st.multiselect')

options = st.multiselect(
     'What are your favorite colors',
     ['Green', 'Yellow', 'Red', 'Blue'],
     ['Yellow', 'Red'])

st.write('You selected:', options)
```

### Day12 checkbox

```python
import streamlit as st

st.header('st.checkbox')

st.write ('What would you like to order?')

icecream = st.checkbox('Ice cream')
coffee = st.checkbox('Coffee')
cola = st.checkbox('Cola')

if icecream:
     st.write("Great! Here's some more 🍦")

if coffee: 
     st.write("Okay, here's some coffee ☕")

if cola:
     st.write("Here you go 🥤")
```

### Day13 latex

```python
import streamlit as st

st.header('st.latex')

st.latex(r'''
     a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
     \sum_{k=0}^{n-1} ar^k =
     a \left(\frac{1-r^{n}}{1-r}\right)
     ''')
```

### Day17 secrets

### Day18 file_uploader

```python
import streamlit as st
import pandas as pd

st.title('st.file_uploader')

st.subheader('Input CSV')
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
  st.subheader('DataFrame')
  st.write(df)
  st.subheader('Descriptive Statistics')
  st.write(df.describe())
else:
  st.info('☝️ Upload a CSV file')
```

### Day19 layout

1.   `st.expander`
2.   `st.sidebar`
3.   `st.columns()`

```python
import streamlit as st

st.set_page_config(layout="wide")

st.title('How to layout your Streamlit app')

with st.expander('About this app'):
  st.write('This app shows the various ways on how you can layout your Streamlit app.')
  st.image('https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.png', width=250)

st.sidebar.header('Input')
user_name = st.sidebar.text_input('What is your name?')
user_emoji = st.sidebar.selectbox('Choose an emoji', ['', '😄', '😆', '😊', '😍', '😴', '😕', '😱'])
user_food = st.sidebar.selectbox('What is your favorite food?', ['', 'Tom Yum Kung', 'Burrito', 'Lasagna', 'Hamburger', 'Pizza'])

st.header('Output')

col1, col2, col3 = st.columns(3)

with col1:
  if user_name != '':
    st.write(f'👋 Hello {user_name}!')
  else:
    st.write('👈  Please enter your **name**!')

with col2:
  if user_emoji != '':
    st.write(f'{user_emoji} is your favorite **emoji**!')
  else:
    st.write('👈 Please choose an **emoji**!')

with col3:
  if user_food != '':
    st.write(f'🍴 **{user_food}** is your favorite **food**!')
  else:
    st.write('👈 Please choose your favorite **food**!')
```

### Day21 progress

进度条

```python
import streamlit as st
import time

st.title('st.progress')

with st.expander('About this app'):
     st.write('You can now display the progress of your calculations in a Streamlit app with the `st.progress` command.')

my_bar = st.progress(0)

for percent_complete in range(100):
     time.sleep(0.05)
     my_bar.progress(percent_complete + 1)

st.balloons()
```

### Day22 form

`st.select_slider`

```python
import streamlit as st

st.title('st.form')

# Full example of using the with notation
st.header('1. Example of using `with` notation')
st.subheader('Coffee machine')

with st.form('my_form'):
    st.subheader('**Order your coffee**')

    # Input widgets
    coffee_bean_val = st.selectbox('Coffee bean', ['Arabica', 'Robusta'])
    coffee_roast_val = st.selectbox('Coffee roast', ['Light', 'Medium', 'Dark'])
    brewing_val = st.selectbox('Brewing method', ['Aeropress', 'Drip', 'French press', 'Moka pot', 'Siphon'])
    serving_type_val = st.selectbox('Serving format', ['Hot', 'Iced', 'Frappe'])
    milk_val = st.select_slider('Milk intensity', ['None', 'Low', 'Medium', 'High'])
    owncup_val = st.checkbox('Bring own cup')

    # Every form must have a submit button
    submitted = st.form_submit_button('Submit')

if submitted:
    st.markdown(f'''
        ☕ You have ordered:
        - Coffee bean: `{coffee_bean_val}`
        - Coffee roast: `{coffee_roast_val}`
        - Brewing: `{brewing_val}`
        - Serving type: `{serving_type_val}`
        - Milk: `{milk_val}`
        - Bring own cup: `{owncup_val}`
        ''')
else:
    st.write('☝️ Place your order!')


# Short example of using an object notation
st.header('2. Example of object notation')

form = st.form('my_form_2')
selected_val = form.slider('Select a value')
form.form_submit_button('Submit')

st.write('Selected value: ', selected_val)
```

### Day23 experimental_get_query_params

```python
import streamlit as st

st.title('st.experimental_get_query_params')

with st.expander('About this app'):
  st.write("`st.experimental_get_query_params` allows the retrieval of query parameters directly from the URL of the user's browser.")

# 1. Instructions
st.header('1. Instructions')
st.markdown('''
In the above URL bar of your internet browser, append the following:
`?firstname=Jack&surname=Beanstalk`
after the base URL `http://share.streamlit.io/dataprofessor/st.experimental_get_query_params/`
such that it becomes 
`http://share.streamlit.io/dataprofessor/st.experimental_get_query_params/?firstname=Jack&surname=Beanstalk`
''')


# 2. Contents of st.experimental_get_query_params
st.header('2. Contents of st.experimental_get_query_params')
st.write(st.experimental_get_query_params())


# 3. Retrieving and displaying information from the URL
st.header('3. Retrieving and displaying information from the URL')

firstname = st.experimental_get_query_params()['firstname'][0]
surname = st.experimental_get_query_params()['surname'][0]

st.write(f'Hello **{firstname} {surname}**, how are you?')
```

### Day24 cache

https://docs.streamlit.io/library/advanced-features/caching

When you mark a function with the @st.cache decorator, it tells Streamlit that whenever the function is called it needs to check a few things:

1.  The input parameters that you called the function with
2.  The value of any external variable used in the function
3.  The body of the function
4.  The body of any function used inside the cached function

As mentioned, there are two caching decorators:

-   `st.cache_data` is the recommended way to cache computations that return data: loading a DataFrame from CSV, transforming a NumPy array, querying an API, or any other function that returns a serializable data object (str, int, float, DataFrame, array, list, …). It creates a new copy of the data at each function call, making it safe against [mutations and race conditions](https://docs.streamlit.io/library/advanced-features/caching#mutation-and-concurrency-issues). The behavior of `st.cache_data` is what you want in most cases – so if you're unsure, start with `st.cache_data` and see if it works!

-   `st.cache_resource` is the recommended way to cache global resources like ML models or database connections – unserializable objects that you don’t want to load multiple times. Using it, you can share these resources across all reruns and sessions of an app without copying or duplication. Note that any mutations to the cached return value directly mutate the object in the cache (more details below).

-   ![Streamlit's two caching decorators and their use cases. Use st.cache_data for anything you'd store in a database. Use st.cache_resource for anything you can't store in a database, like a connection to a database or a machine learning model.](https://docs.streamlit.io/images/caching-high-level-diagram.png)

    Streamlit's two caching decorators and their use cases.

```python
import streamlit as st
import numpy as np
import pandas as pd
from time import time

st.title('st.cache')

# Using cache
a0 = time()
st.subheader('Using st.cache')

@st.cache_data(suppress_st_warning=True)
def load_data_a():
  df = pd.DataFrame(
    np.random.rand(2000000, 5),
    columns=['a', 'b', 'c', 'd', 'e']
  )
  return df

st.write(load_data_a())
a1 = time()
st.info(a1-a0)


# Not using cache
b0 = time()
st.subheader('Not using st.cache')

def load_data_b():
  df = pd.DataFrame(
    np.random.rand(2000000, 5),
    columns=['a', 'b', 'c', 'd', 'e']
  )
  return df

st.write(load_data_b())
b1 = time()
st.info(b1-b0)
```

### Day 25 session_state

```python
import streamlit as st

st.title('st.session_state')

def lbs_to_kg():
  st.session_state.kg = st.session_state.lbs/2.2046
def kg_to_lbs():
  st.session_state.lbs = st.session_state.kg*2.2046

st.header('Input')
col1, spacer, col2 = st.columns([2,1,2])
with col1:
  pounds = st.number_input("Pounds:", key = "lbs", on_change = lbs_to_kg)
with col2:
  kilogram = st.number_input("Kilograms:", key = "kg", on_change = kg_to_lbs)

st.header('Output')
st.write("st.session_state object:", st.session_state)
```

### Day26 request

```python
import streamlit as st
import requests

st.title('🏀 Bored API app')

st.sidebar.header('Input')
selected_type = st.sidebar.selectbox('Select an activity type', ["education", "recreational", "social", "diy", "charity", "cooking", "relaxation", "music", "busywork"])

suggested_activity_url = f'http://www.boredapi.com/api/activity?type={selected_type}'
json_data = requests.get(suggested_activity_url)
suggested_activity = json_data.json()

c1, c2 = st.columns(2)
with c1:
  with st.expander('About this app'):
    st.write('Are you bored? The **Bored API app** provides suggestions on activities that you can do when you are bored. This app is powered by the Bored API.')
with c2:
  with st.expander('JSON data'):
    st.write(suggested_activity)

st.header('Suggested activity')
st.info(suggested_activity['activity'])

col1, col2, col3 = st.columns(3)
with col1:
  st.metric(label='Number of Participants', value=suggested_activity['participants'], delta='')
with col2:
  st.metric(label='Type of Activity', value=suggested_activity['type'].capitalize(), delta='')
with col3:
  st.metric(label='Price', value=suggested_activity['price'], delta='')
```

### Day27 数据可视化Nivo

`pip install streamlit-elements==0.1.*`

`pip install pathlib`

```python
# First, we will need the following imports for our application.

import json
import streamlit as st
from pathlib import Path

# As for Streamlit Elements, we will need all these objects.
# All available objects and there usage are listed there: https://github.com/okld/streamlit-elements#getting-started

from streamlit_elements import elements, dashboard, mui, editor, media, lazy, sync, nivo

# Change page layout to make the dashboard take the whole page.

st.set_page_config(layout="wide")

with st.sidebar:
    st.title("🗓️ #30DaysOfStreamlit")
    st.header("Day 27 - Streamlit Elements")
    st.write("Build a draggable and resizable dashboard with Streamlit Elements.")
    st.write("---")

    # Define URL for media player.
    media_url = st.text_input("Media URL", value="https://www.youtube.com/watch?v=vIQQR_yq-8I")

# Initialize default data for code editor and chart.
#
# For this tutorial, we will need data for a Nivo Bump chart.
# You can get random data there, in tab 'data': https://nivo.rocks/bump/
#
# As you will see below, this session state item will be updated when our
# code editor change, and it will be read by Nivo Bump chart to draw the data.

if "data" not in st.session_state:
    st.session_state.data = Path("data.json").read_text()

# Define a default dashboard layout.
# Dashboard grid has 12 columns by default.
#
# For more information on available parameters:
# https://github.com/react-grid-layout/react-grid-layout#grid-item-props

layout = [
    # Editor item is positioned in coordinates x=0 and y=0, and takes 6/12 columns and has a height of 3.
    dashboard.Item("editor", 0, 0, 6, 3),
    # Chart item is positioned in coordinates x=6 and y=0, and takes 6/12 columns and has a height of 3.
    dashboard.Item("chart", 6, 0, 6, 3),
    # Media item is positioned in coordinates x=0 and y=3, and takes 6/12 columns and has a height of 4.
    dashboard.Item("media", 0, 2, 12, 4),
]

# Create a frame to display elements.

with elements("demo"):

    # Create a new dashboard with the layout specified above.
    #
    # draggableHandle is a CSS query selector to define the draggable part of each dashboard item.
    # Here, elements with a 'draggable' class name will be draggable.
    #
    # For more information on available parameters for dashboard grid:
    # https://github.com/react-grid-layout/react-grid-layout#grid-layout-props
    # https://github.com/react-grid-layout/react-grid-layout#responsive-grid-layout-props

    with dashboard.Grid(layout, draggableHandle=".draggable"):

        # First card, the code editor.
        #
        # We use the 'key' parameter to identify the correct dashboard item.
        #
        # To make card's content automatically fill the height available, we will use CSS flexbox.
        # sx is a parameter available with every Material UI widget to define CSS attributes.
        #
        # For more information regarding Card, flexbox and sx:
        # https://mui.com/components/cards/
        # https://mui.com/system/flexbox/
        # https://mui.com/system/the-sx-prop/

        with mui.Card(key="editor", sx={"display": "flex", "flexDirection": "column"}):

            # To make this header draggable, we just need to set its classname to 'draggable',
            # as defined above in dashboard.Grid's draggableHandle.

            mui.CardHeader(title="Editor", className="draggable")

            # We want to make card's content take all the height available by setting flex CSS value to 1.
            # We also want card's content to shrink when the card is shrinked by setting minHeight to 0.

            with mui.CardContent(sx={"flex": 1, "minHeight": 0}):

                # Here is our Monaco code editor.
                #
                # First, we set the default value to st.session_state.data that we initialized above.
                # Second, we define the language to use, JSON here.
                #
                # Then, we want to retrieve changes made to editor's content.
                # By checking Monaco documentation, there is an onChange property that takes a function.
                # This function is called everytime a change is made, and the updated content value is passed in
                # the first parameter (cf. onChange: https://github.com/suren-atoyan/monaco-react#props)
                #
                # Streamlit Elements provide a special sync() function. This function creates a callback that will
                # automatically forward its parameters to Streamlit's session state items.
                #
                # Examples
                # --------
                # Create a callback that forwards its first parameter to a session state item called "data":
                # >>> editor.Monaco(onChange=sync("data"))
                # >>> print(st.session_state.data)
                #
                # Create a callback that forwards its second parameter to a session state item called "ev":
                # >>> editor.Monaco(onChange=sync(None, "ev"))
                # >>> print(st.session_state.ev)
                #
                # Create a callback that forwards both of its parameters to session state:
                # >>> editor.Monaco(onChange=sync("data", "ev"))
                # >>> print(st.session_state.data)
                # >>> print(st.session_state.ev)
                #
                # Now, there is an issue: onChange is called everytime a change is made, which means everytime
                # you type a single character, your entire Streamlit app will rerun.
                #
                # To avoid this issue, you can tell Streamlit Elements to wait for another event to occur
                # (like a button click) to send the updated data, by wrapping your callback with lazy().
                #
                # For more information on available parameters for Monaco:
                # https://github.com/suren-atoyan/monaco-react
                # https://microsoft.github.io/monaco-editor/api/interfaces/monaco.editor.IStandaloneEditorConstructionOptions.html

                editor.Monaco(
                    defaultValue=st.session_state.data,
                    language="json",
                    onChange=lazy(sync("data"))
                )

            with mui.CardActions:

                # Monaco editor has a lazy callback bound to onChange, which means that even if you change
                # Monaco's content, Streamlit won't be notified directly, thus won't reload everytime.
                # So we need another non-lazy event to trigger an update.
                #
                # The solution is to create a button that fires a callback on click.
                # Our callback doesn't need to do anything in particular. You can either create an empty
                # Python function, or use sync() with no argument.
                #
                # Now, everytime you will click that button, onClick callback will be fired, but every other
                # lazy callbacks that changed in the meantime will also be called.

                mui.Button("Apply changes", onClick=sync())

        # Second card, the Nivo Bump chart.
        # We will use the same flexbox configuration as the first card to auto adjust the content height.

        with mui.Card(key="chart", sx={"display": "flex", "flexDirection": "column"}):

            # To make this header draggable, we just need to set its classname to 'draggable',
            # as defined above in dashboard.Grid's draggableHandle.

            mui.CardHeader(title="Chart", className="draggable")

            # Like above, we want to make our content grow and shrink as the user resizes the card,
            # by setting flex to 1 and minHeight to 0.

            with mui.CardContent(sx={"flex": 1, "minHeight": 0}):

                # This is where we will draw our Bump chart.
                #
                # For this exercise, we can just adapt Nivo's example and make it work with Streamlit Elements.
                # Nivo's example is available in the 'code' tab there: https://nivo.rocks/bump/
                #
                # Data takes a dictionary as parameter, so we need to convert our JSON data from a string to
                # a Python dictionary first, with `json.loads()`.
                #
                # For more information regarding other available Nivo charts:
                # https://nivo.rocks/

                nivo.Bump(
                    data=json.loads(st.session_state.data),
                    colors={ "scheme": "spectral" },
                    lineWidth=3,
                    activeLineWidth=6,
                    inactiveLineWidth=3,
                    inactiveOpacity=0.15,
                    pointSize=10,
                    activePointSize=16,
                    inactivePointSize=0,
                    pointColor={ "theme": "background" },
                    pointBorderWidth=3,
                    activePointBorderWidth=3,
                    pointBorderColor={ "from": "serie.color" },
                    axisTop={
                        "tickSize": 5,
                        "tickPadding": 5,
                        "tickRotation": 0,
                        "legend": "",
                        "legendPosition": "middle",
                        "legendOffset": -36
                    },
                    axisBottom={
                        "tickSize": 5,
                        "tickPadding": 5,
                        "tickRotation": 0,
                        "legend": "",
                        "legendPosition": "middle",
                        "legendOffset": 32
                    },
                    axisLeft={
                        "tickSize": 5,
                        "tickPadding": 5,
                        "tickRotation": 0,
                        "legend": "ranking",
                        "legendPosition": "middle",
                        "legendOffset": -40
                    },
                    margin={ "top": 40, "right": 100, "bottom": 40, "left": 60 },
                    axisRight=None,
                )

        # Third element of the dashboard, the Media player.

        with mui.Card(key="media", sx={"display": "flex", "flexDirection": "column"}):
            mui.CardHeader(title="Media Player", className="draggable")
            with mui.CardContent(sx={"flex": 1, "minHeight": 0}):

                # This element is powered by ReactPlayer, it supports many more players other
                # than YouTube. You can check it out there: https://github.com/cookpete/react-player#props

                media.Player(url=media_url, width="100%", height="100%", controls=True)
```



### Uber pickups

```python
import streamlit as st
import pandas as pd
import numpy as np

st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
            'streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

data_load_state = st.text('Loading data...')
data = load_data(10000)
data_load_state.text("Done! (using st.cache_data)")

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

st.subheader('Number of pickups by hour')
hist_values = np.histogram(data[DATE_COLUMN].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# Some number in the range 0-23
hour_to_filter = st.slider('hour', 0, 23, 17)
filtered_data = data[data[DATE_COLUMN].dt.hour == hour_to_filter]

st.subheader('Map of all pickups at %s:00' % hour_to_filter)
st.map(filtered_data)
```



# Transformers

```python
import transformers
from transformers import AutoTokenizer
```

## Docker

```
# 添加docker用户组（已经有了）
sudo groupadd docker
# 将用户添加到docker用户组
sudo gpasswd -a ${USER} docker
# 重启docker服务
sudo service docker restart
# 退出重连
```

```
# 列出所有images文件
docker images

# --gpus all 参数代表 docker 使用全部 gpu
# -t: 在新容器内制定一个伪终端
# -i: 允许对容器内的标准输入进行交互
# -d: 后台模式

# 查看所有容器
docker ps -a
# 删除容器
docker rm 'CONTAINER ID'
# 启动容器
docker start 'CONTAINER ID'
# 停止容器
docker stop 'CONTAINER ID'
# 运行容器
docker exec -it test_api /bin/bash
# 退出容器命令行
CTRL+D
```
