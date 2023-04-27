# tmux

`tmux`: 新建session
`tmux new -s <session-name>`

`tmux detach`: 离开session

`tmux ls`: 查看session列表
`ctrl + b s`

`tmux attach -t <session-name>`: 进入session

`tmux kill-session -t <session-name>`: 关闭session
`ctrl+d`: 关闭当前session

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
learning http://10.99.4.2:8891 密码为空
chatglm https://10.99.4.2:8890 密码为空
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

`pip install streamlit`
`streamlit run demo.py`
`streamlit run web_demo2.py --server.port 6008`
`python -m streamlit run your_script.py`

`streamlit run https://raw.githubusercontent.com/streamlit/demo-uber-nyc-pickups/master/streamlit_app.py`

## 常用方法

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

