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

`.jupyter/jupyter_notebook_config.py`

```python
c.NotebookApp.open_browser = False#不需要自动打开火狐浏览器
c.NotebookApp.port = 8888
c.NotebookApp.ip = '服务器IP'
c.NotebookApp.notebook_dir = '/home/a09/code/python_jupyter'
#设定默认打开的目录。
c.NotebookApp.allow_remote_access = True
#如今需要你之前设置的密码登录，填入下面的即可免密登录。
c.NotebookApp.token = ''
```

`jupyter notebook &`
