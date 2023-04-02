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
`ctrl+b $`