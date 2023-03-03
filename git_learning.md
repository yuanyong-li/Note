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

## 小知识

- 工作区：在电脑里能看到的目录；
- 版本库：在工作区有一个隐藏目录`.git`，是Git的版本库。
- `git add`实际上是把文件添加到暂存区
- `git commit`实际上是把暂存区的所有内容提交到当前分支