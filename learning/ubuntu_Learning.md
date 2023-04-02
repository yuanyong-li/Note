# Ubuntu Learning

## 基本操作命令

* 查看文件权限：

  * ls -l filename
  * ls -ld folder

* 更改文件权限：

  * chmod u+x filename

    u 代表所有者（user）

    g 代表所有者所在的组群（group）

    o 代表其他人，但不是u和g （other）

    a 代表全部的人，也就是包括u，g和o

    r w x 

    sudo chmod 777 ：所有者、群组、其他用户都全权限

* 下载命令：
  
  * wget url
* 解压命令：
  
  - tar -zxvf filename.tar.gz

### nohup

```
nohup /home/scandb/code/predict/label_predict_main.py > temp.log 2>&1 &
# 将标准错误 2 重定向到标准输出 &1 ，标准输出 &1 再被重定向输入到 runoob.log 文件中
# https://www.runoob.com/linux/linux-comm-nohup.html 
nohup /home/scandb/anaconda3/envs/pytorch/bin/python3.7 /home/scandb/code/predict/label_predict_main.py
ps -aux | grep "label_predict_main.py"
kill -9
```





## 后台运行程序Screen

sudo apt-get install screen

screen -ls
screen -S name

conda activate envname
python pyname.py
Ctrl+a d	# 退出

screen -r temp	进入temp窗口