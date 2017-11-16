1.创建远程仓库(github上边创建)  
2.将远程仓库和本地git关联 
  使用如下命令:  'git remote add origin https://github.com/sunwantong/O2Otianchi.git'(origin为远程库的名字,约定俗成叫origin,也可以改为别的名字,随意)
3.关联之后提交本地文件或者文件夹到远程仓库,首先切换到你要提交的文件夹所在的目录
  鼠标右击打开git bash,然后执行git init(这是很重要的一步),
  然后 git add '文件夹名字' 
  git commit -m '天池o2o新手赛'  引号里为对你提交内容的描述，
  然后第一次提交内容到远程仓库(git push -u origin master)加-u参数
  注意这个origin不一定是是这个单词，你喜欢就好，包括当面那个'git remote add origin https://github.com/sunwantong/O2Otianchi.git'仓库名.git中的origin，
  相当于你给这个地址起了一个短点的好记的名字
  这个命令 是将主分支master提交到远程仓库
  这个带有-u这个参数是指，将master分支的所有内容都提交
  第一次关联之后后边你再提交就可以不用这个参数了,之后你的每一次修改，你就可以直接push就好了(git push origin master)
4.git ssh免秘钥配置
   命令行输入ssh-keygen -t rsa 然后连续四个回车， 
   然后cd,ls -la 发现.ssh文件    id_rsa.pub的文件拷贝到github上边。