#用来处理数据集视频文件路径，存放在filelist.txt中
import os
#当前文件件所在文件夹
path = "/home/aistudio/split_video/"
#视频文件夹
video_path = "/home/aistudio/data/data101478/UCF-101/"
#创建一个文件列表
filelist = []
#打开文件
with open(path + 'files.txt','r') as f:
    #读取文件所有行，返回一个列表
    for line in f.readlines():
        #去掉换行符
        line = line.strip('\n')
        #列出文件夹下的所有文件
        print(video_path + line)
        for filename in os.listdir(video_path + line):
            #拼接路径
            filelist.append(video_path + line + '/' + filename)
#保存文件列表
with open(path + 'filelist.txt','w') as f:
    for file in filelist:
        f.write(file+'\n')