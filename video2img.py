#UCF101数据集抽取图片
import cv2,os
def video_to_img(video_path,save_path):
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        i=0
        while True:
            ret,frame = cap.read()
            if ret!=True:
                break
            #保存图片
            cv2.imwrite(save_path+str(100000+i)+'.png', frame)
            i += 1

#当前文件件所在文件夹
path = "/home/aistudio/split_video/"
#读取图片文件列表
filelist = []
with open(path + 'filelist.txt','r') as f:
    for line in f.readlines():
        #去掉换行符,加入列表
        filelist.append(line.strip('\n'))

#生成数据与标签
def generate_data_label(filelist):
    data = []
    label = []
    for video_path in filelist:
        print(video_path)
        print(video_path.split("/")[-2])
        save_path = "/home/aistudio/images/"+video_path.split("/")[-2]+"/"+video_path.split("/")[-1].split(".")[0]+"/"
        #检测文件夹是否存在
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        video_to_img(video_path, save_path)


generate_data_label(filelist)