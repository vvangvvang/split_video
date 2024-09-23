import paddle
import cv2
import numpy as np
print(paddle.__version__)  #3.0.0
import os
import random
import shutil

#将视频转换为图片
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
def generate_image(filelist):
    #删除文件夹
    if os.path.exists("/home/aistudio/images"):
        shutil.rmtree("/home/aistudio/images")
    #文件列表乱序
    random.shuffle(filelist)
    for video_path in filelist[:200]:
        print(video_path)
        print(video_path.split("/")[-2])
        save_path = "/home/aistudio/images/"+video_path.split("/")[-2]+"/"+video_path.split("/")[-1].split(".")[0]+"/"
        #检测文件夹是否存在
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        video_to_img(video_path, save_path)

#generate_image(filelist)

#生成数据集,图片对和相应标签
def generate_data_label():
    data = []
    image_list_all = []
    for root, dirs, files in os.walk("/home/aistudio/images"):
        for filename in files:
            if filename.split(".")[-1] == "png":
                image_list_all.append(os.path.join(root, filename))
    #列出文件夹下的所有子目录
    for folder in os.listdir("/home/aistudio/images"):
        #读取文件夹下的所有图片
        for img_folder in os.listdir(os.path.join("/home/aistudio/images",folder)):
            #因为图片列表中，有可能要连续取两张，所以最后一张不在列表中
            img_list = os.listdir(os.path.join("/home/aistudio/images",folder,img_folder))
            img_list.sort()
            for i in range(len(img_list)-1):
                x1 = os.path.join("/home/aistudio/images",folder,img_folder,img_list[i])
                x2 = os.path.join("/home/aistudio/images",folder,img_folder,img_list[i+1])
                data.append([x1,x2,[1]])
            for i in range(len(img_list)-1):
                x1 = os.path.join("/home/aistudio/images",folder,img_folder,img_list[i])
                #第二张图片，在总图片列表中随机选一张
                x2 = random.sample(image_list_all,1)[0]
                data.append([x1,x2,[0]])
    return data
#generate_data_label()


#搭设框架

#建立一个网络模型，输入两张图片，输出两张图片是否是连续的
class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net,self).__init__()
        #卷积层,pading=1,stride=1
        #输入通道数3，输出通道数64，卷积核大小为(3,3)
        self.conv1 = paddle.nn.Conv2D(in_channels=3,out_channels=64,kernel_size=(3,3))
        self.BN1 = paddle.nn.BatchNorm2D(num_features=64)
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=(2,2))
        self.conv2 = paddle.nn.Conv2D(in_channels=128,out_channels=128,kernel_size=(3,3))
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=(2,2))
        self.BN2 = paddle.nn.BatchNorm2D(num_features=128)
        self.conv3 = paddle.nn.Conv2D(in_channels=128,out_channels=256,kernel_size=(3,3))
        self.pool3 = paddle.nn.MaxPool2D(kernel_size=(2,2))
        self.BN3 = paddle.nn.BatchNorm2D(num_features=256)
        self.conv4 = paddle.nn.Conv2D(in_channels=256,out_channels=256,kernel_size=(3,3))
        self.pool4 = paddle.nn.MaxPool2D(kernel_size=(2,2))
        self.BN4 = paddle.nn.BatchNorm2D(num_features=256)
        self.conv5 = paddle.nn.Conv2D(in_channels=256,out_channels=512,kernel_size=(3,3))
        self.pool5 = paddle.nn.MaxPool2D(kernel_size=(2,2))
        self.fc1 = paddle.nn.Linear(in_features=256*4*4,out_features=64)
        self.fc2 = paddle.nn.Linear(in_features=64,out_features=1)
        self.pad = paddle.nn.Pad2D(padding=[0,1,1,0], mode='constant')
        self.pad2 = paddle.nn.Pad2D(padding=[0,1,0,0], mode='constant')
    def forward(self,x1,x2):
        #输入两张图片，输出两张图片是否连续的
        x1 = self.conv1(x1)
        x1 = self.pool1(x1)
        x2 = self.pool2(self.conv1(x2))
        x1 = self.BN1(x1)
        x2 = self.BN1(x2)
        #将两张图片拼接
        x = paddle.concat([x1,x2],axis=1)
        #沿宽度和高度方向填充1步
        x = self.pad(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.pad(x)
        x = self.BN2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.pad2(x)
        x = self.BN3(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.pad2(x)
        x = self.BN4(x)
        x = self.conv5(x)
        x = self.pool5(x)
        x = paddle.reshape(x,(x.shape[0],-1))
        x = self.fc1(x)
        x = self.fc2(x)
        return x

#实例化网络
model = Net()
#打印网络结构
print(model)
#加载模型参数

model.set_state_dict(paddle.load("./model_46_29.pdparams"))

#分类损失函数
loss_fn = paddle.nn.BCEWithLogitsLoss()
#随机梯度下降
optimizer = paddle.optimizer.SGD(learning_rate=0.001,parameters=model.parameters())
#数据流经网络，计算损失，反向传播，更新权重
min_loss = 100.0
for k in range(8,100):
    generate_image(filelist)
    data = generate_data_label()
    #顺序打乱数据
    random.shuffle(data)
    batch_size = 100
    batch_num = len(data) // batch_size
    for i in range(10, batch_num):
        xx1,xx2, y = [], [], []
        for j in range(batch_size):
            d = data[i*batch_size + j]
            img_x1 = cv2.resize(cv2.imread(d[0]),(160,120))
            xx1.append(img_x1)
            img_x2 = cv2.resize(cv2.imread(d[1]),(160,120))
            xx2.append(img_x2)
            y.append(d[2])
        xx1 = np.array(xx1) - 127.5
        xx2 = np.array(xx2) - 127.5
        y = np.array(y)
        
        #前向传播
        x1 = paddle.to_tensor(xx1/128.).astype('float32')
        x2 = paddle.to_tensor(xx2/128.).astype('float32')
        y = paddle.to_tensor(y).astype('float32')
        #数据转换成NCHW
        x1 = paddle.transpose(x1, [0, 3, 1, 2])
        x2 = paddle.transpose(x2, [0, 3, 1, 2])
        out = model(x1,x2)
        #计算损失
        loss = loss_fn(out,y)
        if loss.numpy() < min_loss:
            min_loss = loss.numpy()
        if loss.numpy() < min_loss*1.1:
            paddle.save(model.state_dict(),"./model_"+str(k)+"_"+str(i)+".pdparams")
        if i%2==0:
            print('loss:',loss.numpy())
        #反向传播
        loss.backward()
        #更新参数
        optimizer.step()
        #清除梯度
        optimizer.clear_grad()