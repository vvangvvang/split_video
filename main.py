import paddle
import cv2
import numpy as np
print(paddle.__version__)  #3.0.0

#搭设框架
#将图片读取为numpy数组
import cv2
import numpy as np
#图片路径
img_path = "/home/aistudio/images/"
#读取图片
img = cv2.imread(img_path + "100000.png")
#缩放图片
img = cv2.resize(img,(160,120))
#将图片转换为numpy数组
img = np.array(img)
print(type(img))
#将numpy数组转换为tensor
import paddle
tensor_img = paddle.to_tensor(img)
#打印tensor的形状
print(tensor_img.shape)
#缩放到固定大小

#建立一个网络模型，输入两张图片，输出两张图片是否是连续的
class Net(paddle.nn.Layer):
    def __init__(self):
        super(Net,self).__init__()
        #卷积层,pading=1,stride=1
        #输入通道数3，输出通道数64，卷积核大小为(3,3)
        self.conv1 = paddle.nn.Conv2D(in_channels=3,out_channels=64,kernel_size=(3,3))
        self.pool1 = paddle.nn.MaxPool2D(kernel_size=(2,2))
        self.conv2 = paddle.nn.Conv2D(in_channels=128,out_channels=128,kernel_size=(3,3))
        self.pool2 = paddle.nn.MaxPool2D(kernel_size=(2,2))
        self.conv3 = paddle.nn.Conv2D(in_channels=128,out_channels=256,kernel_size=(3,3))
        self.pool3 = paddle.nn.MaxPool2D(kernel_size=(2,2))
        self.conv4 = paddle.nn.Conv2D(in_channels=256,out_channels=256,kernel_size=(3,3))
        self.pool4 = paddle.nn.MaxPool2D(kernel_size=(2,2))
        self.conv5 = paddle.nn.Conv2D(in_channels=256,out_channels=512,kernel_size=(3,3))
        self.pool5 = paddle.nn.MaxPool2D(kernel_size=(2,2))
        self.fc1 = paddle.nn.Linear(in_features=256*4*4,out_features=64)
        self.fc2 = paddle.nn.Linear(in_features=64,out_features=2)
        self.pad = paddle.nn.Pad2D(padding=[0,1,1,0], mode='constant')
        self.pad2 = paddle.nn.Pad2D(padding=[0,1,0,0], mode='constant')
    def forward(self,x1,x2):
        #输入两张图片，输出两张图片是否连续的
        x1 = self.conv1(x1)
        x1 = self.pool1(x1)
        x2 = self.pool2(self.conv1(x2))
        #将两张图片拼接
        x = paddle.concat([x1,x2],axis=1)
        #沿宽度和高度方向填充1步
        x = self.pad(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.pad(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.pad2(x)
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.pad2(x)
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
#定义损失函数
#梯度下降SGD
loss_fn = paddle.nn.CrossEntropyLoss()
#随机梯度下降
optimizer = paddle.optimizer.SGD(learning_rate=0.01,parameters=model.parameters())
#数据流经网络，计算损失，反向传播，更新权重
for i in range(100):
    #前向传播
    img_tensor = paddle.to_tensor([img])
    #数据转换成float32
    img_tensor = paddle.cast(img_tensor, 'float32')
    #数据转换成NCHW
    img_tensor = paddle.transpose(img_tensor, [0, 3, 1, 2])
    out = model(img_tensor,img_tensor)
    #计算损失
    loss = loss_fn(out,paddle.to_tensor([0]))
    if i%10==0:
        print('loss:',loss.numpy())
    #反向传播
    loss.backward()
    #更新参数
    optimizer.step()
    #清除梯度
    optimizer.clear_grad()