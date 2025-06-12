import os
import cv2
import torch
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

classes = ['飞机','汽车','鸟','猫','鹿','狗','青蛙','马','船','卡车']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 将模型放入训练设备中
model = torch.load("D:\pythonCode\Tudui\cif\model\model_7.pth")
model = model.to(device)

file_path = 'D:\pythonCode\Tudui\cif\imgs'
# 得到所有的文件名
file_name = os.listdir(file_path)
# 得到每个图片的地址
images_files = [os.path.join(file_path,f) for f in file_name ]
for img in images_files:
    image = cv2.imread(img)
    cv2.imshow("image",image)
    #修改图片大小
    image = cv2.resize(image,(32,32))
    #将图片转成RGB，即3个通道
    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    image = transform(image)
    image = torch.reshape(image,(1,3,32,32))
    image = image.to(device)
    output = model(image)
    index = torch.argmax(output)
    print(f"预测结果：{classes[index]}")
    cv2.waitKey(0)
    #等待用户按键

