import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets,transforms

from cif.net import MyModel
"""
conda activate
tensorboard --logdir =logs
"""
writer = SummaryWriter(log_dir="logs")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32,padding=4),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
# 训练数据集
train_data_set = datasets.CIFAR10(root="./data",
                                  train=True,
                                  transform=transform,
                                  download=True)
# 测试数据集
test_data_set = datasets.CIFAR10(root="./data",
                                  train=False,
                                  transform=transform,
                                  download=True)
# 加载数据集
train_data_loader = DataLoader(train_data_set,batch_size=64,shuffle=True)
test_data_loader = DataLoader(test_data_set,batch_size=64,shuffle=False)

# 数据集的大小
train_data_size = len(train_data_set)
test_data_size = len(test_data_set)

# 定义网络
myModel = MyModel()
#定义训练的设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(device)

loss_fn = torch.nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

optimizer = torch.optim.SGD(myModel.parameters(), lr=0.01)
epochs = 300
# 测试
best_acc = 0.0
for epoch in range(epochs):
    #损失变量
    train_total_loss = 0.0
    test_total_loss = 0.0
    # 准确率
    train_total_acc = 0.0
    test_total_acc = 0.0
    print(f"训练轮数{epoch + 1}/{epochs}")

    #开始训练
    for data in train_data_loader:
        inputs, targets = data
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        #前向传播，得到真实值
        outputs = myModel(inputs)
        #计算损失
        loss = loss_fn(outputs, targets)
        #反向传播
        loss.backward()
        #更新参数
        optimizer.step()

        acc = (torch.argmax(outputs,1) == targets).sum().item()
        train_total_loss += loss.item()
        train_total_acc += acc

    # 训练一轮过后进行测试
    with torch.no_grad():
        for data in test_data_loader:
            inputs, targets = data
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 前向传播，得到真实值
            outputs = myModel(inputs)

            # 计算损失
            loss = loss_fn(outputs, targets)

            acc = (torch.argmax(outputs, 1) == targets).sum().item()
            test_total_loss += loss.item()
            test_total_acc += acc

    print("train_loss:{}, train_acc:{}, test_loss:{}, test_acc:{}".format(train_total_loss,
                                                                         train_total_acc/train_data_size,
                                                                         test_total_loss,
                                                                         test_total_acc/test_data_size))
    writer.add_scalar("loss/train",train_total_loss,epoch + 1)
    writer.add_scalar("acc/train",train_total_acc/train_data_size,epoch + 1)
    writer.add_scalar("loss/test",test_total_loss,epoch + 1)
    writer.add_scalar("acc/test",test_total_acc/test_data_size,epoch + 1)

    if((test_total_acc/test_data_size) > best_acc):
        best_acc = test_total_acc/test_data_size
        torch.save(myModel,"model/model_{}.pth".format(epoch+1))