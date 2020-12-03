# 使用pytorch 完成手写数字的识别
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from  torchvision.datasets import MNIST
from  torchvision.transforms import  Compose,Normalize,ToTensor

BATCH_SIZE = 128
TEST_BATCH_SIZE = 1000
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 1.准备数据集
def get_dataloader(train=True,batch_size=BATCH_SIZE):
    transform_fn =  Compose([
        ToTensor(),
        Normalize(mean=(0.1307,),std=(0.3081,)) # 均值和标准差
    ])

    dataset = MNIST(root= r"E:\git_code\Machine_learning\resource", train=train, download=False,transform=transform_fn)
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle= True)
    return  data_loader

# 2.构建模型
class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel,self).__init__()
        self.fc1 = nn.Linear(28*28*1,28)
        self.fc2 = nn.Linear(28,10)
    def forward(self,input):
        """
        :param input: [batch_size,1,28,28]
        :return:
        """
        # 1.形状的修改
        x = input.view([-1,1*28*28])
        # 2.全连接的操作
        x = self.fc1(x)
        # 3.激活函数的处理
        x = F.relu(x)
        # 4.输出层
        out = self.fc2(x)
        return F.log_softmax(out,dim=-1)

model = MnistModel().to(device)
optimizer = Adam(model.parameters(),lr = 0.001)

if os.path.exists(r"E:\git_code\Machine_learning\py\model\model.pkl"):
    model.load_state_dict(torch.load(r"E:\git_code\Machine_learning\py\model\model.pkl"))
    optimizer.load_state_dict(torch.load(r"E:\git_code\Machine_learning\py\model\optimizer.pkl"))

def train(epoch):
    data_loader = get_dataloader()
    for idx,(input,target) in enumerate(data_loader):
        optimizer.zero_grad()
        input = input.to(device)
        target = target.to(device)
        output = model(input) # 调用模型,得到预测值
        loss = F.nll_loss(output,target).to(device) # 得到损失
        loss.backward() # 反向传播
        optimizer.step() # 梯度的更新
        if idx%100 ==0:
            print(epoch,idx,loss.item())
            torch.save(model.state_dict(),r"E:\git_code\Machine_learning\py\model\model.pkl")
            torch.save(optimizer.state_dict(),r"E:\git_code\Machine_learning\py\model\optimizer.pkl")

def test():
    loss_list= []
    acc_list =[]
    test_dataloader = get_dataloader(train=False,batch_size=TEST_BATCH_SIZE)
    for idx,(input,target) in enumerate(test_dataloader):
        with torch.no_grad():
            input = input.to(device)
            target = target.to(device)
            output = model(input) # 调用模型,得到预测值
            cur_loss = F.nll_loss(output,target).to(device) # 得到损失
            loss_list.append(cur_loss)
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc)
            # 计算准确率
            # output[batch_size,10] target:[batch_size]
    print("平均准确率,平均损失",torch.mean(torch.stack(acc_list)).item(),torch.mean(torch.stack(loss_list)).item())
if __name__ == '__main__':
    for i in range(1): # 训练三轮
        train(i)
    test()

