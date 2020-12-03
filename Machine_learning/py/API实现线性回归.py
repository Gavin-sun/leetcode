from torch import nn
import torch


# 准备数据
# 首先构建模型,创建init 和forward 函数,并实例化模型
# 创建损失函数,并实例化损失函数
# 在循环中计算预测值,更新损失函数
# 输出

## Gpu 运行代码
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class mylinear(nn.Module):
    def __init__(self):
        super(mylinear, self).__init__()
        self.linear = nn.Linear(1, 1) #输入的维数:1, 输出的维数,1
    def forward(self, x):
        out = self.linear(x)
        return out

x = torch.rand([500,1]).to(device) #因为x变成了cuda 的tensor 所以y经过计算也是
y_true = 3*x+0.8

model = mylinear().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),0.01)

for i in range(2000):
    y_predict = model(x)
    loss = criterion(y_predict, y_true)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i%100==0:
        print(list(model.parameters())[0].item(),list(model.parameters())[1].item())