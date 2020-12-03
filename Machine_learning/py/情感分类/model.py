"""
    定义模型
    模型优化方法:
    添加一个新的全连接层作为输出层,激活函数处理
    把双向的Lstm的output穿个一个单向LSTM再进行处理
"""

import torch.nn as nn
import torch
from torch.optim import Adam
from torch.optim import Adam
from lib import ws,max_len
import os
import lib
import torch.nn.functional  as F
from dataset import get_dataloader
import numpy as np
from tqdm import tqdm

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.embedding = nn.Embedding(len(ws),100) # 调用word embedding 前面是词典数量,后面是用来表示的维度数
        # 加入LSTM
        self.lstm = nn.LSTM(input_size=100,hidden_size=lib.hidden_size,num_layers=lib.num_layers,
                batch_first=True,bidirectional=lib.bidriectional,dropout=lib.dropout)
        self.fc = nn.Linear(lib.hidden_size*2, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, input):
        """
        :param input: [batch_size,max_len]
        :return:
        """
        x = self.embedding(input) # 进行embedding操作,形状:[batch_size, max_len,100]
        x,(h_n,c_n) = self.lstm(x) # x:[batch_size,max_len,2*hidden_size] h_n:[2*2,batch_size,hidden_size]
        # 获取两个方向最后一次的output,进行concat
        output_fw = h_n[-2,:,:] # 正向最后一次的输出
        output_bw = h_n[-1,:,:] # 反向最后一次的输出
        output = torch.cat([output_fw,output_bw],dim=-1) # output:[batch_size,hidden_size*2]

        out = self.fc(output)
        out = self.fc2(out)

        return F.log_softmax(out,dim=-1)


model = MyModel().to(lib.device)
optimizer = Adam(model.parameters(), lr=0.001)
if os.path.exists("./model/model.pkl"):
    model.load_state_dict(torch.load("./model/model.pkl"))
    optimizer.load_state_dict(torch.load("./model/optimizer.pkl"))

def train(epoch):
    for idx,(input,target) in enumerate(get_dataloader(train=True,batch_size=128)):
        input = input.to(lib.device)
        target = target.to(lib.device)
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        print(epoch,idx,loss.item())
        if idx%100==0:
            torch.save(model.state_dict(),"./model/model.pkl")
            torch.save(optimizer.state_dict(),"./model/optimizer.pkl")

def eval():
    loss_list =[]
    acc_list = []
    data_loader = get_dataloader(train=False,batch_size=lib.test_batch_size)
    for idx,(input,target) in tqdm(enumerate(data_loader),total=len(data_loader),ascii=True,desc="测试"):
        input = input.to(lib.device)
        target = target.to(lib.device)
        with torch.no_grad():
            output = model(input)
            cur_loss = F.nll_loss(output,target)
            loss_list.append(cur_loss.cpu().item())
            # 计算准确率
            pred = output.max(dim=-1)[-1]
            cur_acc = pred.eq(target).float().mean()
            acc_list.append(cur_acc.cpu().item())
    print("total loss acc:",np.mean(loss_list),np.mean(acc_list))


if __name__ == "__main__":
     for i in range(10):
         train(i)
    # eval()