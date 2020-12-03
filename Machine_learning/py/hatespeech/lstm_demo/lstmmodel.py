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
import os
import torch.nn.functional  as F
import numpy as np
from tqdm import tqdm
import pickle
ws = pickle.load(open("E:\git_code\Machine_learning\py\hatespeech\data\ws.pkl","rb"))
max_len=500
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


from torch.utils.data import  DataLoader,Dataset
import  pandas as pd

class ImdbDataset(Dataset):
    def __init__(self, train=True):
        # 先选择是哪个文件地址 是要训练还是测试
        self.path = "E:/git_code/Machine_learning/py/新闻分类/rs/2.xls"
        self.df = pd.read_excel(self.path)
        # 把所有的文件名放入列表 拼凑文件 path与文件名,得到唯一的文件地址,放入列表中
        if train==True:
            self.df=self.df[:40000]
        else:
            self.df=self.df[40000:]

    def __getitem__(self, index):
        label = self.df['label'][index]
        # 获取内容
        content = self.df['text'][index]
        tokens = content.split(" ") #分词后将内容和标签返回
        return tokens,label


    def __len__(self):
        return len(self.df)

def get_dataloader(train = True,batch_size=128):
    """
    :param train: 是否为训练样本
    :return: 数据迭代器(功能,打乱,分片,可以自己写在 collate中写 处理函数
    """
    imdb_dataset = ImdbDataset(train)
    data_loader = DataLoader(imdb_dataset, batch_size=batch_size, shuffle= True,collate_fn=collate_fn)
    return data_loader

def collate_fn(batch):
    content,label = list(zip(*batch))  # *batch 意思是解压,返回二维矩阵模式
    content = [ws.transform(i,max_len=max_len) for i in content]
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    return content,label




class MyModel(nn.Module):
    def __init__(self):
        super(MyModel,self).__init__()
        self.embedding = nn.Embedding(len(ws),100) # 调用word embedding 前面是词典数量,后面是用来表示的维度数
        # 加入LSTM
        self.lstm = nn.LSTM(input_size=100,hidden_size=128,num_layers=3,
                            batch_first=True,bidirectional=True,dropout=0.5)
        self.fc = nn.Linear(128*2, 128)
        self.fc2 = nn.Linear(128, 14)

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


model = MyModel().to(device)
optimizer = Adam(model.parameters(), lr=0.02)

def train(epoch):
    for idx,(input,target) in enumerate(get_dataloader(train=True,batch_size=8)):
        input = input.to(device)
        target = target.to(device)
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
    data_loader = get_dataloader(train=False,batch_size=64)
    for idx,(input,target) in tqdm(enumerate(data_loader),total=len(data_loader),ascii=True,desc="测试"):
        input = input.to(device)
        target = target.to(device)
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
    eval()
# eval()