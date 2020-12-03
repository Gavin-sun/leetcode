import torch
import re
from lib import ws,max_len
import pandas as pd

from torch.utils.data import  DataLoader,Dataset


class MyDataset(Dataset):
    def __init__(self, train=True):
        self.train_df = pd.read_csv(r'rs/train_set.csv', sep='\t')

    def __getitem__(self, index):
        # 获取标志
        label = self.train_df["label"][index]
        # 获取内容
        content = self.train_df["text"][index].split(" ")
        return content,label

    def __len__(self):
        return self.train_df.shape[0]

def get_dataloader(train=True,batch_size=None):
    mydataset = MyDataset(True)
    data_loader = DataLoader(mydataset, batch_size=batch_size, shuffle= True,collate_fn=collate_fn)
    return data_loader

def collate_fn(batch):
    content,label = list(zip(*batch))  # *batch 意思是解压,返回二维矩阵模式
    content = [ws.transform(i,max_len=1500) for i in content]
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    return content,label

if __name__ == '__main__':
    train_df = pd.read_csv(r'rs/train_set.csv', sep='\t')
    content = train_df["text"][2].split(" ")
    print(content)
