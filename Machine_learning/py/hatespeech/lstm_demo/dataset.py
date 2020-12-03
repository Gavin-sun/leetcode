import torch
import pickle
from torch.utils.data import  DataLoader,Dataset
import  pandas as pd
#data_base_path = r"E:/git_code/Machine_learning/py/hatespeech/data/"
ws = pickle.load(open("E:\git_code\Machine_learning\py\hatespeech\data\ws.pkl","rb"))
max_len=10

class ImdbDataset(Dataset):
    def __init__(self, train=True):
        # 先选择是哪个文件地址 是要训练还是测试
        self.path = "E:/git_code/Machine_learning/py/hatespeech/data/2.xls"
        self.df = pd.read_excel(self.path);
        # 把所有的文件名放入列表 拼凑文件 path与文件名,得到唯一的文件地址,放入列表中
        if train==True:
            self.df=self.df[:18000]
        else:
            self.df=self.df[18000:]

    def __getitem__(self, index):
        label = self.df['class'][index]
        # 获取内容
        content = self.df['tweet'][index]

        tokens = content.split(" ") #分词后将内容和标签返回
        return tokens,label


    def __len__(self):
        return len(self.df)

def get_dataloader(train = True,batch_size=None):
    """
    :param train: 是否为训练样本
    :return: 数据迭代器(功能,打乱,分片,可以自己写在 collate中写 处理函数
    """
    imdb_dataset = ImdbDataset(train)
    data_loader = DataLoader(imdb_dataset, batch_size=batch_size, shuffle= True,collate_fn=collate_fn)
    return data_loader

def collate_fn(batch):
    print(batch)
    content,label = list(zip(*batch))  # *batch 意思是解压,返回二维矩阵模式
    content = [ws.transform(i,max_len=max_len) for i in content]
    content = torch.LongTensor(content)
    label = torch.LongTensor(label)
    return content,label

if __name__ == '__main__':
    for idx,(input,target) in enumerate(get_dataloader(train=True)):
        print(idx)
        print(input)
        print(target)
        break

