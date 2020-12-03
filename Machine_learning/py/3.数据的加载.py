import torch
from torch.utils.data import Dataset,DataLoader

data_path= r"E:\git_code\Machine_learning\resource\SMSSpamCollection"

# 完成数据集类

class MyDataset(Dataset):
    def __init__(self):
        # 获取每一行
        self.lines = open(data_path,encoding='utf-8').readlines() #按行读取文件

    def __getitem__(self, index):
        # 获取索引对应位置的一条数据
        cur_line = self.lines[index].strip() #strip() 用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        lable = cur_line[:4].strip() # 前面的内容定义为标签
        content = cur_line[4:].strip() # 后面的内容定义为文本
        return lable,content # 返回一个元组

    def __len__(self):
        return len(self.lines)

my_dataset = MyDataset()
data_loader = DataLoader(dataset=my_dataset,batch_size=2,shuffle=True) # type :<torch.utils.data.dataloader.DataLoader object at 0x00000215C7A1BE10>


if __name__ == '__main__':
    for i in data_loader:
        print(i)