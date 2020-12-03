import torch
import os
import re
from torch.utils.data import  DataLoader,Dataset
from lib import ws,max_len
data_base_path = r"E:\git_code\Machine_learning\resource\aclImdb"

# 1.定义tokenize的方法
#       使用正则表达式 来对词语进行分词和过滤
def tokenize(text):
    filters = [
        '!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '\.', '-', '<', '=', '>', '\?', '@',
        '\[', '\\', '\]', '^', '_', ':', ';', '\x97', '\x96', '\t', '\n', '\|', '\{', '\}',
    ]
    text = re.sub("<.*?>"," ",text,flags=re.S)
    text = re.sub("|".join(filters)," ", text, flags= re.S)
    return [i.strip().lower() for i in text.split()]

class ImdbDataset(Dataset):
    def __init__(self, train=True):
        # 先选择是哪个文件地址 是要训练还是测试
        self.train_data_path = r"E:\git_code\Machine_learning\resource\aclImdb\train"
        self.test_data_path = r"E:\git_code\Machine_learning\resource\aclImdb\test"
        data_path = self.train_data_path if train==True else self.test_data_path

        # 把所有的文件名放入列表 拼凑文件 path与文件名,得到唯一的文件地址,放入列表中
        temp_data_path = [os.path.join(data_path,"pos"), os.path.join(data_path,"neg")]
        self.total_file_path = [] # 所有的评论的文件的路径
        for path in temp_data_path:
            file_name_list = os.listdir(path) # 列出当前地址的所有文件,输出一个列表
            file_path_list = [os.path.join(path, i) for i in file_name_list if i.endswith(".txt")]
            self.total_file_path.extend(file_path_list)

    def __getitem__(self, index):
        file_path = self.total_file_path[index]
        label_str = file_path.split("\\")[-2] # 取pos/neg 将其转换为二分类的样本
        # 获取label
        label = 0 if label_str== "neg" else 1
        # 获取内容
        content = open(file_path,encoding="utf-8").read()

        tokens = tokenize(content) #分词后将内容和标签返回
        return tokens,label


    def __len__(self):
        return len(self.total_file_path)

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

