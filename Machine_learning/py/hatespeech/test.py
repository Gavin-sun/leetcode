import pandas as pd
import numpy as np
#1.向csv文件中写入数据
from tqdm import tqdm
df= pd.read_csv(r'D:/OneDrive - stu.cqupt.edu.cn/研究生代码/news/train_set.csv',sep='\t')
sum=0
for i in tqdm(range(0,len(df['text']))):
    if len(df['text'][i].split(" ")) <=1000:
        sum=sum+1
print(sum/len(df['text']))

