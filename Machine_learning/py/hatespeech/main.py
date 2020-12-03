from word_sequence import Word2Sequence
import pickle
import os
import re
import pandas as pd
from tqdm import tqdm

def tokenize(text):
    filters = [
        '!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '\.', '-', '<', '=', '>', '\?', '@',
        '\[', '\\', '\]', '^', '_', ':', ';', '\x97', '\x96', '\t', '\n', '\|', '\{', '\}',
    ]
    text = re.sub("<.*?>"," ",str(text),flags=re.S)
    text = re.sub("|".join(filters)," ", text, flags= re.S)
    return [i.strip().lower() for i in text.split()]

if __name__ == "__main__":
    train_df = pd.read_csv(r'D:/OneDrive - stu.cqupt.edu.cn/研究生代码/news/train_set.csv', sep='\t',encoding="utf-8")
    ws = Word2Sequence()
    for i in tqdm(range(0,len(train_df['label']))):
        sentence = str(train_df['text'][i]).split(" ")
        ws.fit(sentence)
    ws.build_vocab(min=10,max_features=10000)
    pickle.dump(ws, open("D:/OneDrive - stu.cqupt.edu.cn/研究生代码/news/ws.pkl","wb"))
    print(len(ws))