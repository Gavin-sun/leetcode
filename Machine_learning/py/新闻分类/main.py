from word_sequence import Word2Sequence
import pickle
import os
import dataset
import pandas as pd
from tqdm import tqdm
if __name__ == "__main__":
    train_df = pd.read_csv(r'rs/train_set.csv', sep='\t')
    mydataset = dataset.MyDataset()
    ws = Word2Sequence()
    for i in tqdm(range(len(mydataset))):
        ws.fit(mydataset[i][0].split(" "))
    ws.build_vocab(min=10,max_features=10000)
    pickle.dump(ws, open("./model/ws.pkl","wb"))
    print(len(ws))

    # if __name__ == "__main__":
    #     train_df = pd.read_csv(r'rs/test_a.csv', sep='\t')
    #
    # mydataset = dataset.MyDataset()
    #
    # ws = Word2Sequence()
    # ws.fit(mydataset[0][0].split(" "))
    # ws.build_vocab(min=0, max=10000, max_features=500)
    # print(ws.dict)
    # ret = ws.transform(mydataset[0][0].split(" "),max_len=1500)
    # print(ret)