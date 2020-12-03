from word_sequence import Word2Sequence
import pickle
import os
from dataset import tokenize
from tqdm import tqdm
if __name__ == "__main__":
    path = r"E:\git_code\Machine_learning\resource\aclImdb\train"
    ws = Word2Sequence()
    temp_data_path =  [os.path.join(path,"pos"),os.path.join(path,"neg")]
    for data_path in temp_data_path:
        file_paths = [os.path.join(data_path,file_name) for file_name in os.listdir(data_path) if file_name.endswith("txt")]
        for file_path in tqdm(file_paths):
            sentence = tokenize(open(file_path,encoding='UTF-8').read())
            ws.fit(sentence)
    ws.build_vocab(min=10,max_features=10000)
    pickle.dump(ws, open("./model/ws.pkl","wb"))
    print(len(ws))