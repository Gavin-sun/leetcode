import pickle
import pandas as pd

ws = pickle.load(open("E:\git_code\Machine_learning\py\hatespeech\data\ws.pkl","rb"))
train = pd.read_excel(r"E:\git_code\Machine_learning\py\hatespeech\data\1.xls")
print(train['tweet'][0])

text = ws.transform(train['tweet'][1])
print(text)