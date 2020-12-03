import pandas as pd
from sklearn.metrics import f1_score
import fasttext

# 转换为FastText需要的格式
train_df = pd.read_csv(r'D:/OneDrive - stu.cqupt.edu.cn/研究生代码/news/train_set.csv', sep='\t', nrows=200000,encoding="utf-8")
train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
train_df[['text','label_ft']].iloc[:-150000].to_csv('train.csv', index=None, header=None, sep='\t',encoding="utf-8")


model = fasttext.train_supervised('train.csv', lr=1, wordNgrams=2,
                                       verbose=2, minCount=1, epoch=25, loss="hs")

val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-150000:]['text']]
print(f1_score(train_df['label'].values[-150000:].astype(str), val_pred, average='macro'))
# 0.82