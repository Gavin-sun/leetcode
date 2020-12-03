import pandas as pd

train_df = pd.read_csv(r'rs/test_a.csv', sep='\t') # 查看文本结构
print(train_df.head(10))
