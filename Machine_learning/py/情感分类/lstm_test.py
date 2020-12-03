"""
lstm 的尝试使用
"""

import torch.nn as nn
import torch

batch_size = 10
seq_len = 20 # 句子的长度
vocab_size = 100 # 词典的数量
embedding_dim = 30 # 用长度30的向量来表示一个词语

hidden_size = 18
num_layer = 1


# 构造一个batch的数据
input = torch.randint(low=0,high=100,size=[batch_size,seq_len]) # [10,20]

# 数据经过embedding 处理
embedding=nn.Embedding(vocab_size,embedding_dim)
input_embedding = embedding(input) # [10,20,30]

# 把embedding之后的数据传入lstm中
lstm = nn.LSTM(input_size=embedding_dim,hidden_size=hidden_size,num_layers=num_layer,batch_first=True)
output,(h_n,c_n) = lstm(input_embedding)
print(output)
print("*"*100)
print(h_n)
print("*"*100)
print(c_n)

# 获取最后一个时间步上的输出
last_output = output[:,-1,:]
# 获取最后一次的hidden_state
last_hidden_state = h_n[-1,:,:]

print(last_hidden_state==last_output)