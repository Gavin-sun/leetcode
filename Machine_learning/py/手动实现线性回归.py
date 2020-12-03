import torch

# 准备数据->通过模型计算预测值->通过循环,反向传播,更新参数->输出

# 1.准备数据
# y = 3x+0.8
x = torch.rand([500, 1])
y_true = x*3+0.8


# 2.通过模型计算y_predict
w = torch.rand([1, 1], requires_grad=True)
b = torch.tensor(0, requires_grad=True, dtype=torch.float32)
learning_rate = 0.01

# y_predict = torch.matmul(x, w)+b
# 3.计算loss
# loss = (y_predict - y_true).pow(2).mean()

# 4.通过循环,反向传播,更新参数
for i in range(2000):
    # 2.通过模型计算y_predict
    y_predict = torch.matmul(x, w)+b
    # 3.计算loss
    loss = (y_predict - y_true).pow(2).mean()
    # 如果不将梯度设为0.则会累加
    if w.grad is not None:
        w.data.zero_()
    if b.grad is not None:
        b.data.zero_()
    # 损失函数前项计算,更新梯度
    loss.backward()

    w.data = w.data - learning_rate * w.grad
    b.data = b.data - learning_rate * b.grad
    if i%100 == 0:
        print("w,b.loss", w.data.item(), b.data.item(), loss.item())
