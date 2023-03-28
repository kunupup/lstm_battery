import torch
import torch.nn as nn
from dataload import dataload
from dataload import scaler_result
import matplotlib.pyplot as plt
# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size=30, hidden_size=1, output_size=1):
        super().__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        lstm_out, _ = self.lstm(input.view(len(input),-1, 30))
        predictions = self.linear(lstm_out.view(len(input), 1))
        return predictions[-1]

# 准备数据
X_train,Y_train,X_test,Y_test=dataload()
X_train=torch.tensor(X_train,dtype=torch.float32)
x_test=torch.tensor(X_test,dtype=torch.float32)
model = LSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
epochs = 100
for i in range(epochs):
    y_pred = model(X_train)
    y_pred = y_pred.to(torch.float32)
    loss = criterion(y_pred, Y_train)
    optimizer.zero_grad()
    loss = loss.to(torch.float32)
    loss.backward()
    optimizer.step()

    if i % 10 == 0:
        print(f'Epoch {i}, loss: {loss.item()}')

# 进行预测
train_predict = model(X_train)
test_predict = model(X_test)
test_predict,Y_test=scaler_result(test_predict)
# 绘制预测结果和真实结果的折线图
plt.figure(figsize=(12, 8))
plt.plot(Y_test, label='True')
plt.plot(test_predict, label='Predicted')
plt.legend()
plt.title('LSTM Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.show()
