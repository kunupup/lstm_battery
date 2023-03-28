import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
#获取数据
df=pd.read_csv("20200708.csv",parse_dates=["Date"],index_col=[0])
df=df['WH-OUT'].values
df=df.reshape(-1,1)
test_split=round(len(df)*0.20)
df_for_training=df[:-882]
df_for_testing=df[-882:]
print(df_for_training.shape)
print(df_for_testing.shape)
scaler = StandardScaler()
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled=scaler.transform(df_for_testing)


def createXY(dataset, n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
        dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
        dataY.append(dataset[i, 0])
    return np.array(dataX), np.array(dataY)
def dataload():

    trainX,trainY=createXY(df_for_training_scaled,30)
    testX,testY=createXY(df_for_testing_scaled,30)
    return torch.from_numpy(trainX),torch.from_numpy(trainY),torch.from_numpy(testX),torch.from_numpy(testY)
def scaler_result(prediction):
    testX, testY = createXY(df_for_testing_scaled, 30)
    pred = scaler.inverse_transform(prediction)[:, 0]
    original = scaler.inverse_transform(np.reshape(testY, (len(testY), 1)))[:]
    return pred,original

