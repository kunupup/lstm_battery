import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Dropout
import pandas as pd
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
df=pd.read_csv("20200708.csv",parse_dates=["Date"],index_col=[0])
df = df['WH-OUT'].values
df=df.reshape(-1,1)
test_split=round(len(df)*0.20)
df_for_training=df[:-882]
df_for_testing=df[-880:]
print(df_for_training.shape)
print(df_for_testing.shape)
scaler = StandardScaler()
df_for_training_scaled = scaler.fit_transform(df_for_training)
df_for_testing_scaled=scaler.transform(df_for_testing)
def createXY(dataset,n_past):
    dataX = []
    dataY = []
    for i in range(n_past, len(dataset)):
            dataX.append(dataset[i - n_past:i, 0:dataset.shape[1]])
            dataY.append(dataset[i,0])
    return np.array(dataX),np.array(dataY)
trainX,trainY=createXY(df_for_training_scaled,30)
testX,testY=createXY(df_for_testing_scaled,30)
print("trainX Shape-- ",trainX.shape)
print("trainY Shape-- ",trainY.shape)
print("testX Shape-- ",testX.shape)
print("testY Shape-- ",testY.shape)
print("trainX[0]-- \n",trainX[1])
print("trainY[0]-- ",trainY[1])
def build_model(optimizer):
    grid_model = Sequential()
    grid_model.add(LSTM(50,return_sequences=True,input_shape=(30,1)))
    grid_model.add(LSTM(50))
    grid_model.add(Dropout(0.2))
    grid_model.add(Dense(1))

    grid_model.compile(loss = 'mse',optimizer = optimizer)
    return grid_model

grid_model = KerasRegressor(build_fn=build_model,verbose=1)
parameters = {'batch_size' : [16,20],
              'epochs' : [8,10],
              'optimizer' : ['adam','Adadelta'] }

grid_search  = GridSearchCV(estimator = grid_model,
                            param_grid = parameters,
                            cv = 2)
grid_search = grid_search.fit(trainX,trainY,validation_data=(testX,testY))
grid_search.best_params_
my_model=grid_search.best_estimator_.model
prediction=my_model.predict(testX)
print("prediction\n", prediction)
print("\nPrediction Shape-",prediction.shape)
pred=scaler.inverse_transform(prediction)[:,0]
original=scaler.inverse_transform(np.reshape(testY,(len(testY),1)))[:]
plt.plot(original, color = 'red', label = 'Real  w_H')
plt.plot(pred, color = 'blue', label = 'Predicted  w_H')
plt.title(' Prediction')
plt.xlabel('Time')
plt.ylabel(' Stock Price')
plt.legend()
plt.show()
