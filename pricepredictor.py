import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt

import math
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense,LSTM

plt.style.use('bmh')

df = pd.read_csv("datasets_1869_18570_bitcoin_cash_price.csv")
price = df.Close
date = df.Date
date_format = pd.get_dummies(date)

dataset = df.filter(['Close'])
# print(dataset)
training_data_len = math.ceil( len(dataset) * .8 )
# print(dataset.shape)

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

# print("ayyyy",scaled_data)

train_data = scaled_data[0:training_data_len, :]

x_train = []
y_train = []

for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i,0])
    if i <= 60:
        print(x_train)
        print("yo")
        print(y_train)

x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train.shape)

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape= (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train,y_train, batch_size=1,epochs=1)

test_data = scaled_data[training_data_len - 60: , :]

x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt( np.mean( predictions - y_test )**2 )
print(rmse)
# df['Prediction'] = df[['Close']].shift(periods=5, axis=1)

# plt.figure(figsize=(16,8))
# plt.title("Price history of Bitcoin")
# plt.xlabel('Days')
# plt.ylabel('Closing Price USD ($)')
# plt.plot(price)
# plt.show()

# X_train, X_test, y_train, y_test = train_test_split(date_format.values,price,test_size=0.2, random_state=10)


# prediction = LinearRegression()
# prediction.fit(X_train,y_train)
# prediction.score(X_test,y_test)
# print(prediction.score(X_test,y_test))

# crossvalid = ShuffleSplit(n_splits=5,test_size=0.2, random_state=0)
#
# print(cross_val_score(LinearRegression(), date_format, price, cv=crossvalid))

# def find_best_model_using_gridsearchcv(x,y):
#     algos = {
#         'linear_regression': {
#             'model': LinearRegression(),
#             'params': {
#                 'normalize': [True, False]
#             }
#         },
#         'lasso': {
#             'model': Lasso(),
#             'params': {
#                 'alpha': [1,2],
#                 'selection': ['random', 'cyclic']
#             }
#         },
#         'decision_tree': {
#             'model': DecisionTreeRegressor(),
#             'params': {
#                 'criterion': ['mse', 'friedman_mse'],
#                 'splitter': ['best', 'random']
#             }
#         }
#     }
#     scores = []
#     cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
#     for algo_name, config in algos.items():
#         gs = GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
#         gs.fit(date_format, price)
#         scores.append({
#             'model': algo_name,
#             'best_score': gs.best_score_,
#             'best_params': gs.best_params_
#         })
#
#     final = pd.DataFrame(scores, columns=['model', 'best_score', 'best_params'])
#     final.to_csv("final.csv")

# find_best_model_using_gridsearchcv(date_format,price)
#
# def predict_price(dates,price):
#     date_index = np.where(date_format.columns == dates)[0][0]
#
#     x = np.zeros(len(date_format.columns))
#     if date_index >= 0:
#         x[date_index] = 1
#
#     return prediction.predict([x])[0]

# predict_price('Feb 20, 2019', 1000)