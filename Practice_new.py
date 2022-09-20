#Comparitive Study 



import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , confusion_matrix , mean_absolute_error , mean_absolute_percentage_error
import seaborn as sns 
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential
import tensorflow as tf

from matplotlib import pyplot as plot


data = pd.read_csv('C:/AQI project/New_data_with_aqi_cleaned.csv')

print(data.head())


features = data


label = features['Air_Quality_Index(AQI)']

print(label.head())

print(label.shape)


feature_data = pd.read_csv('C:/AQI project/feature_data_mod1.csv')

print(feature_data.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

scaler.fit(feature_data)

scaled_data = scaler.transform(feature_data)

print(scaled_data.shape)
print(scaled_data)

from sklearn.decomposition import PCA

pca = PCA(n_components=8)

pca.fit(scaled_data)

x_pca = pca.transform(scaled_data)

print(scaled_data.shape)

print(x_pca.shape)

print(x_pca)


X_train,X_Test,Y_train,Y_test = train_test_split(x_pca,label,test_size=0.25)

print(X_train.shape,Y_train.shape)
print(X_Test.shape)

#model = LinearRegression()

model = Sequential()


model.add(Dense(1024,activation='relu',input_shape = X_train.shape[1:]))

model.add(Dense(512,activation='relu'))

model.add(Dense(256,activation='relu'))

model.add(Dense(128,activation='relu'))

model.add(Dense(64,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(16,activation='relu'))

model.add(Dense(8,activation='relu'))

model.add(Dense(4,activation='relu'))

model.add(Dense(1,activation='linear'))

model.compile(loss='mse',optimizer='adam')

model.summary()




history = model.fit(X_train,Y_train,epochs=200)

Y_pred = model.predict(X_Test)

r2 = r2_score(Y_test,Y_pred)

print(r2)


from sklearn.metrics import mean_squared_error
from math import sqrt

rms = sqrt(mean_squared_error(Y_test, Y_pred))

print(rms)

MAE = mean_absolute_error(Y_test, Y_pred)
print(MAE)


MAPE = mean_absolute_percentage_error(Y_test, Y_pred)
print(MAPE)


plot.figure(figsize=(10,10))
plot.scatter(Y_test,Y_pred,c='crimson')


p1 = max(max(Y_pred), max(Y_test))
p2 = min(min(Y_pred), min(Y_test))
plot.plot([p1, p2], [p1, p2], 'b-')
plot.ylabel('True Value of AQI', fontsize=15)
plot.xlabel('Estimated AQI using PCR', fontsize=15)
plot.axis('equal')
plot.show()

import matplotlib.pyplot as plt
plt.plot(history.history['acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper left')
plt.show()

