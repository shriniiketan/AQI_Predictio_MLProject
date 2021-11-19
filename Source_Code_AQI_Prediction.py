import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , confusion_matrix
import seaborn as sns 
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential
import tensorflow as tf



data = pd.read_csv('C:/AQI project/city_day.csv')

print(data.head())

print(data.info())

print(data.isnull().sum())

print(data['AQI'].value_counts())


data = data.drop(['PM10','NH3','Xylene','Toluene'], axis = 1)


cleanedData = data.dropna()

print(cleanedData.isnull().sum())

print(cleanedData.head())

print(cleanedData.shape)

CityData = cleanedData.loc[cleanedData['City'] == 'Mumbai']

print(CityData.isnull().sum())

print(CityData.head())

print(CityData.shape)

features = CityData

features = features.drop(['Date','City','AQI','AQI_Bucket'],axis=1)

print(features.head())

label = CityData['AQI']

print(label.head())

X_train,X_Test,Y_train,Y_test = train_test_split(features,label,test_size=0.25)

print(X_train.shape,Y_train.shape)

model = Sequential()


model.add(Dense(500,activation='relu',input_shape = X_train.shape[1:]))

model.add(Dropout(0.05))

model.add(Dense(250,activation='relu'))

model.add(Dropout(0.02))

model.add(Dense(125,activation='relu'))

model.add(Dense(50,activation='relu'))

model.add(Dense(1,activation='linear'))

model.compile(loss='mse',optimizer='adam')

model.summary()


'''

not in use 


model.add(Dense(1024,activation='relu',input_shape = X_train.shape[1:]))
      
model.add(Dropout(0.2))

model.add(Dense(512,activation='relu'))
        
model.add(Dropout(0.2))

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(128,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(64,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(16,activation='relu'))

model.add(Dense(8,activation='relu'))

model.add(Dense(4,activation='relu'))

model.add(Dense(1,activation='linear'))

model.compile(loss='mse',optimizer='adam')

model.summary()
'''


'''
not in use

model = LinearRegression()

'''

model.fit(X_train,Y_train,epochs=200)

Y_pred = model.predict(X_Test)

r2 = r2_score(Y_test,Y_pred)

print(r2)




