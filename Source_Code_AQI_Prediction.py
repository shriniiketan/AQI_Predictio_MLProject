import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score , confusion_matrix , mean_absolute_error , mean_absolute_percentage_error
import seaborn as sns 
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.models import Sequential
import tensorflow as tf



data = pd.read_csv('C:/AQI project/New_data_with_aqi_cleaned.csv')

print(data.head())

print(data.info())

print(data.isnull().sum())




data.drop(data[data.T.str.contains(r'[@#&$%+-/*]')
                 | data.TM.str.contains(r'[@#&$%+-/*]')
                 | data.Tm.str.contains(r'[@#&$%+-/*]')
                 | data.SLP.str.contains(r'[@#&$%+-/*]')
                 | data.H.str.contains(r'[@#&$%+-/*]')
                 | data.VV.str.contains(r'[@#&$%+-/*]')
                 | data.V.str.contains(r'[@#&$%+-/*]')
                 | data.VM.str.contains(r'[@#&$%+-/*]')].index)


data.drop(data[data['T'] == -100].index, inplace = True)
data.drop(data[data['TM'] == -100].index, inplace = True)
data.drop(data[data['Tm'] == -100].index, inplace = True)
data.drop(data[data['SLP'] == -100].index, inplace = True)
data.drop(data[data['H'] == -100].index, inplace = True)
data.drop(data[data['VV'] == -100].index, inplace = True)
data.drop(data[data['V'] == -100].index, inplace = True)
data.drop(data[data['VM'] == -100].index, inplace = True)

cleanedData = data.dropna()

print(cleanedData.isnull().sum())

print(cleanedData.head())

print(cleanedData.shape)

cleanedData.to_csv('C:/AQI project/New_data_with_aqi_cleaned.csv')

features = cleanedData

def calculate_pm2_aqi(value):
    value = int(value[0])
    I = 0
    if value >= 0 and value <= 30:
        Clow = 0
        Chigh = 30
        Ilow = 0
        Ihigh = 50
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 31 and value <= 60:
        Clow = 30
        Chigh = 60
        Ilow = 51
        Ihigh = 100
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 61 and value <= 90:
        Clow = 61
        Chigh = 90
        Ilow = 101
        Ihigh = 200
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 91 and value <= 120:
        Clow = 91
        Chigh = 120
        Ilow = 201
        Ihigh = 300
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 121 and value <= 250:
        Clow = 121
        Chigh = 250
        Ilow = 301
        Ihigh = 400
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value > 251 :
        Clow = 251
        Chigh = value
        Ilow = 401
        Ihigh = 500
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow
    return int(I)

'''
# calculating Air Quality Index using PM10

def calculate_pm10_aqi(value):
    value = int(value[0])
    if value >= 0 and value <= 50:
        Clow = 0
        Chigh = 50
        Ilow = 0
        Ihigh = 50
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 51 and value <= 100:
        Clow = 51
        Chigh = 100
        Ilow = 51
        Ihigh = 100
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 101 and value <= 250:
        Clow = 101
        Chigh = 250
        Ilow = 101
        Ihigh = 200
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 251 and value <= 350:
        Clow = 251
        Chigh = 350
        Ilow = 201
        Ihigh = 300
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 351 and value <= 430:
        Clow = 351
        Chigh = 430
        Ilow = 301
        Ihigh = 400
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value > 430 :
        Clow = 430
        Chigh = value
        Ilow = 401
        Ihigh = 500
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow
    return int(I)
'''

# calculate AQI using CO


def calculate_CO_aqi(value):
    value = value[0]
    I = 0
    if value >= 0 and value <= 1:
        Clow = 0
        Chigh = 1
        Ilow = 0
        Ihigh = 50
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value > 1.0 and value <= 2.0:
        Clow = 1.1
        Chigh = 2.0
        Ilow = 51
        Ihigh = 100
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 2.1 and value < 10:
        Clow = 2.1
        Chigh = 10
        Ilow = 101
        Ihigh = 200
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 10.1 and value < 17:
        Clow = 10.1
        Chigh = 17
        Ilow = 201
        Ihigh = 300
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 17 and value < 34:
        Clow = 17
        Chigh = 34
        Ilow = 301
        Ihigh = 400
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 34 :
        Clow = 34
        Chigh = value
        Ilow = 401
        Ihigh = 500
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow
    return int(I)


# calculate AQI using SO2


def calculate_SO2_aqi(value):
    value = int(value[0])
    if value >= 0 and value <= 40:
        Clow = 0
        Chigh = 40
        Ilow = 0
        Ihigh = 50
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 41 and value <= 80:
        Clow = 41
        Chigh = 80
        Ilow = 51
        Ihigh = 100
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 81 and value <= 380:
        Clow = 81
        Chigh = 380
        Ilow = 101
        Ihigh = 200
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 381 and value <= 800:
        Clow = 381
        Chigh = 800
        Ilow = 201
        Ihigh = 300
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 801 and value < 1600:
        Clow = 801
        Chigh = 1600
        Ilow = 301
        Ihigh = 400
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 1600 :
        Clow = 1600
        Chigh = value
        Ilow = 401
        Ihigh = 500
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow
    return int(I)


# calculate AQI using NH3

'''
def calculate_NH3_aqi(value):
    value = value[0]
    if value >= 0 and value <= 200:
        Clow = 0
        Chigh = 200
        Ilow = 0
        Ihigh = 50
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 201 and value <= 400:
        Clow = 201
        Chigh = 400
        Ilow = 51
        Ihigh = 100
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 401 and value <= 800:
        Clow = 401
        Chigh = 800
        Ilow = 101
        Ihigh = 200
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 801 and value < 1200:
        Clow = 801
        Chigh = 1200
        Ilow = 201
        Ihigh = 300
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 1200 and value <= 1800:
        Clow = 1201
        Chigh = 1800
        Ilow = 301
        Ihigh = 400
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value > 1800 :
        Clow = 1800
        Chigh = value
        Ilow = 401
        Ihigh = 500
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow
    return int(I)
'''

# calculate AQI using NO2


def calculate_NO2_aqi(value):

    value = int(value[0])
    if value >= 0 and value <= 40:
        Clow = 0
        Chigh = 40
        Ilow = 0
        Ihigh = 50
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 41 and value <= 80:
        Clow = 41
        Chigh = 80
        Ilow = 51
        Ihigh = 100
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 81 and value <= 180:
        Clow = 81
        Chigh = 180
        Ilow = 101
        Ihigh = 200
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 181 and value <= 280:
        Clow = 181
        Chigh = 280
        Ilow = 201
        Ihigh = 300
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 281 and value <= 400:
        Clow = 281
        Chigh = 400
        Ilow = 301
        Ihigh = 400
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value > 400 :
        Clow = 400
        Chigh = value
        Ilow = 401
        Ihigh = 500
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow
    return int(I)

# calculate AQI using O3


def calculate_O3_aqi(value):

    value = int(value[0])
    if value >= 0 and value <= 50:
        Clow = 0
        Chigh = 50
        Ilow = 0
        Ihigh = 50
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 51 and value <= 100:
        Clow = 51
        Chigh = 100
        Ilow = 51
        Ihigh = 100
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 101 and value <= 168:
        Clow = 101
        Chigh = 168
        Ilow = 101
        Ihigh = 200
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 169 and value <= 208:
        Clow = 169
        Chigh = 208
        Ilow = 201
        Ihigh = 300
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value >= 209 and value <= 748:
        Clow = 209
        Chigh = 748
        Ilow = 301
        Ihigh = 400
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow

    elif value > 748 :
        Clow = 748
        Chigh = value
        Ilow = 401
        Ihigh = 500
        C = value
        I = (((Ihigh - Ilow)/(Chigh - Clow))*(C - Clow)) + Ilow
    return int(I)



import pandas as pd
import csv



def calculating_aqi(data_without_missing_values):
    pm2_data = pd.read_csv(data_without_missing_values, usecols=["PM2.5 (ug/m3)"]).values.tolist()
    no2_data = pd.read_csv(data_without_missing_values, usecols=["NO2 (ug/m3)"]).values.tolist()
    ozone_data = pd.read_csv(data_without_missing_values, usecols=["Ozone (ug/m3)"]).values.tolist()
    so2_data = pd.read_csv(data_without_missing_values, usecols=["SO2 (ug/m3)"]).values.tolist()
    aqi = []
    for i in range(len(pm2_data)):
        column = []
        single_column = calculate_pm2_aqi(pm2_data[i])
        column.append(single_column)
        single_column = calculate_SO2_aqi(so2_data[i])
        column.append(single_column)
        single_column = calculate_NO2_aqi(no2_data[i])
        column.append(single_column)
        single_column = calculate_O3_aqi(ozone_data[i])
        column.append(single_column)
        aqi.append(column)
    return aqi


def creating_csv(filename,header,data_with_aqi): # creating a csv file with Air Quality Index data in it
    with open(filename, "r") as my_csv:
        csv_reader = csv.reader(my_csv)
        data = []
        i=0
        for line in csv_reader:
            if i is 0:
                i += 1
            else:
                data.append(line)

        with open(data_with_aqi, "w") as aqi_file:
            csv_writer = csv.writer(aqi_file, dialect="excel")
            csv_writer.writerow(header)
            i = 0
            aqi = calculating_aqi(filename)
            for single_row in data:
                row = []
                for column in single_row:
                    row.append(column)
                row.append(max(aqi[i]))
                if 0 <= max(aqi[i]) <= 100:
                    row.append(1)
                elif 101 <= max(aqi[i]) <= 200:
                    row.append(2)
                elif 201 <= max(aqi[i]) <= 300:
                    row.append(3)
                elif 301 <= max(aqi[i]):
                    row.append(4)
                i += 1
                csv_writer.writerow(row)

header_for_pollutant = ['From Date','To Date','PM10 (ug/m3)','PM2.5 (ug/m3)','CO (mg/m3)',
                        'NH3 (ug/m3)','NO2 (ug/m3)','Ozone (ug/m3)','SO2 (ug/m3)']
path_for_pollutant_data = "C:/AQI project/"
header_for_metereological = ["From Date","To Date","T","TM","Tm","SLP","H","PP","VV","V","VM"]
path_for_metereological_data = "Data/met_data/"

header = header_for_pollutant
header.append('Air_Quality_Index(AQI)')
header.append('AQI Category')
path = path_for_pollutant_data
data_without_missing_values = path + "AQI_new_for_index.csv"
data_with_aqi = path + "data_with_aqi.csv"
creating_csv(data_without_missing_values,header,data_with_aqi)

features = data

features = features.drop(['AQI Category'],axis=1)

print(features.shape)
print(features.head())

label = features['Air_Quality_Index(AQI)']

print(label.head())

print(label.shape)

features = features.drop(['Air_Quality_Index(AQI)'],axis=1)

print(features.head())

features.to_csv('C:/AQI project/feature_data.csv')

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

model = Sequential()

'''
model.add(Dense(256,activation='relu',input_shape = X_train.shape[1:]))

model.add(Dense(128,activation='relu'))

model.add(Dense(64,activation='relu'))

model.add(Dense(32,activation='relu'))

model.add(Dense(1,activation='linear'))

model.compile(loss='mse',optimizer='adam')

model.summary()


'''



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



'''
not in use

model = LinearRegression()

'''


model.fit(X_train,Y_train,epochs=500)

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





import seaborn as sns
corrmat = feature_data.corr()
top_corr_features = corrmat.index
g=sns.heatmap(feature_data[top_corr_features].corr(),annot=True,cmap="RdYlGn")

