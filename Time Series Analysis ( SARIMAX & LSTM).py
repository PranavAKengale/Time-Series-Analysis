import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
%matplotlib inline

df=pd.read_csv(r'C:\Users\prana\Downloads\Python data science\Untitled Folder\Time Series\ARIMA-And-Seasonal-ARIMA-master\perrin-freres-monthly-champagne-.csv')

df.head()

## Cleaning up the data
df.columns=["Month","Sales"]
df.head()

df.shape

## Drop last 2 rows
df.drop(106,axis=0,inplace=True)

df.drop(105,axis=0,inplace=True)

# Convert Month into Datetime
df['Month']=pd.to_datetime(df['Month'])

df.set_index('Month',inplace=True)

df.describe()

df.plot()

### Testing For Stationarity

from statsmodels.tsa.stattools import adfuller

test_result=adfuller(df['Sales'])

#Ho: It is non stationary
#H1: It is stationary

def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations Used']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )
    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data has no unit root and is stationary")
    else:
        print("weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")

adfuller_test(df['Sales'])

Differencing

df['Sales First Difference'] = df['Sales'] - df['Sales'].shift(1)

df['Seasonal First Difference']=df['Sales']-df['Sales'].shift(12)

df.head(5)

df.shape

adfuller_test(df['Seasonal First Difference'].dropna())

df['Seasonal First Difference'].plot()

#from pandas.tools.plotting import autocorrelation_plot
#autocorrelation_plot(df['Sales'])
#plt.show()

from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

N, M = 14, 6
fig, ax = plt.subplots(figsize=(N, M))
plot_pacf(df['Sales'], lags = 25,ax=ax)
plt.show()

import statsmodels.api as sm



df.iloc[:90]

train=df.iloc[:90]
test=df.iloc[90:]

model=sm.tsa.statespace.SARIMAX(train['Sales'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()

df['forecast']=results.predict(start=90,end=105,dynamic=True)
df[['Sales','forecast']].plot(figsize=(12,8))

df1=df.dropna()

from math import sqrt
from sklearn.metrics import mean_squared_error
df1=df.dropna()
rmse = sqrt(mean_squared_error(df1['Sales'],df1['forecast']))
print('RMSE: %.3f' % rmse)


from sklearn.metrics import r2_score
r2_score(df1['Sales'], df1['forecast'],multioutput='variance_weighted')

df1



from pandas.tseries.offsets import DateOffset
future_dates=[df.index[-1]+ DateOffset(months=x)for x in range(0,24)]

future_datest_df=pd.DataFrame(index=future_dates[1:],columns=df.columns)

future_datest_df.tail()

future_df=pd.concat([df,future_datest_df])

future_df['forecast'] = results.predict(start = 104, end = 120, dynamic= True)  
future_df[['Sales', 'forecast']].plot(figsize=(12, 8)) 

# LSTM

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()

train_1=train.drop(['Sales First Difference','Seasonal First Difference'],axis=1)

test_1=test.drop(['Sales First Difference','Seasonal First Difference'],axis=1)

scaled_train=scaler.fit_transform(train_1)
scaled_test=scaler.transform(test_1)

from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

## Define generator
n_input=12
n_features=1
generator=TimeseriesGenerator(scaled_train,scaled_train,length=n_input,batch_size=1)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM


model=Sequential()
model.add(LSTM(100,activation='relu',input_shape=(n_input,n_features)))
model.add(Dense(1))
model.compile(optimizer='adam',loss='mse')

model.fit(generator,epochs=20)

loss_per_epoch=model.history.history['loss']

plt.plot(loss_per_epoch)

test_predictions=[]
first_batch=scaled_train[-n_input:]
current_batch=first_batch.reshape(1,12,1)
for i in range(len(test)):
    current_pred=model.predict(current_batch)[0]
    
    test_predictions.append(current_pred)
    
    current_batch=np.append(current_batch[:,1:,:],[[current_pred]],axis=1)

test_1['Prediction_Lstm']=scaler.inverse_transform(test_predictions)

from math import sqrt
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(test_1['Sales'],test_1['Prediction_Lstm']))
print('RMSE: %.3f' % rmse)


test_1.plot(figsize=(12,8))

from sklearn.metrics import r2_score
r2_score(test_1['Sales'], test_1['Prediction_Lstm'],multioutput='variance_weighted')
