from tensorflow.keras.backend import batch_dot, mean
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential


X=pd.read_csv('10_x.csv')
Y=pd.read_csv('10_y.csv')

X_train, X_test, y_train, y_test = train_test_split(X,Y , test_size=0.2,train_size=0.8, random_state=21)

model = Sequential()
model.add(Dense(9, input_dim = X_train.shape[1], kernel_initializer='normal', activation='relu'))
model.add(Dense(1,kernel_initializer='normal', activation='sigmoid'))

early_stop = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=10)


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=15, batch_size=16, validation_data=(X_test, y_test))


prediction = model.predict(X_test)


rmse = mean_squared_error(y_test, prediction)


text_file = open("metrics.txt", "w")
n = text_file.write(f"mean squared error: {rmse}")
text_file.close()


model.save('vgsales_model_dvc.h5')