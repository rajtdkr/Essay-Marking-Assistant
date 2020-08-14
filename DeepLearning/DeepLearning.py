import pandas as pd
from embedding_as_service.text.encode import Encoder
from keras.layers import Dense
from keras.models import Sequential





def NeuralNetsTrain(list2,Marks):

    Dimension = len(Marks)

    model = Sequential()
    model.add(Dense(1, input_dim=Dimension, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(list2, Marks, epochs=10000, verbose=0)

    #ynew = model.predict(list2)
    return model