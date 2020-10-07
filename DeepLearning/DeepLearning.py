import pandas as pd
from embedding_as_service.text.encode import Encoder
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Bidirectional
from keras.layers import LSTM, Dropout
from keras import Model
import keras
from keras import layers


def NeuralNetsTrain(list2,Marks):
    model = Sequential()
    #embedding = Embedding(1024, 64, 2, name='embedding')
    #model.add(embedding)
    model.add(LSTM(64,input_shape=(256,1024), dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
    model.add(LSTM(64, dropout=0.4, recurrent_dropout=0.4, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(32, dropout=0.4, recurrent_dropout=0.4))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    model.fit(list2 , Marks, batch_size=3, epochs=2)
    #model.summary()

    return model