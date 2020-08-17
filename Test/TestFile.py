
import pandas as pd
from embedding_as_service.text.encode import Encoder
from keras.layers import Dense
from keras.models import Sequential




Dataset = pd.read_excel('C:/Users/Raj/PycharmProjects/Essay-Marking-Assistant/Dataset/Typed Dataset/1(a) completed.xlsx', 'Sheet1')
number_of_rows = len(Dataset)
listofvecs = []

for i in range(8, 9):
    Answer = pd.DataFrame(Dataset.iloc[i - 1:i, 0:1])
    Marks = pd.DataFrame(Dataset.iloc[i - 1:i, 1:2])
    print("MARKSSSS",Marks,Answer)
    Answer = Answer.to_string()
    en = Encoder(embedding='xlnet', model='xlnet_large_cased', max_seq_length=256)
    vecs = en.encode(texts=[Answer])


print("Step 2 : Preprocessing the data")
listofvecs = []
list2 = []
array_1d = vecs.flatten()
listofvecs.append(array_1d)
list2.append(listofvecs)


model = Sequential()
model.add(Dense(1, input_dim=262144, kernel_initializer='normal', activation='relu'))
model.add(Dense(8, input_dim=8, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(list2, Marks, epochs=10000, verbose=0)


for i in range(2, 3):
    Answer = pd.DataFrame(Dataset.iloc[i - 1:i, 0:1])
    Marks = pd.DataFrame(Dataset.iloc[i - 1:i, 1:2])
    Answer = Answer.to_string()
    en = Encoder(embedding='xlnet', model='xlnet_large_cased', max_seq_length=256)
    vecs = en.encode(texts=[Answer])

print("Step 2 : Preprocessing the data")
listofvecs = []
list2 = []
array_1d = vecs.flatten()
listofvecs.append(array_1d)
list2.append(listofvecs)

ynew = model.predict(list2)

print(ynew)