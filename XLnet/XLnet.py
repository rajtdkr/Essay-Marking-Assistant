import pandas as pd
from embedding_as_service.text.encode import Encoder

Dataset = pd.read_excel('C:/Users/Raj/Desktop/UWA/Semester 4/Research/1(a) completed.xlsx', 'Sheet1')
number_of_rows = len(Dataset)


def embedd(Answer):
    Answer = Answer.to_string()
    en = Encoder(embedding='xlnet', model='xlnet_large_cased', max_seq_length=256)
    vecs = en.encode(texts=[Answer])
    print(vecs)


for i in range(1, number_of_rows):
    Answer = pd.DataFrame(Dataset.iloc[i - 1:i, 0:1])
    # print(Answer)
    Marks = pd.DataFrame(Dataset.iloc[i - 1:i, 1:2])
    # print(Marks)
    khai = embedd(Answer)
