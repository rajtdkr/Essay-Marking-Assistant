import pandas as pd
from XLnet import XLnet
from DeepLearning import Preprocessing
from DeepLearning import DeepLearning
from openpyxl import load_workbook


def main():
    loadedData = Datasetloader()
    DeepLearningAnswers = XLnetandDeepLearning(loadedData)


def XLnetandDeepLearning(loadedData):
    # Length = len(loadedData)
    Length = 2
    list_of_Embedded_Data = []
    list_of_Marks = []
    for i in range(0, Length):  # Length
        print("Embedding", i + 1, "out of", Length, "Answers")
        Answer = pd.DataFrame(loadedData.iloc[i - 1:i, 0:1])
        Marks = pd.DataFrame(loadedData.iloc[i - 1:i, 1:2])
        EmbeddedData = XLnet.XLnetembeddings(Answer)
        # print(EmbeddedData)
        PreprocessEmbeddedData = Preprocessing.Convert_into_2d(EmbeddedData)
        # print(PreprocessEmbeddedData)
        print(EmbeddedData)
        listofvecs = []
       # list2 = []
       # array_1d = PreprocessEmbeddedData.flatten()
        listofvecs.append(EmbeddedData)
       #list2.append(EmbeddedData)

    DLModel = DeepLearning.NeuralNetsTrain(listofvecs, Marks)


def TFIDF():
    print("Need to Connect TFIDF Part")


def Datasetloader():
    Dataset = pd.read_excel('Dataset/Typed Dataset/1(a) completed.xlsx', 'Sheet1')
    number_of_rows = len(Dataset)
    return Dataset


if __name__ == "__main__":
    main()
