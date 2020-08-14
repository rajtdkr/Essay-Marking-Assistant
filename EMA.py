import pandas as pd
from XLnet import XLnet
from DeepLearning import Preprocessing
from DeepLearning import DeepLearning


def main():
    loadedData = Datasetloader()
    DeepLearningAnswers = XLnetandDeepLearning(loadedData)

def XLnetandDeepLearning(loadedData):
    #Length = len(loadedData)
    Length = 2
    list_of_Embedded_Data = []
    list_of_Marks = []
    for i in range(0,Length): #Length
        print("Embedding", i+1 ,"out of",Length , "Answers")
        Answer = pd.DataFrame(loadedData.iloc[i - 1:i, 0:1])
        Marks = pd.DataFrame(loadedData.iloc[i - 1:i, 1:2])
        EmbeddedData = XLnet.XLnetembeddings(Answer)
        PreprocessEmbeddedData = Preprocessing.Convert_into_2d(EmbeddedData)
        list_of_Embedded_Data.append(PreprocessEmbeddedData)
        list_of_Marks.append(Marks)
    #PreprocessEmbeddedDataAgain = Preprocessing.Convert_into_2d(list_of_Embedded_Data)
    #print(PreprocessEmbeddedDataAgain)
    DLModel = DeepLearning.NeuralNetsTrain(list_of_Embedded_Data,Marks)

def TFIDF():
    print("Need to Connect TF-IDF Part")


def Datasetloader():
    Dataset = pd.read_excel('C:/Users/Raj/Desktop/UWA/Semester 4/Research/1(a) completed.xlsx', 'Sheet1')
    number_of_rows = len(Dataset)
    return Dataset

if __name__ == "__main__":
     main()