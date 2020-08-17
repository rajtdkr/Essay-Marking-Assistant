import pandas as pd
from XLnet import XLnet
from DeepLearning import Preprocessing
from DeepLearning import DeepLearning
from openpyxl import load_workbook
from TFIDF import Keywords_Comparison
from HandcraftedFeatures import HandcraftedFeatures

def main():
    loadedData = Datasetloader()
    StudentDataset = StudentDatasetLoader()


    #DeepLearningAnswers = XLnetandDeepLearning(loadedData)


    TFIDF_Score = TFIDF(loadedData,StudentDataset)


    HandcraftedFeature = HandcraftedFeatureFn(loadedData, StudentDataset)


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
        listofvecs = []
       # list2 = []
       # array_1d = PreprocessEmbeddedData.flatten()
        listofvecs.append(PreprocessEmbeddedData)
       #list2.append(EmbeddedData)
    #df = pd.DataFrame(listofvecs)
    print(listofvecs)

    #DLModel = DeepLearning.NeuralNetsTrain(df, Marks)


def TFIDF(Dataset , StudentDataset):
    print("3. TF-IDF : Finding Keywords")

    Preprocessed_DF = Keywords_Comparison.Preprocessing(Dataset)
    Keywords = Keywords_Comparison.TFIDF(Preprocessed_DF)
    keywords = Keywords.iloc[0:9, 0]
    Unstemmed_words = Keywords_Comparison.Unstem(keywords)
    Final_Keywords = Keywords_Comparison.Getmoresimilarwords(Unstemmed_words)
    Matching_keywords = Keywords_Comparison.keywords_Verification(Final_Keywords, StudentDataset)

    print(Matching_keywords) #Returns Matched words that are important


def HandcraftedFeatureFn(Dataset , StudentDataset):
    SpellingMistakes = HandcraftedFeatures.SpellingMistake(StudentDataset) # Written Word Sets that are wrong
    print(SpellingMistakes) #Returns words with potential Spelling Mistake

    WordCount = HandcraftedFeatures.WordCount(Dataset,StudentDataset)
    print(WordCount) #Returns Teachers with Students Percentage


def Datasetloader():
    Dataset = pd.read_excel('Dataset/Typed Dataset/1(a) completed.xlsx', 'Sheet1')
    number_of_rows = len(Dataset)
    return Dataset

def StudentDatasetLoader():
    studentDataset = pd.read_excel('Dataset/Typed Dataset/1(a) completed.xlsx')
    return studentDataset

if __name__ == "__main__":
    main()
