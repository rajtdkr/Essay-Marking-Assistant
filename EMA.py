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

    length = 2

    TeachersData = loadedData.values.tolist()
    StudentData = StudentDataset.values.tolist()

    DeepLearningAnswers = XLnetandDeepLearning(TeachersData[0][0],StudentData[0][0])

    TFIDF_Score = TFIDF(TeachersData[0][0],StudentData[0][0])

    HandcraftedFeature = HandcraftedFeatureFn(TeachersData[0][0],StudentData[0][0])


def XLnetandDeepLearning(loadedData,StudentDataset):
    # Length = len(loadedData)
    Length = 2
    list_of_Embedded_Data = []
    list_of_Marks = []

    for i in range(1, Length):  # Length

        print("Embedding", i, "out of", Length, "Answers")
        Answer = pd.DataFrame(loadedData.iloc[i - 1:i, 0:1])
        Marks = pd.DataFrame(loadedData.iloc[i - 1:i, 1:2])

        print(Answer)

        list_of_Marks.append(Marks)
        List1 = []
        EmbeddedData = XLnet.XLnetembeddings(Answer)
        PreprocessEmbeddedData = Preprocessing.Convert_into_2d(EmbeddedData)
        list_of_Embedded_Data.append(PreprocessEmbeddedData)
        List1.append(EmbeddedData)
   # DLModel = DeepLearning.NeuralNetsTrain(EmbeddedData, list_of_Marks)


def TFIDF(Dataset, StudentDataset):


    print("2. TF-IDF : Finding Keywords")
    Preprocessed_DF = Keywords_Comparison.Preprocessing(Dataset)
    Keywords = Keywords_Comparison.TFIDF(Preprocessed_DF)
    keywords = Keywords.iloc[0:9, 0]
    Unstemmed_words = Keywords_Comparison.Unstem(keywords)
    Final_Keywords = Keywords_Comparison.Getmoresimilarwords(Unstemmed_words)
    Matching_keywords = Keywords_Comparison.keywords_Verification(Final_Keywords, StudentDataset)
    print(Matching_keywords) #Returns Matched words that are important


def HandcraftedFeatureFn(Dataset, StudentDataset):

    #Length = len(StudentDataset)
    Length = 2


    #print("Teachers Dataset : ",Dataset.to_string())
    #print("Students Dataset : ",StudentDataset.to_string())

    SpellingMistakes = HandcraftedFeatures.SpellingMistake(StudentDataset) # Written Word Sets that are wrong
    print(len(SpellingMistakes)) #Returns words with potential Spelling Mistake
    #print(SpellingMistakes)

    WordCount = HandcraftedFeatures.WordCount(Dataset,StudentDataset)
    print(WordCount) #Returns Teachers with Students Percentage

    GrammarError = HandcraftedFeatures.GrammarCheck(Dataset) #Returns All Gramatical Errors
    print(GrammarError)

    return SpellingMistakes,WordCount,GrammarError


def Datasetloader():
    Dataset = pd.read_excel('Dataset/Typed Dataset/1(a) Teachers Answer.xlsx', 'Sheet1')
    number_of_rows = len(Dataset)
    return Dataset

def StudentDatasetLoader():
    studentDataset = pd.read_excel('Dataset/Typed Dataset/1(a) completed.xlsx')
    return studentDataset

if __name__ == "__main__":
    main()
