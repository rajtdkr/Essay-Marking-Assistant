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


    TeachersData = loadedData.values.tolist()
    StudentData = StudentDataset.values.tolist()


    DeepLearningAnswers = XLnetandDeepLearning(TeachersData,StudentData[0][0])

    #TFIDF_Score = TFIDF(TeachersData[0][0],StudentData[0][0])

    #HandcraftedFeature = HandcraftedFeatureFn(TeachersData[0][0],StudentData[0][0])


def XLnetandDeepLearning(loadedData,StudentDataset):
    # Length = len(loadedData)
    Full_Marks = []
    Marks = loadedData[0][1]
    Full_Marks.append(Marks)
    Teachers_EmbeddedData = XLnet.XLnetembeddings(loadedData[0][0])
    DLModel_Trained = DeepLearning.NeuralNetsTrain(Teachers_EmbeddedData, Full_Marks)
    Students_EmbeddedData = XLnet.XLnetembeddings(StudentDataset[0][0])
    Marks = DLModel_Trained.predict(Students_EmbeddedData)

    print(Marks*loadedData[0][1])

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
