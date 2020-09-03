import pandas as pd
from XLnet import XLnet
from DeepLearning import Preprocessing
from DeepLearning import DeepLearning
from openpyxl import load_workbook
from TFIDF import Keywords_Comparison
from HandcraftedFeatures import HandcraftedFeatures
from Paraphrasing import Paraphrase

def main():

    loadedData = Datasetloader()
    StudentDataset = StudentDatasetLoader()

    TeachersData = loadedData.values.tolist()
    StudentData = StudentDataset.values.tolist()

    #Paraphrased_Sentence = Paraphrase.wordParaphrasing(TeachersData[0][0])

    DeepLearningAnswers = XLnetandDeepLearning(TeachersData, StudentData)

    TFIDF_Score = TFIDF(TeachersData[0][0],StudentData[0][0])

    HandcraftedFeature = HandcraftedFeatureFn(TeachersData[0][0],StudentData[0][0])


def XLnetandDeepLearning(loadedData, StudentDataset):

    Full_Marks = []
    Marks = loadedData[0][1]

    Full_Marks.append(Marks)
    Teachers_EmbeddedData = XLnet.XLnetembeddings(loadedData[0][0])
    DLModel_Trained = DeepLearning.NeuralNetsTrain(Teachers_EmbeddedData, Full_Marks)

    StudentSentences = []
    StudentSentencesMarks = []
    StudentSentencesFinalMarks = []
    StudentDS = []
    for i in range(0,len(StudentDataset)):
        Str = StudentDataset[i][0]
        STRA = Str.split('.')
        StudentSentences.append(STRA)

    No_of_Answers_to_Check = len(StudentSentences)
    No_of_Answers_to_Check = 1
    for i in range(0,No_of_Answers_to_Check):
        length = len(StudentSentences[i])
        StudentSentencesFinalMarks.append(StudentSentencesMarks)
        StudentSentencesMarks = []
        for j in range(0,length):

            print("Embedding",j ,"sentence", i+1 ,"th answer out of",No_of_Answers_to_Check ,"student sentence")
            print(StudentSentences[i][j])

            if bool(StudentSentences[i][j].strip()) == True:
                print(StudentSentences[i][j].strip())
                print(bool(StudentSentences[i][j].strip()))
                Students_EmbeddedData = XLnet.XLnetembeddings(StudentSentences[i][j])
                print("Inserting into the model")
                Marks = DLModel_Trained.predict(Students_EmbeddedData)
                StudentSentencesMarks.append(Marks)
            else:
                print("Found an empty String")

    #FinalMarksDF = pd.DataFrame(StudentSentencesMarks)
    #FinalMarksDF.to_excel("C:/Users/Raj/Desktop/UWA/Semester 4/Research/OUTPUT", sheet_name='Sheet_name_1')
    print((DLModel_Trained))
    print("___________________DL Model________________________")
    print("Total Marks given by DL model", Marks*loadedData[0][1])
    print("Marks allocated based on each sentences", StudentSentencesMarks)

def TFIDF(Dataset, StudentDataset):
    print("___________________Keyword Finder________________________")
    print("2. TF-IDF : Finding Keywords")
    Preprocessed_DF = Keywords_Comparison.Preprocessing(Dataset)
    Keywords = Keywords_Comparison.TFIDF(Preprocessed_DF)
    keywords = Keywords.iloc[0:9, 0]
    Unstemmed_words = Keywords_Comparison.Unstem(keywords)
    Final_Keywords = Keywords_Comparison.Getmoresimilarwords(Unstemmed_words)
    Matching_keywords = Keywords_Comparison.keywords_Verification(Final_Keywords, StudentDataset)
    print(Matching_keywords) #Returns Matched words that are important


def HandcraftedFeatureFn(Dataset, StudentDataset):

    print("___________________Finding Other Features________________________")
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
