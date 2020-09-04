import pandas as pd
from XLnet import XLnet
from DeepLearning import Preprocessing
from DeepLearning import DeepLearning
from openpyxl import load_workbook
from TFIDF import Keywords_Comparison
from HandcraftedFeatures import HandcraftedFeatures
from Paraphrasing import Paraphrase
from OutputDisplay import DisplayOutput

def main():

    loadedData = Datasetloader()
    StudentDataset = StudentDatasetLoader()
    TeachersData = loadedData.values.tolist()
    StudentData = StudentDataset.values.tolist()

    print("The data is Training")
    Paraphrased_Sentence = Paraphrase.wordParaphrasing(TeachersData[0][0])
    TeachersData[0][0] = str(TeachersData[0][0]) + str(Paraphrased_Sentence)
    DeepLearningTrained = XLnetandDeepLearning(TeachersData[0][0], TeachersData[0][1])

    Input = input("Do you want to check answer? \n y for Yes and \n n for No")

    AnswerNumber = 0
    while(Input == 'y'):

        DeepLearningEvaluate = XLnetandDeepLearningEvaluate(DeepLearningTrained,StudentData[AnswerNumber][0])
        TFIDF_Words = TFIDF(TeachersData[0][0],StudentData[AnswerNumber][0])
        HandcraftedFeature = HandcraftedFeatureFn(TeachersData[0][0],StudentData[AnswerNumber][0])
        DisplayOutput.DisplayOutput(0,0,0)
        AnswerNumber = AnswerNumber + 1

        if AnswerNumber < len(StudentData):
            Input = input("Do you want to check another answer? \n y for Yes and \n n for No")
        else:
            exit()


def XLnetandDeepLearningEvaluate(DLModel_Trained,StudentData):

    print("_______________DeepLearning______________")
    StudentSentences = []
    StudentSentencesMarks = []
    StudentSentences = StudentData.split('.')
    for i in range(0,len(StudentSentences)):
        if bool(StudentSentences[i].strip()) == True:
            Students_EmbeddedData = XLnet.XLnetembeddings(StudentSentences[i])
            Marks = DLModel_Trained.predict(Students_EmbeddedData)
            StudentSentencesMarks.append(Marks)
        else:
            print("Found an empty String")
    return StudentSentencesMarks



def XLnetandDeepLearning(TeachersAnswer, Full_Marks ):

    Full_Marks_List = []
    Full_Marks_List.append(Full_Marks)
    Teachers_EmbeddedData = XLnet.XLnetembeddings(TeachersAnswer[0][0])
    DLModel_Trained = DeepLearning.NeuralNetsTrain(Teachers_EmbeddedData, Full_Marks_List)
    return DLModel_Trained

def TFIDF(Dataset, StudentDataset):

    print("___________________Keyword Finder________________________")
    #print("2. TF-IDF : Finding Keywords")
    Preprocessed_DF = Keywords_Comparison.Preprocessing(Dataset)
    Keywords = Keywords_Comparison.TFIDF(Preprocessed_DF)
    keywords = Keywords.iloc[0:9, 0]
    Unstemmed_words = Keywords_Comparison.Unstem(keywords)
    Final_Keywords = Keywords_Comparison.Getmoresimilarwords(Unstemmed_words)
    Matching_keywords = Keywords_Comparison.keywords_Verification(Final_Keywords, StudentDataset)
    # print(Matching_keywords) #Returns Matched words that are important

    return Matching_keywords

def HandcraftedFeatureFn(Dataset, StudentDataset):

    print("___________________Finding Other Features________________________")
    SpellingMistakes = HandcraftedFeatures.SpellingMistake(StudentDataset) # Written Word Sets that are wrong
    WordCount = HandcraftedFeatures.WordCount(Dataset,StudentDataset)#Returns Teachers with Students Percentage
    GrammarError = HandcraftedFeatures.GrammarCheck(Dataset) #Returns All Gramatical Errors

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
