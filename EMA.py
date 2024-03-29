import pandas as pd
from XLnetandBert import XLnetandBert
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
    Paraphrased_Sentence = Paraphrase.wordParaphrasing(TeachersData[0][0])
    TeachersData[0][0] = str(TeachersData[0][0]) + str(Paraphrased_Sentence)
    AnswerNumber = 18
    Input = 'y'
    while (Input == 'y'):
        DeepLearningTrained = XLnetandDeepLearning(TeachersData[0][0], TeachersData[0][1])
        print(DeepLearningTrained)
        #Input = input("Do you want to check answer? \n y for Yes and \n n for No")

        DeepLearningEvaluated = XLnetandDeepLearningEvaluate(DeepLearningTrained,StudentData[AnswerNumber][0])
        TFIDF_Words = TFIDF(TeachersData[0][0],StudentData[AnswerNumber][0])
        HandcraftedFeature = 0

        # HandcraftedFeature = HandcraftedFeatureFn(TeachersData[0][0],StudentData[AnswerNumber][0])
        DisplayOutput.DisplayOutput(TFIDF_Words,DeepLearningEvaluated,HandcraftedFeature,StudentData[AnswerNumber][0],AnswerNumber)
        #TeachersData[0][0] = str(TeachersData[0][0]) + str(StudentData[AnswerNumber][0])
       # TeachersData[0][1] = TeachersData[0][1] + StudentData[AnswerNumber][1]
        #print(TeachersData[0][0])
        AnswerNumber = AnswerNumber + 1


            # if AnswerNumber < len(StudentData):
            #     Input = input("Do you want to check another answer? \n y for Yes and n for No:")
            # else:
            #     exit()


def XLnetandDeepLearningEvaluate(DLModel_Trained,StudentData):

    #print("_______________DeepLearning______________")
    StudentSentences = []
    StudentSentencesMarks = []
    StudentSentences = StudentData.split('.')
    for i in range(0,len(StudentSentences)-1):
        if bool(StudentSentences[i].strip()) == True:
            Students_EmbeddedData = XLnetandBert.XLnetembeddings(StudentSentences[i])
            Marks = DLModel_Trained.predict(Students_EmbeddedData)
            StudentSentencesMarks.append(Marks)
        else:
            print("Found an empty String, Ignoring the emppty String")
    return StudentSentencesMarks



def XLnetandDeepLearning(TeachersAnswer, Full_Marks ):
    #print("The data is Training")
    Full_Marks_List = []
    Full_Marks_List.append(Full_Marks)
    Teachers_EmbeddedData = XLnetandBert.XLnetembeddings(TeachersAnswer[0][0])
    DLModel_Trained = DeepLearning.NeuralNetsTrain(Teachers_EmbeddedData, Full_Marks_List)
    return DLModel_Trained


def TFIDF(Dataset, StudentDataset):

    #print("___________________Keyword Finder________________________")
    #print("2. TF-IDF : Finding Keywords")
    Preprocessed_DF = Keywords_Comparison.Preprocessing(StudentDataset)
    Keywords = Keywords_Comparison.TFIDF(Preprocessed_DF)
    keywords = Keywords.iloc[0:9, 0]
    #print(keywords)
    #Unstemmed_words = Keywords_Comparison.Unstem(keywords)
    #Final_Keywords = Keywords_Comparison.Getmoresimilarwords(Unstemmed_words)
    #print(Final_Keywords)
    #Matching_keywords = Keywords_Comparison.keywords_Verification(Final_Keywords, StudentDataset)
    # print(Matching_keywords) #Returns Matched words that are important
    #print("2. Keywords Finding Complete")
    return keywords

def HandcraftedFeatureFn(Dataset, StudentDataset):

    #print("___________________Finding Other Features________________________")
    SpellingMistakes = HandcraftedFeatures.SpellingMistake(StudentDataset) # Written Word Sets that are wrong
    WordCount = HandcraftedFeatures.WordCount(Dataset,StudentDataset)#Returns Teachers with Students Percentage
    GrammarError = HandcraftedFeatures.GrammarCheck(Dataset) #Returns All Gramatical Errors

    return SpellingMistakes,WordCount,GrammarError


def Datasetloader():

    #Dataset = pd.read_excel('Dataset/Typed Dataset/4(c) Teachers Answer.xlsx', 'Sheet1')
    Dataset = pd.read_excel('Dataset/Typed Dataset/4(c) Teachers Answer.xlsx', 'Sheet1')

    number_of_rows = len(Dataset)
    return Dataset

def StudentDatasetLoader():

    studentDataset = pd.read_excel('Dataset/Typed Dataset/4(c) completed.xlsx')
    return studentDataset

if __name__ == "__main__":

    main()
