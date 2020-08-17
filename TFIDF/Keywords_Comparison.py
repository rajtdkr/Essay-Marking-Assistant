import os
import sys

# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath("C:/Users/Raj/Desktop/UWA/Semester 3/Research project/Playing Around With Dataset/TF-IDF/"))

# Do the import
from spellchecker import SpellChecker
from TFIDF import Tokenizer
from TFIDF import Stopwords_Remover
from TFIDF import Lemmtizer
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from nltk.corpus import wordnet
from flashtext import KeywordProcessor

#Dataframe_Hewlett_essay = pd.read_excel('C:/Users/Raj/Desktop/UWA/Semester 3/Research project\Playing Around With Dataset/Dataset.xlsx','Sheet1')
#Dataframe_Hewlett_essay_Check = pd.read_excel('C:/Users/Raj/Desktop/UWA/Semester 3/Research project\Playing Around With Dataset/Students_Data.xlsx','Sheet1')

def Preprocessing(Dataframe_Hewlett_essay):
    Dataframe_Hewlett_essay_string = Dataframe_Hewlett_essay.to_string() #Converting DF to string
    Dataframe_Hewlett_essay_string_lower = Dataframe_Hewlett_essay_string.lower() #Converting into lowercase
    Dataframe_Hewlett_essay_string_nospch = re.sub('[;:!*,/$()?]@', '', Dataframe_Hewlett_essay_string_lower) #Removing special Charecters
    Tokenwords = Tokenizer.tokenizer(Dataframe_Hewlett_essay_string) #Tokenizing the string
    NoStopwords_DF = Stopwords_Remover.stopwords_remover(Tokenwords) #Removing the stopwords
    Lemmetizer_String = Lemmtizer.Lemmaztizer_fn(NoStopwords_DF) #String Stemmer
    print("preprocessing_Completed")
    return Lemmetizer_String
    
    
def TFIDF(Stemmedwords):
    sentences = list()
    for l in re.split(r"\.\s|\?\s|\!\s|\n",Stemmedwords):
                if l:
                    sentences.append(l)
    cvec = CountVectorizer(stop_words='english', min_df=3, max_df=0.5, ngram_range=(1,2))
    sf = cvec.fit_transform(sentences)
    transformer = TfidfTransformer()
    transformed_weights = transformer.fit_transform(sf)
    weights = np.asarray(transformed_weights.mean(axis=0)).ravel().tolist()
    weights_df = pd.DataFrame({'term': cvec.get_feature_names(), 'weight': weights})
    Keywords = weights_df.sort_values(by='weight', ascending=False).head(10)
    print("TFIDF Completed")
    return Keywords


def Unstem(keywords): 
    #print(keywords)
    spell = SpellChecker()
    Unstem_list = []
    for word in keywords:
       Unstem_list.append(spell.correction(word))
        #print(f'{word}:{spell.correction(word)}:probability {spell.word_probability(word)}')
    #print(Unstem_list)
    print("Orginal Words Retrived")
    return Unstem_list

def Getmoresimilarwords(Unstemmed_words):
    #print(Unstemmed_words)
    Final_Keywords = []
    for i in Unstemmed_words:
       # print("It is for word", i)
        for syn in wordnet.synsets(i):
        	for l in syn.lemmas():
        		Final_Keywords.append(l.name())
    print("Values Retured")
    return(Final_Keywords)            
        
def keywords_Verification(Final_Keywords,Dataframe_Hewlett_essay_Check):
    print("Keyword Verfication Entered")
    Dataframe_Hewlett_essay_Check = Dataframe_Hewlett_essay_Check.to_string()
    keyword_processor = KeywordProcessor()
    # keyword_processor.add_keyword(<unclean name>, <standardised name>)
    for i in Final_Keywords:
        keyword_processor.add_keyword(i)
    keywords_found = keyword_processor.extract_keywords(Dataframe_Hewlett_essay_Check)
    return keywords_found




'''
Preprocessed_DF = Preprocessing(Dataframe_Hewlett_essay)
Keywords = TFIDF(Preprocessed_DF) 
keywords = Keywords.iloc[0:9,0]
Unstemmed_words = Unstem(keywords)
Final_Keywords = Getmoresimilarwords(Unstemmed_words)
Matching_keywords = keywords_Verification(Final_Keywords,Dataframe_Hewlett_essay_Check)


print(Matching_keywords)


'''