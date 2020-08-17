import os
import sys

# Add the directory containing your module to the Python path (wants absolute paths)
sys.path.append(os.path.abspath("C:/Users/Raj/Desktop/UWA/Semester 3/Research project/Playing Around With Dataset/TF-IDF/"))

# Do the import
from TFIDF import Tokenizer
from TFIDF import Stopwords_Remover
from TFIDF import Stemmer_File
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
Dataframe_Hewlett_essay = pd.read_excel('C:/Users/Raj/Desktop/UWA/Semester 3/Research project\Playing Around With Dataset/Dataset.xlsx','Sheet1')


def Preprocessing(Dataframe_Hewlett_essay):
    Dataframe_Hewlett_essay_string = Dataframe_Hewlett_essay.to_string() #Converting DF to string
    Dataframe_Hewlett_essay_string_lower = Dataframe_Hewlett_essay_string.lower() #Converting into lowercase
    Dataframe_Hewlett_essay_string_nospch = re.sub('[;:!*,/$()?]@', '', Dataframe_Hewlett_essay_string_lower) #Removing special Charecters
    Tokenwords = Tokenizer.tokenizer(Dataframe_Hewlett_essay_string) #Tokenizing the string
    NoStopwords_DF = Stopwords_Remover.stopwords_remover(Tokenwords) #Removing the stopwords
    Stemmed_String = Stemmer_File.Stemmer(NoStopwords_DF) #String Stemmer
    return Stemmed_String
    
    
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
    print(weights_df.sort_values(by='weight', ascending=False).head(10))
    
Preprocessed_DF = Preprocessing(Dataframe_Hewlett_essay)
TFIDF(Preprocessed_DF)
