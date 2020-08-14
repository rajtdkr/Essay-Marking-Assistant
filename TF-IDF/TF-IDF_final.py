# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:28:47 2020

@author: Raj
"""

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import word_tokenize 
import re

print("Executing part 1")
Dataframe_Hewlett_essay1 = pd.read_excel('C:/Users/Raj/Desktop/UWA/Semester 3/Research project\Playing Around With Dataset/Dataset.xlsx','Sheet1')
Dataframe_Hewlett_essay1_string = Dataframe_Hewlett_essay1.to_string()
Dataframe_Hewlett_essay1_string_lower = Dataframe_Hewlett_essay1_string.lower()
print("part 1 - completed")

print("Executing part 2")
Dataframe_Hewlett_essay1_string_nospch = re.sub('[;:!*,/$()?]@', '', Dataframe_Hewlett_essay1_string_lower)
#print(Dataframe_Hewlett_essay1_string_nospch)

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
text_tokens = word_tokenize(Dataframe_Hewlett_essay1_string_nospch)
#print(text_tokens)

print("Executing part 3")
#tokens_without_sw = [word for word in text_tokens if not word in set(stopwords.words())]
my_stopwords = set(stopwords.words())
tokens_without_sw1 = [word for word in text_tokens if not word in my_stopwords]
#print(tokens_without_sw)
print("_____________________________")
print(tokens_without_sw1)
stopwords_added = ['use']
tokens_without_sw = [word for word in tokens_without_sw1 if not word in stopwords_added]


print("part 3 - completed")


print("Executing part 4")
ps = PorterStemmer()
a = " "
b = " "
ps = PorterStemmer()
Stemmedwords = " ".join([ps.stem(w) for w in tokens_without_sw])

print(Stemmedwords)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


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

