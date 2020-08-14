import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def stopwords_remover(text_tokens):
    #tokens_without_sw = [word for word in text_tokens if not word in set(stopwords.words())]
    my_stopwords = set(stopwords.words())
    tokens_without_sw1 = [word for word in text_tokens if not word in my_stopwords]
    #print(tokens_without_sw)
    print("_____________________________")
    #print(tokens_without_sw1)
    stopwords_added = ['use']
    tokens_without_sw = [word for word in tokens_without_sw1 if not word in stopwords_added]
    return tokens_without_sw
    
