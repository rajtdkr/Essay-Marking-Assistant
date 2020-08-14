import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def tokenizer(Dataframe_Hewlett_essay1_string_nospch):
    text_tokens = word_tokenize(Dataframe_Hewlett_essay1_string_nospch)
    return text_tokens
