from nltk.stem import PorterStemmer

def Stemmer(tokens_without_sw):
    
    ps = PorterStemmer()
    a = " "
    b = " "
    ps = PorterStemmer()
    Stemmedwords = " ".join([ps.stem(w) for w in tokens_without_sw])
    return Stemmedwords
