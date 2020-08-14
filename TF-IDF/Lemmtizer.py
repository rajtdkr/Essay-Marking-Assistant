from nltk.stem import WordNetLemmatizer 

def Lemmaztizer_fn(tokens_without_sw):
    
    Lemmetizer = WordNetLemmatizer()
    
    Lemmaztized_words = " ".join([Lemmetizer.lemmatize(w) for w in tokens_without_sw])
    return Lemmaztized_words
