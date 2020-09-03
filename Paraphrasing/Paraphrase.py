from TFIDF import Keywords_Comparison
from nltk.corpus import wordnet
from TFIDF import Tokenizer

def wordParaphrasing(Sentence):

    Preprocessed_DF = Keywords_Comparison.Preprocessing(Sentence)
    Keywords = Keywords_Comparison.TFIDF(Preprocessed_DF)

    #print(keywords)
    Final_Keywords = []
    for i in range(0,10):
        keywords = Keywords.iloc[i, 0]
        Final_Keywords.append('Another Word')
        for syn in wordnet.synsets(keywords):
        	for l in syn.lemmas():
        		Final_Keywords.append(l.name())


    TokenizedSentence = Tokenizer.tokenizer(Sentence)

    length = len(Final_Keywords)
    count_sentence = 0
    count_Keywords = 0
    for i in TokenizedSentence:
        count_sentence = count_sentence + 1
        for j in range(0,len(Final_Keywords)-1):
            count_Keywords = count_Keywords + 1
            if i == j:
                    if j != "Another Word" and Final_Keywords[count_Keywords+1]:

                        TokenizedSentence[count_sentence] = Final_Keywords[count_Keywords+1]

    print(TokenizedSentence)
    #Unstemmed_words = Keywords_Comparison.Unstem(keywords)
    #Final_Keywords = Keywords_Comparison.Getmoresimilarwords(Unstemmed_words)
    #print(Final_Keywords)