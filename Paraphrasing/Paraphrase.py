from TFIDF import Keywords_Comparison
from nltk.corpus import wordnet
from TFIDF import Tokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer

def wordParaphrasing(Sentence):

    Preprocessed_DF = Keywords_Comparison.Preprocessing(Sentence)
    Keywords = Keywords_Comparison.TFIDF(Preprocessed_DF)

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

    Length = len(Final_Keywords)

    ParaphrasedAnswer = []

    for i in TokenizedSentence:
        count = -1
        for j in Final_Keywords:
            count = count + 1
            if i == j and Length > count-1:
                #print(Length,count)
                i = Final_Keywords[count+1]
                #print(i,"needs to be replaced by",Final_Keywords[count+1])
                break
        ParaphrasedAnswer.append(i)

    ParaphrasedAnswer_Final = TreebankWordDetokenizer().detokenize(ParaphrasedAnswer)

    return ParaphrasedAnswer_Final
