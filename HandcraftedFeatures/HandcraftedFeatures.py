from spellchecker import SpellChecker
from TFIDF import Keywords_Comparison
from TFIDF import Tokenizer

def SpellingMistake(StudentDataset):

    print("3. HandCrafted Features : Checking SpellingMistake")

    Keywords = Keywords_Comparison.Preprocessing(StudentDataset)
    Tokens = Tokenizer.tokenizer(Keywords)
    print(Tokens)
    spell = SpellChecker()
    ListofUnknownWords = []

    ListofUnknownWordsFinal = []
    for i in Tokens:
        UnknownWord = spell.unknown([i])
        ListofUnknownWords.append(UnknownWord)

   # print(len(ListofUnknownWords))
    for i in ListofUnknownWords:
        if i != set():
            ListofUnknownWordsFinal.append(i)

    return ListofUnknownWordsFinal

def WordCount(Dataset, StudentDataset):
    print("3. HandCrafted Features : Checking Word Counts")
    Keywords = Keywords_Comparison.Preprocessing(StudentDataset)
    Tokens_Students = Tokenizer.tokenizer(Keywords)
    #print(len(Tokens_Students))

    Keywords = Keywords_Comparison.Preprocessing(Dataset)
    Tokens_Teacher = Tokenizer.tokenizer(Keywords)
    #print(len(Tokens_Teacher))

    PercentagetoTarget = (len(Tokens_Teacher)/len(Tokens_Students)) * 100

    return (PercentagetoTarget)


def GrammarCheck():
    print("Checking Grammar Here")