from spellchecker import SpellChecker
from TFIDF import Keywords_Comparison
from TFIDF import Tokenizer
import language_check

def SpellingMistake(StudentDataset):

    print("3.1. HandCrafted Features : Checking Spelling Mistake")

    Keywords = Keywords_Comparison.Preprocessing(StudentDataset)
    Tokens = Tokenizer.tokenizer(Keywords)
    spell = SpellChecker()
    ListofUnknownWords = []
    ListofUnknownWordsFinal = []

    for i in Tokens:
        UnknownWord = spell.unknown([i])
        ListofUnknownWords.append(UnknownWord)

    for i in ListofUnknownWords:
        if i != set():
            ListofUnknownWordsFinal.append(i)
    return ListofUnknownWordsFinal

def WordCount(Dataset, StudentDataset):

    print("3.2. HandCrafted Features : Checking Word Counts")
    Keywords = Keywords_Comparison.Preprocessing(StudentDataset)
    Tokens_Students = Tokenizer.tokenizer(Keywords)

    Keywords = Keywords_Comparison.Preprocessing(Dataset)
    Tokens_Teacher = Tokenizer.tokenizer(Keywords)

    PercentagetoTarget = (len(Tokens_Teacher)/len(Tokens_Students)) * 100

    return (PercentagetoTarget)


def GrammarCheck(fin):

    print("3.2. HandCrafted Features : Checking Grammar Mistakes")

    tool = language_check.LanguageTool('en-US')
    i = 0

    for line in fin:
        matches = tool.check(line)
        i = i + len(matches)
        pass

    #print(matches)

    return matches

