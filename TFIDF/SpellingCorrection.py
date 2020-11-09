from spellchecker import SpellChecker
import pandas as pd
from Tokenizer import tokenizer

spell = SpellChecker()

Dataframe_Hewlett_essay = pd.read_excel('C:/Users/Raj/Desktop/UWA/Semester 3/Research project\Playing Around With Dataset/Dataset_small.xlsx','Sheet1')
Dataframe_Hewlett_essay_string = Dataframe_Hewlett_essay.to_string()
Dataframe_Hewlett_essay_string_lower = Dataframe_Hewlett_essay_string.lower()
Tokenwords = tokenizer(Dataframe_Hewlett_essay_string)

# for word in Tokenwords:
#     print(f'{word}:{spell.correction(word)}:probability {spell.word_probability(word)}')


list1 = []
# pipe text to pylanguagetool
for word in Tokenwords:
    list1.append(list(spell.unknown([word])))
    
    #print(spell.unknown([word]))
#Select[misspelled_words, UnsameQ[{}] &]

list2 = [x for x in list1 if x != []]
Missspelled_Words = [item for sublist in list2 for item in sublist]

for word in Missspelled_Words:
     print(f'{word}:{spell.correction(word)}:probability {spell.word_probability(word)}')