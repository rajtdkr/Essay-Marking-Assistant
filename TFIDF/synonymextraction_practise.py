from nltk.corpus import wordnet
synonyms = []
antonyms = ['compute', 'people', 'family', 'help', 'think', 'caps', 'friend', 'time', 'thing']
for i in antonyms:
 for syn in wordnet.synsets("active"):
		for l in syn.lemmas():
			synonyms.append(l.name())           
print(synonyms)