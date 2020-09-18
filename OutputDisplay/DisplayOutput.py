from sty import fg, bg, ef, rs
from sty import Style, RgbFg
from TFIDF import Tokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer


def DisplayOutput(TFIDF,DeepLearning,HandcraftedFeatures,CheckedAnswer):


    Similarities = []
    for i in DeepLearning:
        for j in i.tolist():
            for k in j:
                Similarities.append(k)
    print("Marks For Each Sentences" , Similarities)
   # print(HandcraftedFeatures)

    CheckedAnswerTokenized = Tokenizer.tokenizer(CheckedAnswer)
    #print(CheckedAnswerTokenized)
    #print("The import that words are",TFIDF)
    for i in range(0, len(CheckedAnswerTokenized)):
        for j in TFIDF:
            if CheckedAnswerTokenized[i] == j:
                Colored_Keyword = ef.bold + j + rs.bold_dim
                CheckedAnswerTokenized[i] = Colored_Keyword



    print(TreebankWordDetokenizer().detokenize(CheckedAnswerTokenized))






#    foo = fg.red + "The Keywords Are" + fg.rs
#    print(foo)
#    print(bar)
#    bar = bg.blue + 'This has a blue background!' + bg.rs
#    baz = ef.italic + 'This is italic text' + rs.italic
#    qux = fg(201) + 'This is pink text using 8bit colors' + fg.rs
#    qui = fg(255, 10, 10) + 'This is red text using 24bit colors.' + fg.rs

#    fg.orange = Style(RgbFg(255, 150, 50))
#    buf = fg.orange + 'Yay, Im orange.' + fg.rs
#    print(foo, bar, baz, qux, qui, buf, sep='\n')