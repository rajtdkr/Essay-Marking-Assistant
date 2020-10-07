from sty import fg, bg, ef, rs
from sty import Style, RgbFg
from TFIDF import Tokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd
from openpyxl import load_workbook

def DisplayOutput(TFIDF,DeepLearning,HandcraftedFeatures,CheckedAnswer,AnswerNumber):

    TotalMarks = 0
    Similarities = []
    for i in DeepLearning:
        for j in i.tolist():
            for k in j:
                Similarities.append(k)
                TotalMarks = TotalMarks + k
    #print("Marks For Each Sentences" , Similarities)
    print("Total Marks Allocated is", ((TotalMarks/len(Similarities))*5))

    CheckedAnswerTokenized = Tokenizer.tokenizer(CheckedAnswer)

    for i in range(0, len(CheckedAnswerTokenized)):
        for j in TFIDF:
            if CheckedAnswerTokenized[i] == j:
                Colored_Keyword = ef.bold + j + rs.bold_dim
                CheckedAnswerTokenized[i] = Colored_Keyword

    CheckedAnswerKeywords = TreebankWordDetokenizer().detokenize(CheckedAnswerTokenized)

    CheckedAnswerSentences = []
    CheckedAnswerSentences = CheckedAnswerKeywords.split('.');



    for i,j in zip(range(len(CheckedAnswerSentences)),Similarities):

        if (j>0 and j<0.45):
            CheckedAnswerSentences[i] = fg.blue + CheckedAnswerSentences[i] + fg.rs
        if (j>0.45 and j < 0.55):
            CheckedAnswerSentences[i] = fg.yellow + CheckedAnswerSentences[i] + fg.rs
        if (j> 0.55 and j <1 ):
            CheckedAnswerSentences[i] = fg.green + CheckedAnswerSentences[i] + fg.rs

    final_output = ".".join(CheckedAnswerSentences)

    print(final_output)

    # if AnswerNumber == 0:
    #     listofanswers = []
    #     listofanswers.append(final_output)
    #     df2 = pd.DataFrame(listofanswers)
    #     with pd.ExcelWriter('output.xlsx') as writer:
    #         df2.to_excel(writer)
    # else:
    #     listofanswers.append(final_output)
    #     df2 = pd.DataFrame(listofanswers)
    #     with pd.ExcelWriter('output.xlsx') as writer:
    #         df2.to_excel(writer)




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