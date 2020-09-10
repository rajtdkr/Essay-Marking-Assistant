from sty import fg, bg, ef, rs
from sty import Style, RgbFg


def DisplayOutput(TFIDF,DeepLearning,HandcraftedFeatures):

    print("The import that words are",TFIDF)
    print("Marks For Each Sentences" , DeepLearning)
    print(HandcraftedFeatures)

    bar = bg.blue + 'Printing the Keywords' + bg.rs
    foo = fg.red + "The Keywords Are" + fg.rs
    print(foo)
    print(bar)
    bar = bg.blue + 'This has a blue background!' + bg.rs
    baz = ef.italic + 'This is italic text' + rs.italic
    qux = fg(201) + 'This is pink text using 8bit colors' + fg.rs
    qui = fg(255, 10, 10) + 'This is red text using 24bit colors.' + fg.rs

    fg.orange = Style(RgbFg(255, 150, 50))
    buf = fg.orange + 'Yay, Im orange.' + fg.rs
    print(foo, bar, baz, qux, qui, buf, sep='\n')