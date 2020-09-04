
from embedding_as_service.text.encode import Encoder

def XLnetembeddings(Answer):

    en = Encoder(embedding='xlnet', model='xlnet_large_cased', max_seq_length=256)
    vecs = en.encode(texts=[Answer])
    return vecs


def BERTembeddings(Answer):

    en = Encoder(embedding='bert', model='bert_base_cased', max_seq_length=256)
    vecs = en.encode(texts=[Answer])
    return vecs
