
from embedding_as_service.text.encode import Encoder

def XLnetembeddings(Answer):
    print("Step 1 : Embedding Words through XLnet")
    en = Encoder(embedding='xlnet', model='xlnet_large_cased', max_seq_length=256)
    vecs = en.encode(texts=[Answer])
    return vecs


def BERTembeddings(Answer):
    print("Embedding Words through BERT")
    en = Encoder(embedding='bert', model='bert_base_cased', max_seq_length=256)
    vecs = en.encode(texts=[Answer])
    return vecs
