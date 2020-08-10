from embedding_as_service.text.encode import Encoder  

en = Encoder(embedding='bert', model='bert_base_cased', max_seq_length=256)