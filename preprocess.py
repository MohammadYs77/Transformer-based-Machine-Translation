import os
import sentencepiece as spm


def build_tokenizer(corpora):
    pwd = os.path.dirname(__file__)
    pwd = pwd.replace('\\', '/')
    filepath = pwd + corpora

    # Train a SentencePiece model
    spm.SentencePieceTrainer.train(input=filepath, model_prefix='spm_joint', vocab_size=32000)

    # # Load tokenizer
    # sp = spm.SentencePieceProcessor(model_file='spm_joint.model')

    # # Tokenize a sentence
    # tokens = sp.encode("This is a test sentence!", out_type=int)
    # print(tokens)
    # print()
    # tokens = sp.encode("Das ist ein Hund!", out_type=int)
    # print(tokens)
    # print()

def load_tokenizer():
    sp = spm.SentencePieceProcessor(model_file='spm_joint.model')
    return sp