import os
import sentencepiece as spm


def build_tokenizer(corpora):
    pwd = os.path.dirname(__file__)
    pwd = pwd.replace('\\', '/') + '/'
    filepath = pwd + corpora

    # Train a SentencePiece model
    spm.SentencePieceTrainer.train(input=filepath,
                                                            model_prefix='spm_joint',
                                                            vocab_size=32000,
                                                            pad_id=0, pad_piece='<pad>',
                                                            unk_id=3, unk_piece='<unk>')

    # # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file='spm_joint.model')

    # # Tokenize a sentence
    tokens = sp.encode("This is a test sentence!", out_type=str, add_bos=True, add_eos=True)
    print(tokens)
    print(sp.encode("This is a test sentence!", out_type=int, add_bos=True, add_eos=True))
    # print(sp.id_to_piece(tokens))
    # tokens = sp.encode("Das ist ein Hund!", out_type=int)
    # print(tokens)
    # print()

def load_tokenizer():
    sp = spm.SentencePieceProcessor(model_file='spm_joint.model')
    return sp


if __name__ == '__main__':
    build_tokenizer('wmt14_combined.txt')