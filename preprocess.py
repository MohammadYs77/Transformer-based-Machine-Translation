import os
import sentencepiece as spm
import argparse

def build_tokenizer(corpora, model_prefix='spm_joint', vocab_size=32000):
    pwd = os.path.dirname(__file__)
    pwd = pwd.replace('\\', '/') + '/'
    filepath = pwd + corpora

    # Train a SentencePiece model
    spm.SentencePieceTrainer.train(input=filepath,
                                                            model_prefix=model_prefix,
                                                            vocab_size=vocab_size,
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
    # build_tokenizer('wmt14_combined.txt')
    parse = argparse.ArgumentParser()

    parse.add_argument('-v', '--vocab_size', dest='vocab', type=int, default=32000)
    parse.add_argument('-s', '--src',dest='src', type=str, default='wmt_combined.txt')
    parse.add_argument('-p', '--model_prefix',dest='pfx', type=str, default='sp')
    
    args = vars(parse.parse_args())
    
    build_tokenizer(args['src'], args['pfx'], args['vocab'])