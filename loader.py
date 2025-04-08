import torch, preprocess
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


class TranslationDataset(Dataset):
    def __init__(self, src_file, tgt_file, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load English-German sentence pairs
        with open(src_file, 'r', encoding='utf-8') as f:
            self.src_sentences = f.readlines()
        with open(tgt_file, 'r', encoding='utf-8') as f:
            self.tgt_sentences = f.readlines()

    def __len__(self):
        return len(self.src_sentences)

    def __getitem__(self, idx):
        src_text = self.src_sentences[idx].strip()
        tgt_text = self.tgt_sentences[idx].strip()

        # Tokenize sentences
        src_tokens = self.tokenizer.encode(src_text, out_type=int, add_bos=True, add_eos=True)
        tgt_tokens = self.tokenizer.encode(tgt_text, out_type=int, add_bos=True, add_eos=True)

        # Add padding
        src_tokens = src_tokens[:self.max_length] + [0] * (self.max_length - len(src_tokens))
        tgt_tokens = tgt_tokens[:self.max_length] + [0] * (self.max_length - len(tgt_tokens))

        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)


def load_data(src_path, tgt_path, batch, test_size=.9, tokenizer=None, return_tokenizer=False):
    
    if tokenizer is None:
        tokenizer = preprocess.load_tokenizer()

    dataset = TranslationDataset(src_path, tgt_path, tokenizer=tokenizer)
    
    train_size = int(test_size * len(dataset))  # 90% training
    val_size = len(dataset) - train_size  # 10% validation

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=True)

    if return_tokenizer:
        return tokenizer, train_loader, val_loader
    else:
        return train_loader, val_loader


if __name__ == "__main__":
    
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file='spm_joint.model')
    
    train_loader, val_loader = load_data(src_path='./training/europarl-v7.de-en.en',
                                                                    tgt_path='./training/europarl-v7.de-en.de',
                                                                    batch=32)
    
    for src_batch, tgt_batch in train_loader:
        src = src_batch.to('cuda')
        tgt = tgt_batch.to('cuda')
        print("Source Batch:", src_batch.shape)
        print("Target Batch:", tgt_batch.shape)
        # print()
        # print(src_batch)
        print()
        print(src_batch[0])
        print()
        print(sp.id_to_piece(src_batch[0].tolist()))
        break