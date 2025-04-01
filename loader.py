import torch, preprocess
from torch.utils.data import Dataset, DataLoader


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
        src_tokens = self.tokenizer.encode(src_text, out_type=int)
        tgt_tokens = self.tokenizer.encode(tgt_text, out_type=int)

        # Add padding
        src_tokens = src_tokens[:self.max_length] + [0] * (self.max_length - len(src_tokens))
        tgt_tokens = tgt_tokens[:self.max_length] + [0] * (self.max_length - len(tgt_tokens))

        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)


def load_data(return_tokenizer=False):
    
    sp = preprocess.load_tokenizer()

    dataset = TranslationDataset("./training/europarl-v7.de-en.en", "./training/europarl-v7.de-en.de", tokenizer=sp)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # # Test the dataloader
    # for src_batch, tgt_batch in dataloader:
    #     print("Source Batch:", src_batch.shape)
    #     print("Target Batch:", tgt_batch.shape)
    #     print()
    #     print(src_batch)
    #     print()
    #     print(tgt_batch)
    #     break
    if return_tokenizer:
        return sp, dataloader
    
    return dataloader


if __name__ == "__main__":
    load_data()