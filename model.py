import torch, math
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=128):
        super(PositionalEncoding, self).__init__()

        # Create a matrix of shape (max_len, embed_size)
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, embed_size)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]  # Add positional encoding
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_size = embed_size

        assert embed_size % num_heads == 0

        self.depth = embed_size // num_heads

        self.W_q = nn.Linear(embed_size, embed_size)
        self.W_k = nn.Linear(embed_size, embed_size)
        self.W_v = nn.Linear(embed_size, embed_size)
        self.W_o = nn.Linear(embed_size, embed_size)

    def forward(self, q, k, v, mask=None):
        batch_size = q.shape[0]
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.depth ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)
        return self.W_o(output)


class Encoder(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_size),
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        return self.norm2(x + self.dropout(ff_output))


class Decoder(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_dim, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_size, num_heads)
        self.norm1 = nn.LayerNorm(embed_size)

        self.cross_attn = MultiHeadAttention(embed_size, num_heads)
        self.norm2 = nn.LayerNorm(embed_size)

        self.ff = nn.Sequential(
            nn.Linear(embed_size, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_size)
        )
        self.norm3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None, src_mask=None):
        attn1 = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn1))
        
        attn2 = self.cross_attn(x, enc_out, enc_out, src_mask)
        x = self.norm2(x + self.dropout(attn2))

        # Feed-forward
        ff_out = self.ff(x)
        return self.norm3(x + self.dropout(ff_out))


class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_heads=8, num_layers=4, dff=2048, dropout=0.1):
        super().__init__()
        
        self.encoder_embedding = nn.Embedding(vocab_size, embed_size)
        self.decoder_embedding = nn.Embedding(vocab_size, embed_size)
        
        self.pos_encoding_input = PositionalEncoding(embed_size)
        self.pos_encoding_target = PositionalEncoding(embed_size)

        self.encoder_layers = nn.ModuleList([
            Encoder(embed_size, num_heads, dff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            Decoder(embed_size, num_heads, dff, dropout) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.encoder_embedding(src)
        src = self.pos_encoding_input(src)
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        tgt = self.decoder_embedding(tgt)
        tgt = self.pos_encoding_target(tgt)
        for layer in self.decoder_layers:
            tgt = layer(tgt, src, tgt_mask)

        return self.fc_out(tgt)


def test_model(model, src, tokenizer, device):

    tokenized_sents = [tokenizer.encode(each, out_type=int, add_bos=True, add_eos=True) for each in src]
    tokenized_sents = [each + [0] * (128 - len(each)) for each in tokenized_sents]
    tokenized_sents = [torch.tensor(each) for each in tokenized_sents]
    tokenized_sents = torch.stack(tokenized_sents).to(device)
    # print(f'shape of tokenized_sents: {tokenized_sents[0]}')
    output = model(tokenized_sents, tokenized_sents)  # (batch, seq_len, vocab)
    print(f'shape of output: {output.shape}')
    output = torch.argmax(output, dim=-1)  # (batch, seq_len)
    print(f'shape of output after argmax: {output.shape}')
    for i in range(output.size(0)):
        pred_ids = output[i].tolist()
        pred_sentence = tokenizer.decode(pred_ids)
        print(f'Predicted Sentence: {pred_sentence}')
        # break  # Remove this to see all sentences



if __name__ == '__main__':

    import sentencepiece as spm
    tokenizer = spm.SentencePieceProcessor(model_file='spm_joint.model')
    
    PATH = './transformer_model.pth'
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = Transformer(vocab_size=32000,
                        embed_size=512,
                        num_heads=8,
                        num_layers=6,
                        dff=2048,
                        dropout=0.1).to('cuda')
    model.load_state_dict(torch.load(PATH, weights_only=True))
    model.eval()
    
    src = ['This is a test!.', 'I am trying to write a translator.', 'I like them big!']
    
    test_model(model, src, tokenizer, DEVICE)
