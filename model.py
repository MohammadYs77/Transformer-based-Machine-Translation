import torch, loader, math
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
        # print(f'shape of q in class mha: {q.shape}')
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        # print(f'shape of q in class mha after reshaping: {q.shape}')
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)
        # print(f'shape of k in class mha: {k.shape}')
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.depth).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.depth ** 0.5)
        # print(f'shape of scores in class mha: {scores.shape}')
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        # print(f'shape of attention weights: {attention_weights.shape}')
        # print(f'shape of v: {v.shape}')
        output = torch.matmul(attention_weights, v)
        # print(f'shape of output in class mha: {output.shape}')
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)
        # print(f'shape of output in class mha: {output.shape}')
        # print()
        return self.W_o(output)


class TransformerBlock(nn.Module):
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


class Transformer(nn.Module):
    def __init__(self, vocab_size, embed_size=512, num_heads=8, num_layers=4, dff=2048, dropout=0.1):
        super().__init__()
        
        self.encoder_embedding = nn.Embedding(vocab_size, embed_size)
        self.decoder_embedding = nn.Embedding(vocab_size, embed_size)
        
        self.pos_encoding_input = PositionalEncoding(embed_size)
        self.pos_encoding_target = PositionalEncoding(embed_size)

        self.encoder_layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, dff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(embed_size, num_heads, dff, dropout) for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(embed_size, vocab_size)

    def generate(self, src, max_len=128, start_symbol=1):  # assuming <s> token has ID 1
        """
        src: (batch_size, src_seq_len)
        returns: output logits (batch_size, max_len, vocab_size)
        """
        self.eval()
        batch_size = src.size(0)
        src_mask = self.make_src_mask(src)  # Implement if you haven't already

        memory = self.encoder(src, src_mask)

        # Start with <s> token for every sequence
        ys = torch.ones(batch_size, 1).fill_(start_symbol).long().to(src.device)  # (batch, 1)

        for i in range(max_len - 1):
            tgt_mask = self.make_tgt_mask(ys)
            out = self.decoder(ys, memory, tgt_mask, src_mask)
            out = self.fc_out(out)  # (batch, seq_len, vocab_size)
            next_token_logits = out[:, -1, :]  # (batch, vocab_size)

            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)  # (batch, 1)
            ys = torch.cat([ys, next_tokens], dim=1)

        return self.fc_out(ys)  # final logits (batch, max_len, vocab_size)

    def make_src_mask(self, src):
        # Mask padding tokens (assuming padding token ID = 0)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, src_len)
        return src_mask  # used in encoder and cross-attention

    def make_tgt_mask(self, tgt):
        tgt_len = tgt.size(1)
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, tgt_len)

        # Prevent attending to future tokens (causal mask)
        causal_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)  # (1, 1, tgt_len, tgt_len)

        return tgt_pad_mask & causal_mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.encoder_embedding(src)
        src = self.pos_encoding_input(src)
        for layer in self.encoder_layers:
            src = layer(src, src_mask)

        tgt = self.decoder_embedding(tgt)
        tgt = self.pos_encoding_target(tgt)
        for layer in self.decoder_layers:
            tgt = layer(tgt, tgt_mask)

        return self.fc_out(tgt)


def test_model():
    model = Transformer(32000, num_layers=6, embed_size=512, num_heads=8, dff=2048, dropout=0.1).to('cuda')

    sp, dataloader = loader.load_data(src_path='./training/europarl-v7.de-en.en',
                                    tgt_path='./training/europarl-v7.de-en.de',
                                    batch=32,
                                    return_tokenizer=True)

    # Test the dataloader
    for src_batch, tgt_batch in dataloader:
        src = src_batch.to('cuda')
        tgt = tgt_batch.to('cuda')
        # print("Source Batch:", src_batch.shape)
        # print("Target Batch:", tgt_batch.shape)
        # print()
        # print(src_batch)
        # print()
        # print(tgt_batch)
        break

    test_batch = model(src, tgt)
    # print(test_batch.shape)
    token_ids = torch.argmax(test_batch, dim=-1).tolist()
    print()
    print(sp.decode(token_ids[0]))
    
    
if __name__ == '__main__':
    test_model()
