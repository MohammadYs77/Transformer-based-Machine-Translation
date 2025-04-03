import torch, preprocess
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from loader import TranslationDataset  # Your dataset class
from model import Transformer  # Your Transformer model
import sentencepiece as spm
import os

# === CONFIGURATION ===
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "transformer_model.pth"
SRC_PATH = './training/europarl-v7.de-en.en'
TGT_PATH = './training/europarl-v7.de-en.de'
LOG_INTERVAL = 100  # Print loss every X steps
GRAD_CLIP = 1.0  # Clip gradients to prevent exploding gradients

# Load SentencePiece tokenizer
sp = preprocess.load_tokenizer()

# Load dataset
# train_dataset = TranslationDataset("wmt14.en", "wmt14.de", tokenizer=sp, max_length=128)
# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
dataset = TranslationDataset(SRC_PATH, TGT_PATH, tokenizer=sp)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define model
model = Transformer(
    num_layers=3, embed_size=512, num_heads=4,
    dff=2048, vocab_size=32000, dropout=0.1
).to(DEVICE)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

# Learning rate scheduler (Inverse Square Root Decay)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda step: min((step+1) ** -0.5, (step+1) * (4000 ** -1.5))
)

# === TRAINING LOOP ===
def train():
    model.train()
    total_loss = 0

    for epoch in range(EPOCHS):
        for batch_idx, (src_tokens, tgt_tokens) in enumerate(dataloader):
            src_tokens, tgt_tokens = src_tokens.to(DEVICE), tgt_tokens.to(DEVICE)

            # Shift target sequence for decoder input (teacher forcing)
            tgt_input = tgt_tokens[:, :-1]  # Remove <EOS>
            tgt_output = tgt_tokens[:, 1:]  # Remove <SOS>

            optimizer.zero_grad()

            # Forward pass
            output = model(src_tokens, tgt_input)  # Shape: (batch, seq_len, vocab_size)

            # Compute loss
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            # Optimizer step
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

            # Logging
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_loss = total_loss / LOG_INTERVAL
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(dataloader)}], Loss: {avg_loss:.4f}")
                total_loss = 0

        # Save model checkpoint after each epoch
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✅ Model saved after epoch {epoch+1}")

# Run training
if __name__ == '__main__':
    train()
