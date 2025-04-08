import torch, nltk, preprocess, loader
import torch.nn as nn
import torch.optim as optim
from model import Transformer  # Your Transformer model
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
# nltk.download('punkt_tab')

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
tr_loader, val_loader = loader.load_data(SRC_PATH, TGT_PATH, BATCH_SIZE)

# Define model
model = Transformer(
    num_layers=3, embed_size=512, num_heads=4,
    dff=2048, vocab_size=32000, dropout=0.1).to(DEVICE)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

# Learning rate scheduler (Inverse Square Root Decay)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda step: min((step+1) ** -0.5, (step+1) * (4000 ** -1.5)))


def calculate_bleu_nltk(model, dataloader, tokenizer):
    model.eval()
    smoothie = SmoothingFunction().method4
    total_score = 0
    count = 0

    with torch.no_grad():
        for src_tokens, tgt_tokens in dataloader:
            src_tokens = src_tokens.to(DEVICE)
            tgt_tokens = tgt_tokens.to(DEVICE)
            
            # Generate translations
            output = model(src_tokens, tgt_tokens)  # (batch, seq_len, vocab)
            output_ids = torch.argmax(output, dim=-1)  # (batch, seq_len)

            for i in range(output_ids.size(0)):
                pred_ids = output_ids[i].tolist()
                tgt_ids = tgt_tokens[i].tolist()

                # Decode to strings
                pred_sentence = tokenizer.decode(pred_ids)
                ref_sentence = tokenizer.decode(tgt_ids)

                # Tokenize
                pred_tokens = word_tokenize(pred_sentence.lower())
                ref_tokens = word_tokenize(ref_sentence.lower())

                # BLEU expects list of references
                score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
                total_score += score
                count += 1
            break

    return total_score / count


# === FUNCTION: Evaluate Validation Loss ===
def validate(model, dataloader):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for src_tokens, tgt_tokens in dataloader:
            src_tokens, tgt_tokens = src_tokens.to(DEVICE), tgt_tokens.to(DEVICE)

            tgt_input = tgt_tokens[:, :-1]  # Remove <EOS>
            tgt_output = tgt_tokens[:, 1:]  # Remove <SOS>

            output = model(src_tokens, tgt_input)
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))

            total_loss += loss.item()
            break

    return total_loss / len(dataloader)


# === TRAINING LOOP ===
def train():
    # print('In train')
    model.train()
    total_loss = 0

    for epoch in range(EPOCHS):
        print(f'epoch: {epoch}')
        for batch_idx, (src_tokens, tgt_tokens) in enumerate(tr_loader):
            # print(f'batch_idx: {batch_idx}')
            src_tokens, tgt_tokens = src_tokens.to(DEVICE), tgt_tokens.to(DEVICE)

            # Shift target sequence for decoder input (teacher forcing)
            tgt_input = tgt_tokens[:, :-1]  # Remove <EOS>
            tgt_output = tgt_tokens[:, 1:]  # Remove <SOS>
            # print(f'got data')
            optimizer.zero_grad()

            # Forward pass
            output = model(src_tokens, tgt_input)  # Shape: (batch, seq_len, vocab_size)
            # print(f'outputted model')
            # Compute loss
            loss = criterion(output.reshape(-1, output.shape[-1]), tgt_output.reshape(-1))
            loss.backward()
            # print(f'computed loss')
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            # print(f'done clipping gradients')
            # Optimizer step
            optimizer.step()
            scheduler.step()
            # print(f'updated weights')
            # print()
            total_loss += loss.item()
            # val_loss = validate(model, val_loader)
            # print(f"📉 Validation Loss: {val_loss:.4f}")

            # Calculate BLEU score
            bleu = calculate_bleu_nltk(model, val_loader, tokenizer=sp)
            print(f"🌍 BLEU Score at Batch {batch_idx+1}: {bleu:.8f}")

            # Logging
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_loss = total_loss / LOG_INTERVAL
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(tr_loader)}], Loss: {avg_loss:.4f}")
                total_loss = 0

        # Calculate validation loss
        # val_loss = validate(model, val_loader)
        # print(f"📉 Validation Loss: {val_loss:.4f}")

        # Calculate BLEU score
        bleu = calculate_bleu_nltk(model, val_loader, tokenizer=sp)
        print(f"🌍 BLEU Score: {bleu:.4f}")
        
        # Save model checkpoint after each epoch
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"✅ Model saved after epoch {epoch+1}")

# Run training
if __name__ == '__main__':
    train()
