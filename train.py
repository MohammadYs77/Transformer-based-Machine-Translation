import torch, plotter, preprocess, loader, argparse
import torch.nn as nn
import torch.optim as optim
from model import Transformer  # Your Transformer model
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
# nltk.download('punkt')
# nltk.download('punkt_tab')


def truncate_at_eos(token_ids, eos_id):
    return token_ids[:token_ids.index(eos_id)] if eos_id in token_ids else token_ids

def calculate_bleu_nltk(model, dataloader, tokenizer):
    model.eval()
    smoothie = SmoothingFunction().method4
    all_refs = []
    all_hyps = []

    eos_id = tokenizer.piece_to_id("</s>")

    with torch.no_grad():
        for src_tokens, tgt_tokens in dataloader:
            src_tokens = src_tokens.to(DEVICE)
            tgt_tokens = tgt_tokens.to(DEVICE)

            output = model(src_tokens, tgt_tokens)
            output_ids = torch.argmax(output, dim=-1)

            for i in range(output_ids.size(0)):
                pred_ids = truncate_at_eos(output_ids[i].tolist(), eos_id)
                tgt_ids = truncate_at_eos(tgt_tokens[i].tolist(), eos_id)

                pred_sentence = tokenizer.decode(pred_ids).replace("▁", " ").strip()
                ref_sentence = tokenizer.decode(tgt_ids).replace("▁", " ").strip()

                pred_tokens = word_tokenize(pred_sentence.lower())
                ref_tokens = word_tokenize(ref_sentence.lower())

                all_refs.append([ref_tokens])
                all_hyps.append(pred_tokens)

    return corpus_bleu(all_refs, all_hyps, smoothing_function=smoothie)


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
def train(model, criterion, optimizer, scheduler, dataloader):
    # print('In train')
    model.train()
    total_loss = 0
    max_bleu = 0
    loss_hist = []
    val_loss_hist = []
    bleu_hist = []
    
    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch}')
        print('-' * 10)
        for batch_idx, (src_tokens, tgt_tokens) in enumerate(dataloader['train']):
            # print(f'batch_idx: {batch_idx}/{len(tr_loader)}')
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

            # Logging
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_loss = total_loss / LOG_INTERVAL
                print(f"Epoch [{epoch+1}/{EPOCHS}], Step [{batch_idx+1}/{len(tr_loader)}], Loss: {avg_loss:.4f}")
                tmp_bleu = calculate_bleu_nltk(model, dataloader['validation'], tokenizer=sp)
                if tmp_bleu > max_bleu:
                    max_bleu = tmp_bleu
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)
                loss_hist.append(avg_loss)
                val_loss_hist.append(validate(model, dataloader['validation']))
                bleu_hist.append(tmp_bleu)
                total_loss = 0
                model.train()

        # Calculate validation loss
        val_loss = validate(model, dataloader['validation'])
        print(f"📉 Validation Loss: {val_loss:.4f}")

        # Calculate BLEU score
        bleu = calculate_bleu_nltk(model, dataloader['validation'], tokenizer=sp)
        print(f"🌍 BLEU Score: {bleu * 100:.4f}")
        model.train()
        # Save model checkpoint after each epoch
        # torch.save(model.state_dict(), MODEL_SAVE_PATH)
        # print(f"✅ Model saved after epoch {epoch+1}")
        print('-' * 20 + '\n')
    
    history = {'loss': loss_hist, 'val_loss': val_loss_hist, 'bleu': bleu_hist}
    return history



# Run training
if __name__ == '__main__':
    
    parse = argparse.ArgumentParser()

    parse.add_argument('-e', '--epoch', dest='epoch', type=int, default=3)
    parse.add_argument('-b', '--batch',dest='batch', type=int, default=16)
    parse.add_argument('-lr', '--learning_rate',dest='lr', type=int, default=7e-4)
    
    args = vars(parse.parse_args())
    
    # === CONFIGURATION ===
    GENERATOR = torch.Generator().manual_seed(42)
    BATCH_SIZE = args['batch']
    EPOCHS = args['epoch']
    LEARNING_RATE = args['lr']
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_SAVE_PATH = "transformer_model.pth"
    SRC_PATH = './training/europarl-v7.de-en.en'
    TGT_PATH = './training/europarl-v7.de-en.de'
    LOG_INTERVAL = 100  
    GRAD_CLIP = 1.0
    SUBSET_SIZE = 0.1

    # Load SentencePiece tokenizer
    sp = preprocess.load_tokenizer()

    # Load dataset
    tr_loader, val_loader = loader.load_data(SRC_PATH,
                                            TGT_PATH,
                                            BATCH_SIZE,
                                            generator=GENERATOR,
                                            subset_size=SUBSET_SIZE)

    dataloader = {'train': tr_loader, 'validation': val_loader} 
    
    model = Transformer(
        num_layers=6, embed_size=512, num_heads=8,
        dff=2048, vocab_size=32000, dropout=0.1).to(DEVICE)

    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding token
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98), eps=1e-9)

    # Learning rate scheduler (Inverse Square Root Decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step+1) ** -0.5, (step+1) * (4000 ** -1.5)))

    
    history = train(model, criterion, optimizer, scheduler, dataloader)
    plotter.plot_loss_bleu(history, figsize=(12, 6))
