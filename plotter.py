import matplotlib.pyplot as plt

def plot_loss_bleu(history, figsize=(12,6)):
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Steps (or batches)')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['bleu'], label='BLEU Score')
    plt.xlabel('Steps (or batches)')
    plt.ylabel('BLEU Score')
    plt.legend()
    plt.show()