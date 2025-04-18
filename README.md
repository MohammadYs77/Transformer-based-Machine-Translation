# Transformer-Based Machine Translation

## Abstract

This project presents the implementation of a transformer-based model tailored for English-to-German machine translation. Despite utilizing only 10% of the WMT14 dataset, the proposed model attains a BLEU score of 46.0394, demonstrating high performance and robustness in low-resource conditions.

## Key Features

- **Data Preprocessing:** Preprocessing was carried out using the *SentencePiece* tokenizer developed by **Google**, ensuring efficient and language-agnostic subword tokenization.
- **Model Architecture:** The model is built upon a customized transformer architecture comprising 8 attention heads and 6 layers each in the encoder and decoder modules.
- **Loss Function:** The training objective utilizes cross-entropy loss.
- **Evaluation Metric:** The BLEU score is adopted as the primary evaluation metric to assess translation quality.

## Dataset

To facilitate a meaningful comparison with recent state-of-the-art methodologies, the WMT14 English-German dataset—containing over one million sentence pairs—was selected for training.

## Dependencies

The project requires the following software packages:

```bash
pip install -r requirements.txt
```

**Primary Libraries:**

- Python ≥ 3.9  
- PyTorch  
- SentencePiece  
- Matplotlib  
- NLTK  

## Usage Instructions

### 1. Repository Cloning

To begin, clone the GitHub repository:

```bash
git clone https://github.com/MohammadYs77/Transformer-based-Machine-Translation.git
```

### 2. Dataset Preparation and Tokenization

The dataset can be downloaded from the [official WMT14 website](https://www.statmt.org/wmt14/translation-task.html). The files of interest are `europarl-v7.de-en.de` and `europarl-v7.de-en.en`. 

A pre-trained tokenizer (`spm.model` and `spm.vocab`) is provided. However, to train a new tokenizer, first concatenate the source files using PowerShell:

```bash
Get-Content training/europarl-v7.de-en.en, training/europarl-v7.de-en.en | Set-Content wmt_combined.txt
```

Then execute the tokenizer training script:

```bash
python preprocess.py --src wmt_combined.txt --model_prefix spm --vocab_size 32000
```

### 3. Data Pipeline

The data pipeline is implemented in `loader.py`. The `load_data` function includes several configurable parameters:

- `batch_size`: Number of English-German sentence pairs per batch.
- `tokenizer`: Accepts a custom tokenizer; otherwise, defaults to the pre-trained tokenizer.
- `max_length`: Maximum allowable sentence length.
- `subset_size`: Allows training on a random subset of the dataset.
- `generator`: Facilitates reproducibility by using a seeded generator.

### 4. Model Training

Once the data pipeline and model are configured, initiate training with:

```bash
python main.py --epochs 10 --batch 16 --resize 224 --device 0
```

## Acknowledgements

- Gratitude is extended to open-source contributors and the providers of the WMT14 dataset.
- This work draws inspiration from contemporary literature on machine translation and neural architectures.