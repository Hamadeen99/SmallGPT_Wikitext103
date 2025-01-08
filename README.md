# **SmallGPT for Wikitext-103**

A PyTorch-based implementation of a small Transformer architecture for word-level language modeling. This project leverages the Wikitext-103 dataset, a BERT tokenizer, and adaptive techniques for efficient training and evaluation. The model achieves a perplexity of ~37 after partial training, demonstrating its effectiveness for language modeling tasks.

---

## **Table of Contents**
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Setup](#setup)
5. [Dataset](#dataset)
6. [Usage](#usage)
7. [Results](#results)


---

## **Project Overview**
This project implements a small GPT-like model for word-level language modeling using the Wikitext-103 dataset. It includes:
- A Transformer-based architecture for sequential data.
- Word-level tokenization using a pre-trained BERT tokenizer.
- Perplexity evaluation for model performance.
- Adaptive training techniques to handle large vocabulary sizes.

---

## **Features**
- **Custom Transformer Implementation**: Includes positional encoding, multi-head attention, and transformer blocks.
- **Tokenization**: Pre-trained BERT tokenizer for efficient tokenization.
- **Evaluation**: Perplexity calculation to measure language modeling performance.
- **Efficient Training**: Supports checkpointing and resume functionality.

---

## **Requirements**
- Python 3.8 or later
- Libraries:
  - PyTorch
  - Transformers (`pip install transformers`)
  - NumPy
  - tqdm
- Hardware: A GPU is recommended for training.

Install dependencies with:
```bash
pip install torch transformers numpy tqdm
```

---

## **Setup**
Clone the Repository:
```bash
git clone https://github.com/hamadeen99/SmallGPT_Wikitext103.git
cd SmallGPT_Wikitext103
```

---

## **Dataset**
Download Wikitext-103 Dataset:
- Download the dataset from Wikitext-103.
- Extract the files (wiki.train.tokens, wiki.valid.tokens, wiki.test.tokens) into the data/wikitext-103/ folder:

```Data Structure
SmallGPT_Wikitext103/
├── data/
│   ├── wikitext-103/
│       ├── wiki.train.tokens
│       ├── wiki.valid.tokens
│       ├── wiki.test.tokens
```
Directory Structure:

Ensure the project directory is structured as follows:
```files Structure
SmallGPT_Wikitext103/
├── AutoRegressiveWrapper.py
├── MyNLPDataSet.py
├── Utils.py
├── TransformerXY_WK103_Main.py
├── models/
│   ├── AHSelfAttention.py
│   ├── PositionEncoding.py
│   ├── SimpleTransformer.py
│   ├── TransformerBlock.py
├── checkpoint/
│   └── [model checkpoints]
├── data/
│   ├── wikitext-103/
│       ├── wiki.train.tokens
│       ├── wiki.valid.tokens
│       ├── wiki.test.tokens
```

---

##**Usage**
Train the Model:
Run the following command to start training:

```bash
python TransformerAH_WK103_Main.py
```

-Modify constants like SEQ_LENGTH and BATCH_SIZE in the TransformerXY_WK103_Main.py file for your hardware capabilities.

---


# Resume Training

Set the following constant in `TransformerXY_WK103_Main.py` to load the latest checkpoint:

```python
RESUME_TRAINING = True
```

This will resume training from the `checkpoint/` folder.

# Evaluate the Model

Evaluate the perplexity or generate text sequences during or after training by running:

```bash
python TransformerAH_WK103_Main.py
```

# Results

## Perplexity

- Achieved a perplexity score of **54.08** with `SEQ_LENGTH = 256` and `GENERATE_LENGTH = 128` after 100% training.
- After modifying `SEQ_LENGTH = 1024` and `GENERATE_LENGTH = 512`, achieved a significant perplexity reduction to **39.44** with only 12% training.

## Bits Per Character (BPC)

- BPC with `SEQ_LENGTH = 256` and `GENERATE_LENGTH = 128`: **5.7548**.
- BPC with `SEQ_LENGTH = 1024` and `GENERATE_LENGTH = 512`: **5.3**.















