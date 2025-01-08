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









