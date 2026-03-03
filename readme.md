# 🎬 Sentiment Analysis with DistilBERT — Full Fine-Tuning on SST-2

> Full fine-tuning of a pre-trained transformer for binary sentiment classification.  
> Base model accuracy: **50.5%** → Fine-tuned accuracy: **91.5%** (+41% in 3 epochs)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sonal12061/distilbert-sentiment-analysis/blob/main/DistilBERT_FineTuning.ipynb)

---

## What This Is

A complete fine-tuning pipeline for sentiment classification using `distilbert-base-uncased` on the SST-2 dataset (67K Stanford movie reviews). This is **full fine-tuning** — all 66M parameters are updated, as opposed to adapter-based methods like LoRA.

> **Analogy:** DistilBERT is a well-read graduate who has absorbed vast general knowledge during pre-training. Fine-tuning is enrolling them in a specialised course — they don't start from scratch, they just calibrate their existing knowledge for a specific task.

---

## Results

| Stage | Accuracy |
|---|---|
| Base model (randomly initialised head) | 50.5% |
| After fine-tuning (3 epochs) | 91.5% |
| **Improvement** | **+41.0%** |

Training ran for ~34 minutes on a free Colab T4 GPU.

---

## Pipeline

```
distilbert-base-uncased (pre-trained, 66M params)
        ↓
Add classification head (num_labels=2) — randomly initialised
        ↓
Load SST-2 (67,349 train / 872 val / 1,821 test samples)
        ↓
Tokenize → max_length=128, padding="max_length"
        ↓
Fine-Tune: 3 epochs, lr=2e-5, batch_size=32
        ↓
Evaluate + Inference on custom sentences
```

---

## Key Design Decisions Explained

**Why DistilBERT over BERT?**  
DistilBERT is 40% smaller and 60% faster than BERT while retaining 97% of its performance — trained via knowledge distillation. For a classification task like SST-2, the accuracy trade-off is negligible but the compute savings are significant, especially on free Colab GPUs.

**Why does accuracy start at 50.5%?**  
The pre-trained encoder is loaded with its original weights — but the classification head (`pre_classifier` + `classifier`) is randomly initialised. Random weights produce near-random output, hence ~50% on a binary task.

**Why `learning_rate=2e-5`?**  
Fine-tuning BERT-family models requires a very small learning rate. A large LR would overwrite the pre-trained representations in the first few gradient steps — a phenomenon called catastrophic forgetting. 2e-5 to 5e-5 is the validated range across NLP fine-tuning literature.

**Why `max_length=128` and not 512?**  
SST-2 sentences are short movie review snippets. 128 tokens covers 99%+ of them. Transformer attention scales as O(n²) with sequence length — going from 128 to 512 is 16x more compute per sample for no accuracy gain here.

**Why `load_best_model_at_end=True`?**  
Validation accuracy across 3 epochs: 90.5% → 90.5% → 90.3%. The model slightly degrades in epoch 3. This flag automatically restores the best checkpoint rather than the final one.

---

## Inference Examples

```python
predict_sentiment("This movie was absolutely brilliant and touching")
# POSITIVE ✅

predict_sentiment("Terrible film, complete waste of time")
# NEGATIVE ❌

predict_sentiment("The acting was not bad but the plot was boring")
# NEGATIVE ❌  ← handles negation + contrast correctly
```

---

## How to Run

**Google Colab (recommended):** Click the badge above — GPU is pre-configured.

**Locally:**
```bash
git clone https://github.com/Sonal12061/distilbert-sentiment-analysis
cd distilbert-sentiment-analysis
pip install transformers datasets evaluate accelerate torch
jupyter notebook DistilBERT_FineTuning.ipynb
```

---

## Stack

`HuggingFace Transformers` · `HuggingFace Datasets` · `PyTorch` · `Evaluate` · `Google Colab T4`

---

## How This Fits Into the Portfolio

This repo demonstrates **full fine-tuning** — all weights are updated. For a comparison of **parameter-efficient fine-tuning** (LoRA — only 0.5% of weights trained), see the companion repo:

🔗 [LoRA Fine-Tuning — TinyLlama Medical QA](https://github.com/Sonal12061/lora-tinyllama-medical-qa)

---

## ✍️ Deep Dive Blog

Not just the code — the *why* behind every decision:

📖 [Fine-Tuning DistilBERT for Sentiment Analysis — What I Actually Learned](https://medium.com/@sonal.mishra1297)

Covers: why 50.5% baseline, why lr=2e-5, what `[CLS]` does, why `max_length=128`, unexpected keys warning explained, full fine-tuning vs LoRA decision framework.

---

## References

- [DistilBERT Paper — Sanh et al. 2019](https://arxiv.org/abs/1910.01108)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [SST-2 Dataset](https://huggingface.co/datasets/stanfordnlp/sst2)