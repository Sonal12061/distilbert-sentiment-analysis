# 🎬 Sentiment Analysis with DistilBERT — Full Fine-Tuning on SST-2

> Full fine-tuning of a pre-trained transformer for binary sentiment classification.  
> Base model accuracy: **~50%** → Fine-tuned accuracy: **91.5%** (+41% in 3 epochs)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sonal12061/distilbert-sentiment-analysis/blob/main/DistilBERT_FineTuning.ipynb)

---

## What This Is

A complete fine-tuning pipeline for sentiment classification using `distilbert-base-uncased` on the SST-2 dataset (67K Stanford movie reviews). This is **full fine-tuning** — all 66M parameters are updated, as opposed to adapter-based methods like LoRA.

> **Analogy:** DistilBERT is a condensed study guide written by someone who read an 800-page encyclopedia carefully. Fine-tuning is enrolling it in a specialised course — it doesn't start from scratch, it calibrates existing knowledge for a specific task.

---

## Results

| Stage | Accuracy | Weighted F1 |
|---|---|---|
| Base model (randomly initialised head) | ~50% | ~0.50 |
| After fine-tuning (3 epochs) | **91.5%** | **~0.91** |
| **Improvement** | **+41%** | **+0.41** |

Training runs in ~36 min on Colab T4 · ~60–90 min on M5 MacBook Air (MPS).

---

## Pipeline

```
distilbert-base-uncased (pre-trained, 66M params)
        ↓
Add classification head (num_labels=2) — randomly initialised
        ↓
Check SST-2 class balance (55.8% pos / 44.2% neg — acceptable)
        ↓
Tokenize → max_length=128, padding="max_length"
        ↓
Fine-Tune: 3 epochs · lr=2e-5 · batch_size=32
        ↓
Evaluate: Accuracy + Weighted F1 + Confusion Matrix
        ↓
Inference on 12 custom sentences (with confidence scores)
```

---

## What the Notebook Covers

| Section | What It Does |
|---------|-------------|
| **Device detection** | Auto-selects MPS (Apple Silicon) → CUDA → CPU |
| **Class balance check** | Bar chart of label distribution — confirms accuracy is a valid metric |
| **Baseline eval** | Accuracy + Weighted F1 before training (~50% / ~0.50) |
| **Training** | `Trainer` API with `compute_metrics` returning both accuracy and F1 per epoch |
| **Training curves** | 3 plots — training loss over steps · validation loss · accuracy + F1 per epoch |
| **Evaluation** | Full `classification_report` + normalised confusion matrix on all 872 val samples |
| **Inference** | 12 hand-picked sentences (clear + ambiguous) with predicted label and confidence % |
| **Model saving** | Saved to `fine_tuned_model/` (git-ignored — not pushed to this repo) |

---

## Key Design Decisions

**Why DistilBERT over BERT?**  
DistilBERT is 40% smaller and 60% faster while retaining 97% of BERT's performance — trained via knowledge distillation (mimicking BERT's attention patterns, not just predictions). For SST-2, the accuracy trade-off is negligible but the compute savings are significant.

**Why does accuracy start near 50%?**  
The pre-trained encoder is loaded intact — but the classification head (`pre_classifier` + `classifier`) is randomly initialised. Random weights produce near-random output, hence ~50% on a binary task. The encoder understands language deeply from day one; only the head needs training.

**Why `learning_rate=2e-5`?**  
A large LR would overwrite pre-trained representations in the first few steps — catastrophic forgetting. 2e-5–5e-5 is the validated range across NLP fine-tuning literature. Think of it as retraining a surgeon to become a cardiologist: you wouldn't make them unlearn anatomy first.

**Why `max_length=128` and not 512?**  
SST-2 sentences are short movie review snippets — 128 tokens covers 99%+ of them. Transformer attention scales as O(n²) with sequence length: going 128 → 512 is 16× more compute per sample for zero accuracy gain here.

**Why `load_best_model_at_end=True`?**  
Validation loss can rise after epoch 1 even as training loss keeps falling — an overfitting signal. This flag automatically restores the best checkpoint rather than the final one, so the deployed model is the one that actually generalised best.

**Why track Weighted F1 alongside accuracy?**  
SST-2 is slightly skewed (55.8% positive). Accuracy alone could mask a model that favours one class. If accuracy ≈ F1, the model is treating both classes fairly. A large gap is a red flag.

---

## Inference Examples

```
Prediction       Conf %    Expected    Sentence
────────────────────────────────────────────────────────────────────
POSITIVE ✅       99.8%    Positive    This movie was absolutely brilliant and touching
NEGATIVE ❌       98.1%    Negative    Terrible film, complete waste of time
NEGATIVE ❌       57.9%    Negative    The acting was not bad but the plot was boring  ← ambiguous
POSITIVE ✅       96.3%    Positive    An emotional rollercoaster that left me speechless
NEGATIVE ❌       83.4%    Negative    The visual effects were stunning but the story was hollow
```

Confidence is high on clear cases and appropriately lower on mixed-signal sentences — the model communicates its own uncertainty.

---

## How to Run

### Google Colab (easiest)
Click the badge at the top — GPU and all packages are pre-configured.

### Locally on Mac (M1/M2/M3/M4/M5)

```bash
git clone https://github.com/Sonal12061/distilbert-sentiment-analysis
cd distilbert-sentiment-analysis

# Create a virtual environment with Python 3.11 (best ML compatibility)
brew install python@3.11
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv

# Install all dependencies
.venv/bin/pip install torch transformers datasets evaluate accelerate scikit-learn seaborn matplotlib ipykernel

# Register as a Jupyter kernel
.venv/bin/python -m ipykernel install --user --name distilbert-sentiment --display-name "Python (distilbert-sentiment)"

# Open the notebook and select "Python (distilbert-sentiment)" as the kernel
jupyter notebook DistilBERT_FineTuning.ipynb
```

> **Keep your Mac awake during training:** open a separate terminal and run `caffeinate -i`

---

## Stack

`HuggingFace Transformers` · `HuggingFace Datasets` · `PyTorch` · `Evaluate` · `scikit-learn` · `seaborn` · `Google Colab T4` / `Apple MPS`

---

## How This Fits Into the Portfolio

This repo demonstrates **full fine-tuning** — all 66M weights are updated. For a comparison with **parameter-efficient fine-tuning** (LoRA — only 0.5% of weights trained), see the companion repo:

🔗 [LoRA Fine-Tuning — TinyLlama Medical QA](https://github.com/Sonal12061/lora-tinyllama-medical-qa)

---

## ✍️ Deep Dive Blog

Not just the code — the *why* behind every decision:

📖 [I Fine-Tuned DistilBERT for Sentiment Analysis — Here's What I Actually Learned](https://medium.com/p/4122ad4c42cc)

Covers: why the baseline is ~50%, why lr=2e-5 (catastrophic forgetting explained), what knowledge distillation actually means, why validation loss rising ≠ failure, how confidence scores expose model uncertainty, and when to switch from accuracy to F1.

---

## References

- [DistilBERT Paper — Sanh et al. 2019](https://arxiv.org/abs/1910.01108)
- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers)
- [SST-2 Dataset](https://huggingface.co/datasets/stanfordnlp/sst2)
