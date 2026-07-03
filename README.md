# 🌊 Flood Detection — Multimodal Image Classifier

A multimodal deep learning project that classifies images as **flood** or **no-flood** using both visual and textual features. Built as a portfolio project targeting AI/ML Engineer internship roles.

---

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 94.6%  |
| F1        | 92.8%  |
| Precision | 89.9%  |
| Recall    | 95.8%  |

Evaluated on a held-out test set of **792 images** (15% of devset, stratified split).

### Confusion Matrix
```
                Pred No-Flood  Pred Flood
True No-Flood       473            31
True Flood           12           276
```

---

##  Architecture

```
Image (224x224)                     Text (title + tags + description)
      │                                           │
ConvNeXt-Tiny (pretrained)            TF-IDF vectorizer (300-dim)
  freeze backbone                              │
  fine-tune last stage                    MLP (300 → 128 → 64)
      │                                           │
      └──────────────── concat ──────────────────┘
                            │
                    Classifier head
                    (fusion_dim → 128 → 2)
                            │
                    Flood / No-Flood
```

- **Image branch**: ConvNeXt-Tiny pretrained on ImageNet-12k, backbone partially frozen, last stage fine-tuned
- **Text branch**: TF-IDF (max 300 features) on `title + user_tags + description` → small MLP
- **Fusion**: feature concatenation → 2-layer classifier
- **Loss**: CrossEntropyLoss with class weights (handles 64/36 imbalance)
- **Training**: batch size 8 + gradient accumulation (effective batch 32), AMP mixed precision (FP16)

---

##  Project Structure

```
flood-detection/
├── app/
│   └── app.py              # Gradio demo app
├── configs/
│   └── config.yaml         # Hyperparameters and paths
├── scripts/
│   ├── split_data.py       # Train/val/test split (stratified)
│   └── fit_tfidf.py        # Fit TF-IDF vectorizer on train set
├── src/
│   ├── dataset.py          # PyTorch Dataset, data loading
│   ├── model.py            # FloodClassifier architecture
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Evaluation on test set
│   └── infer.py            # Single-image inference
└── notebooks/
    └── eda.ipynb           # Exploratory data analysis
```

---

##  Quickstart

### 1. Clone & install
```bash
git clone https://github.com/Thieuatsm/flood-detection.git
cd flood-detection
pip install -r requirements.txt
```

### 2. Prepare data
Download the [MediaEval dataset](http://www.multimediaeval.org/) and place files under `data/raw/`:
```
data/raw/
├── devset_images/devset_images/   # flood images
├── devset_images_gt.csv           # labels
└── devset_images_metadata.json    # text metadata
```

### 3. Preprocess
```bash
# Split into train/val/test
python scripts/split_data.py \
    --gt_csv data/raw/devset_images_gt.csv \
    --metadata_json data/raw/devset_images_metadata.json

# Fit TF-IDF on train set only
python scripts/fit_tfidf.py
```

### 4. Train
```bash
python src/train.py --config configs/config.yaml
```

### 5. Evaluate
```bash
python src/evaluate.py --images_dir data/raw/devset_images/devset_images
```

### 6. Run demo app
```bash
python app/app.py
```

---

##  Training Details

| Setting              | Value                        |
|----------------------|------------------------------|
| Image model          | ConvNeXt-Tiny (timm)         |
| Image size           | 224 × 224                    |
| Batch size           | 8 (effective 32 w/ accum)    |
| Epochs               | 15                           |
| Learning rate        | 1e-4 (AdamW)                 |
| Mixed precision      | FP16 (AMP)                   |
| GPU                  | NVIDIA RTX 2050 (4GB VRAM)   |
| TF-IDF features      | 300                          |

---

##  Tech Stack

- **PyTorch** — model training
- **timm** — ConvNeXt-Tiny pretrained weights
- **scikit-learn** — TF-IDF vectorizer, evaluation metrics
- **Gradio** — demo app
- **Hugging Face Spaces** — deployment
