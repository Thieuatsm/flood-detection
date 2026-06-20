"""
src/evaluate.py

Load checkpoint tot nhat (best_model.pt), chay tren test set, in ra:
- Accuracy, F1, Precision, Recall
- Confusion matrix
- Luu ket qua vao outputs/metrics.json

Cach dung:
    python src/evaluate.py \
        --images_dir data/raw/devset_images/devset_images \
        --image_extension jpg
"""

import argparse
import json
import os
import sys

import joblib
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix,
                             classification_report)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import FloodDataset
from src.model import FloodClassifier

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Su dung device: {device}")

    # --- Load checkpoint ---
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    tfidf_dim  = checkpoint["tfidf_dim"]
    print(f"Checkpoint epoch {checkpoint['epoch']} | val_f1={checkpoint['val_f1']:.4f}")

    # --- Load vectorizer ---
    tfidf_vectorizer = joblib.load(args.tfidf_path)

    # --- Load test set ---
    test_df = pd.read_csv(args.test_csv)

    eval_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_dataset = FloodDataset(
        test_df, args.images_dir, tfidf_vectorizer, eval_tf,
        image_extension=args.image_extension,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    # --- Load model ---
    model = FloodClassifier(tfidf_dim=tfidf_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # --- Inference ---
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            images    = batch["image"].to(device)
            text_vecs = batch["text_vec"].to(device)
            labels    = batch["label"]

            outputs = model(images, text_vecs)
            preds   = outputs.argmax(dim=1).cpu()

            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    # --- Metrics ---
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec  = recall_score(all_labels, all_preds)
    cm   = confusion_matrix(all_labels, all_preds)

    print(f"\n=== KET QUA TREN TEST SET ({len(test_df)} anh) ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"              Pred No-Flood  Pred Flood")
    print(f"True No-Flood     {cm[0][0]:5d}         {cm[0][1]:5d}")
    print(f"True Flood        {cm[1][0]:5d}         {cm[1][1]:5d}")
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds,
                                 target_names=["No-Flood", "Flood"]))

    # --- Luu ket qua ---
    os.makedirs(args.output_dir, exist_ok=True)
    metrics = {
        "accuracy": round(acc, 4),
        "f1": round(f1, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "confusion_matrix": cm.tolist(),
        "num_test_samples": len(test_df),
        "checkpoint_epoch": checkpoint["epoch"],
    }
    out_path = os.path.join(args.output_dir, "metrics.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nDa luu metrics vao: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_csv",        default="data/processed/test.csv")
    parser.add_argument("--images_dir",      required=True)
    parser.add_argument("--checkpoint_path", default="checkpoints/best_model.pt")
    parser.add_argument("--tfidf_path",      default="checkpoints/tfidf_vectorizer.pkl")
    parser.add_argument("--output_dir",      default="outputs")
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--image_size",      type=int, default=224)
    parser.add_argument("--num_workers",     type=int, default=2)
    parser.add_argument("--image_extension", default="jpg")
    args = parser.parse_args()
    main(args)