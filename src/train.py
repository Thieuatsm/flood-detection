"""
src/train.py

Training loop cho FloodClassifier (ConvNeXt-Tiny + TF-IDF), toi uu cho GPU
4GB VRAM bang:
- Batch size nho + gradient accumulation (mo phong batch lon hon)
- Mixed precision (AMP) -> giam VRAM, tang toc do
- Class weight trong loss -> xu ly imbalance (3360 vs 1920)
- Luu checkpoint theo F1 tot nhat tren val set

Cach dung (dung config file):
    python src/train.py --config configs/config.yaml

Cach dung (ghi de tung argument):
    python src/train.py --config configs/config.yaml --epochs 5 --lr 5e-5
"""

import argparse
import os
import sys

import joblib
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import FloodDataset
from src.model import FloodClassifier


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_config(config_path):
    """Doc config.yaml, phang hoa cac section thanh 1 dict."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    flat = {}
    flat.update(cfg.get("data", {}))
    flat.update(cfg.get("model", {}))
    flat.update(cfg.get("training", {}))
    flat.update(cfg.get("paths", {}))
    return flat


def build_transforms(image_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def compute_class_weights(labels):
    counts = pd.Series(labels).value_counts().sort_index()
    total = counts.sum()
    weights = total / (len(counts) * counts)
    return torch.tensor(weights.values, dtype=torch.float32)


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            images    = batch["image"].to(device)
            text_vecs = batch["text_vec"].to(device)
            labels    = batch["label"].to(device)

            outputs = model(images, text_vecs)
            loss    = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    return avg_loss, acc, f1


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Su dung device: {device}")

    train_df         = pd.read_csv(args.train_csv)
    val_df           = pd.read_csv(args.val_csv)
    tfidf_vectorizer = joblib.load(args.tfidf_path)
    tfidf_dim        = len(tfidf_vectorizer.vocabulary_)

    train_tf, eval_tf = build_transforms(args.image_size)

    train_dataset = FloodDataset(train_df, args.images_dir, tfidf_vectorizer, train_tf,
                                  image_extension=args.image_extension)
    val_dataset   = FloodDataset(val_df,   args.images_dir, tfidf_vectorizer, eval_tf,
                                  image_extension=args.image_extension)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                               shuffle=True,  num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size,
                               shuffle=False, num_workers=args.num_workers, pin_memory=True)

    model = FloodClassifier(
        tfidf_dim=tfidf_dim,
        unfreeze_last_n_stages=args.unfreeze_last_n_stages,
    ).to(device)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer  = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    class_weights = compute_class_weights(train_df["label"].values).to(device)
    criterion  = nn.CrossEntropyLoss(weight=class_weights)
    scaler     = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            images    = batch["image"].to(device)
            text_vecs = batch["text_vec"].to(device)
            labels    = batch["label"].to(device)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                outputs = model(images, text_vecs)
                loss    = criterion(outputs, labels) / args.accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * args.accum_steps * images.size(0)

        train_loss = running_loss / len(train_dataset)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device, criterion)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "tfidf_dim": tfidf_dim,
                "epoch": epoch,
                "val_f1": val_f1,
            }, checkpoint_path)
            print(f"  -> Luu checkpoint moi tot nhat (val_f1={val_f1:.4f}) vao {checkpoint_path}")

    print(f"\nHoan thanh training. Best val F1: {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None,
                         help="Duong dan toi config.yaml (tuy chon)")
    # Data
    parser.add_argument("--train_csv",       default=None)
    parser.add_argument("--val_csv",         default=None)
    parser.add_argument("--images_dir",      default=None)
    parser.add_argument("--image_extension", default=None)
    # Paths
    parser.add_argument("--tfidf_path",      default=None)
    parser.add_argument("--checkpoint_dir",  default=None)
    # Training
    parser.add_argument("--epochs",          type=int,   default=None)
    parser.add_argument("--batch_size",      type=int,   default=None)
    parser.add_argument("--accum_steps",     type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--image_size",      type=int,   default=None)
    parser.add_argument("--num_workers",     type=int,   default=None)
    # Model
    parser.add_argument("--unfreeze_last_n_stages", type=int, default=None)

    args = parser.parse_args()

    # Neu co config file, dung lam gia tri mac dinh,
    # argument command line se ghi de neu duoc truyen vao
    if args.config:
        cfg = load_config(args.config)
        for key, val in cfg.items():
            if getattr(args, key, None) is None:
                setattr(args, key, val)

    # Fallback mac dinh cuoi cung neu ca 2 deu None
    defaults = dict(
        train_csv="data/processed/train.csv",
        val_csv="data/processed/val.csv",
        images_dir="data/raw/devset_images/devset_images",
        image_extension="jpg",
        tfidf_path="checkpoints/tfidf_vectorizer.pkl",
        checkpoint_dir="checkpoints",
        epochs=15,
        batch_size=8,
        accum_steps=4,
        lr=1e-4,
        image_size=224,
        num_workers=2,
        unfreeze_last_n_stages=1,
    )
    for key, val in defaults.items():
        if getattr(args, key, None) is None:
            setattr(args, key, val)

    # Ep kieu so dung - yaml co the doc 1e-4 thanh string
    args.lr                     = float(args.lr)
    args.epochs                 = int(args.epochs)
    args.batch_size             = int(args.batch_size)
    args.accum_steps            = int(args.accum_steps)
    args.image_size             = int(args.image_size)
    args.num_workers            = int(args.num_workers)
    args.unfreeze_last_n_stages = int(args.unfreeze_last_n_stages)

    main(args)