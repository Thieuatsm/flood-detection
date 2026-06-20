"""
src/dataset.py

Load va join du lieu cho bai toan Flood Detection:
- devset_images_gt.csv     (id, label)
- devset_images_metadata.json  (title, description, user_tags, ...)
- anh thuc te trong devset_images/devset_images/{id}.jpg

Sau khi join, mỗi sample co: id, label, text (title + tags + description ghep lai).
"""

import os
import json
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


def build_metadata_dataframe(metadata_json_path):
    """Doc metadata.json va chuyen thanh DataFrame phang, 1 dong / anh."""
    with open(metadata_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    records = []
    for item in raw["images"]:
        tags = item.get("user_tags") or []
        records.append({
            "id": str(item["image_id"]),
            "title": item.get("title") or "",
            "description": item.get("description") or "",
            "user_tags": " ".join(tags),
        })
    return pd.DataFrame(records)


def build_text_field(row):
    """Ghep title + user_tags + description thanh 1 chuoi text dung cho TF-IDF."""
    parts = [row.get("title", ""), row.get("user_tags", ""), row.get("description", "")]
    return " ".join(p for p in parts if p).strip()


def load_devset(gt_csv_path, metadata_json_path):
    """
    Join label (gt.csv) voi metadata text, tra ve DataFrame voi cot:
    id, label, text
    """
    gt_df = pd.read_csv(gt_csv_path)
    gt_df["id"] = gt_df["id"].astype(str)

    meta_df = build_metadata_dataframe(metadata_json_path)

    merged = gt_df.merge(meta_df, on="id", how="left")
    merged["text"] = merged.apply(build_text_field, axis=1)
    merged["text"] = merged["text"].fillna("")

    return merged[["id", "label", "text"]]


class FloodDataset(Dataset):
    """
    PyTorch Dataset cho bai toan flood detection.

    df: DataFrame voi cot id, label, text (lay tu load_devset() hoac file
        train/val/test.csv da chia san trong data/processed/)
    images_dir: duong dan toi thu muc chua anh thuc te
                (vd: data/raw/devset_images/devset_images)
    tfidf_vectorizer: sklearn TfidfVectorizer DA FIT SAN (chi fit tren train set!)
    image_transform: torchvision transform cho anh (resize, augment, normalize...)
    """

    def __init__(self, df, images_dir, tfidf_vectorizer=None, image_transform=None,
                 image_extension="jpg"):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.tfidf_vectorizer = tfidf_vectorizer
        self.image_transform = image_transform
        self.image_extension = image_extension

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = str(row["id"])
        label = int(row["label"])
        text = row["text"] if pd.notna(row["text"]) else ""

        image_path = os.path.join(self.images_dir, f"{image_id}.{self.image_extension}")
        if not os.path.exists(image_path):
            for ext in ["jpg", "jpeg", "png"]:
                alt_path = os.path.join(self.images_dir, f"{image_id}.{ext}")
                if os.path.exists(alt_path):
                    image_path = alt_path
                    break
        image = Image.open(image_path).convert("RGB")

        if self.image_transform:
            image = self.image_transform(image)

        if self.tfidf_vectorizer is not None:
            text_vec = self.tfidf_vectorizer.transform([text]).toarray()[0]
            text_vec = torch.tensor(text_vec, dtype=torch.float32)
        else:
            # chua co vectorizer (vd luc EDA) -> tra ve placeholder
            text_vec = torch.zeros(1)

        return {
            "image": image,
            "text_vec": text_vec,
            "label": torch.tensor(label, dtype=torch.long),
            "id": image_id,
        }


if __name__ == "__main__":
    # Quick test - chinh duong dan cho dung voi may ban roi chay:
    # python src/dataset.py
    GT_CSV = "data/raw/devset_images_gt.csv"
    METADATA_JSON = "data/raw/devset_images_metadata.json"

    df = load_devset(GT_CSV, METADATA_JSON)
    print(f"Tong so sample: {len(df)}")
    print(f"Phan bo label:\n{df['label'].value_counts()}")
    print(f"\nVi du 3 dong dau:")
    print(df.head(3))