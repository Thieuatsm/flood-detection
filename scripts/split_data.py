"""
scripts/split_data.py

Vi test.csv cua competition KHONG co label (chi dung de nop bai), nen de co
the danh gia model (accuracy/F1) cho CV project, ta tu chia devset (5280 anh
co label) thanh 3 phan: train / val / test, co stratify theo label de giu
ti le 2 class deu nhau giua cac phan.

Cach dung:
    python scripts/split_data.py \
        --gt_csv data/raw/devset_images_gt.csv \
        --metadata_json data/raw/devset_images_metadata.json \
        --output_dir data/processed \
        --val_size 0.15 --test_size 0.15 --seed 42
"""

import argparse
import os
import sys

from sklearn.model_selection import train_test_split

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.dataset import load_devset


def main(args):
    df = load_devset(args.gt_csv, args.metadata_json)
    print(f"Da load {len(df)} sample. Phan bo label:\n{df['label'].value_counts()}\n")

    # Buoc 1: tach train ra khoi (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=args.val_size + args.test_size,
        stratify=df["label"],
        random_state=args.seed,
    )

    # Buoc 2: tach phan con lai thanh val va test
    relative_test_size = args.test_size / (args.val_size + args.test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["label"],
        random_state=args.seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(args.output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(args.output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(args.output_dir, "test.csv"), index=False)

    print(f"Train: {len(train_df)} sample")
    print(f"Val:   {len(val_df)} sample")
    print(f"Test:  {len(test_df)} sample")
    print(f"\nDa luu vao: {args.output_dir}/{{train,val,test}}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_csv", required=True,
                         help="Duong dan toi devset_images_gt.csv")
    parser.add_argument("--metadata_json", required=True,
                         help="Duong dan toi devset_images_metadata.json")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--val_size", type=float, default=0.15)
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)