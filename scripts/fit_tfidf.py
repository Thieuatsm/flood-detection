"""
scripts/fit_tfidf.py

Fit TfidfVectorizer CHI TREN TRAIN SET (de tranh data leakage tu val/test),
sau do luu vectorizer lai bang joblib de dung lai nhat quan luc train,
evaluate, va inference.

Cach dung:
    python scripts/fit_tfidf.py \
        --train_csv data/processed/train.csv \
        --output_path checkpoints/tfidf_vectorizer.pkl \
        --max_features 300
"""

import argparse
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def main(args):
    df = pd.read_csv(args.train_csv)
    texts = df["text"].fillna("").tolist()

    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        lowercase=True,
        ngram_range=(1, 1),
    )
    vectorizer.fit(texts)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    joblib.dump(vectorizer, args.output_path)

    print(f"Da fit TF-IDF tren {len(texts)} text (train set).")
    print(f"So chieu vector (vocab size): {len(vectorizer.vocabulary_)}")
    print(f"Da luu vectorizer vao: {args.output_path}")
    print(f"\n--> Dung tfidf_dim = {len(vectorizer.vocabulary_)} khi tao FloodClassifier")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", default="data/processed/train.csv")
    parser.add_argument("--output_path", default="checkpoints/tfidf_vectorizer.pkl")
    parser.add_argument("--max_features", type=int, default=300)
    args = parser.parse_args()
    main(args)