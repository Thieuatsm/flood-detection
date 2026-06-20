"""
src/infer.py

Inference 1 anh + text tuy y, tra ve nhan du doan va confidence.
File nay duoc goi boi demo app (app/app.py).

Cach dung standalone:
    python src/infer.py \
        --image_path path/to/image.jpg \
        --text "flood disaster water" \
        --checkpoint_path checkpoints/best_model.pt
"""

import argparse
import os
import sys

import joblib
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.model import FloodClassifier

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

LABELS = {0: "No Flood", 1: "Flood"}


class FloodPredictor:
    """
    Wrapper tien ich de load model 1 lan, goi predict nhieu lan.
    Demo app khoi tao 1 instance roi goi .predict() moi lan user upload anh.
    """

    def __init__(self, checkpoint_path="checkpoints/best_model.pt",
                 tfidf_path="checkpoints/tfidf_vectorizer.pkl",
                 image_size=224, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # Load vectorizer
        self.vectorizer = joblib.load(tfidf_path)
        tfidf_dim = len(self.vectorizer.vocabulary_)

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = FloodClassifier(tfidf_dim=tfidf_dim).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

        print(f"FloodPredictor san sang tren {self.device} "
              f"(checkpoint epoch {checkpoint['epoch']}, "
              f"val_f1={checkpoint['val_f1']:.4f})")

    @torch.no_grad()
    def predict(self, image: Image.Image, text: str = "") -> dict:
        """
        Nhan vao:
            image: PIL.Image
            text:  chuoi text tuy y (title, tags, mo ta... hoac de trong)
        Tra ve dict:
            label:      "Flood" hoac "No Flood"
            confidence: xac suat du doan (0.0 - 1.0)
            probs:      xac suat ca 2 class {"No Flood": ..., "Flood": ...}
        """
        # Xu ly anh
        img_tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)

        # Xu ly text
        text_vec = self.vectorizer.transform([text or ""]).toarray()[0]
        text_tensor = torch.tensor(text_vec, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward
        logits = self.model(img_tensor, text_tensor)
        probs  = F.softmax(logits, dim=1)[0]

        pred_idx    = probs.argmax().item()
        confidence  = probs[pred_idx].item()

        return {
            "label":      LABELS[pred_idx],
            "confidence": round(confidence, 4),
            "probs": {
                "No Flood": round(probs[0].item(), 4),
                "Flood":    round(probs[1].item(), 4),
            },
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path",      required=True)
    parser.add_argument("--text",            default="")
    parser.add_argument("--checkpoint_path", default="checkpoints/best_model.pt")
    parser.add_argument("--tfidf_path",      default="checkpoints/tfidf_vectorizer.pkl")
    args = parser.parse_args()

    predictor = FloodPredictor(
        checkpoint_path=args.checkpoint_path,
        tfidf_path=args.tfidf_path,
    )

    image = Image.open(args.image_path)
    result = predictor.predict(image, args.text)

    print(f"\nKet qua du doan:")
    print(f"  Label     : {result['label']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  No Flood  : {result['probs']['No Flood']:.2%}")
    print(f"  Flood     : {result['probs']['Flood']:.2%}")