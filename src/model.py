"""
src/model.py

Kien truc multimodal cho flood detection, toi uu cho GPU 4GB VRAM:
- Image branch: ConvNeXt-Tiny pretrained (qua timm), FREEZE phan lon backbone,
  chi fine-tune vai stage cuoi -> giam dang ke so luong tham so can train
  va VRAM dung cho gradient/optimizer state.
- Text branch: vector TF-IDF (da fit san tu train set) -> MLP nho.
- Fusion: noi (concat) feature anh + feature text -> classifier head.
"""

import torch
import torch.nn as nn
import timm


class ImageBranch(nn.Module):
    """
    Wrap ConvNeXt-Tiny tu timm, dung nhu feature extractor (num_classes=0).
    Mac dinh freeze toan bo backbone, chi mo (unfreeze) N stage cuoi de
    fine-tune - giup giam VRAM va toc do train nhanh hon nhieu so voi
    train full model.
    """

    def __init__(self, model_name="convnext_tiny", pretrained=True,
                 freeze_backbone=True, unfreeze_last_n_stages=1):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.out_dim = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self._unfreeze_last_stages(unfreeze_last_n_stages)

    def _unfreeze_last_stages(self, n):
        """Mo N stage cuoi cua ConvNeXt (stages la ModuleList trong timm)."""
        if hasattr(self.backbone, "stages") and n > 0:
            stages = list(self.backbone.stages)
            for stage in stages[-n:]:
                for param in stage.parameters():
                    param.requires_grad = True
        # luon mo lop norm cuoi (nho, nhung quan trong de adapt feature)
        if hasattr(self.backbone, "norm_pre"):
            for param in self.backbone.norm_pre.parameters():
                param.requires_grad = True

    def forward(self, x):
        return self.backbone(x)

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TextBranch(nn.Module):
    """MLP nho bien doi vector TF-IDF thanh feature vector dung de fusion."""

    def __init__(self, input_dim, hidden_dim=128, output_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
        )
        self.out_dim = output_dim

    def forward(self, x):
        return self.net(x)


class FloodClassifier(nn.Module):
    """
    Model tong: ImageBranch + TextBranch -> concat -> classifier head.

    tfidf_dim: so chieu cua vector TF-IDF (= max_features luc fit vectorizer,
               phai khop voi vectorizer da fit tren train set)
    """

    def __init__(self, tfidf_dim, image_model_name="convnext_tiny", num_classes=2,
                 freeze_backbone=True, unfreeze_last_n_stages=1,
                 text_hidden_dim=128, text_output_dim=64, dropout=0.3):
        super().__init__()
        self.image_branch = ImageBranch(
            model_name=image_model_name, pretrained=True,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_stages=unfreeze_last_n_stages,
        )
        self.text_branch = TextBranch(tfidf_dim, text_hidden_dim, text_output_dim, dropout)

        fusion_dim = self.image_branch.out_dim + self.text_branch.out_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, text_vec):
        img_feat = self.image_branch(image)
        txt_feat = self.text_branch(text_vec)
        fused = torch.cat([img_feat, txt_feat], dim=1)
        return self.classifier(fused)


if __name__ == "__main__":
    # Quick sanity check - chay: python src/model.py
    TFIDF_DIM = 300  # se khop voi max_features dat khi fit TfidfVectorizer

    model = FloodClassifier(tfidf_dim=TFIDF_DIM)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tong so tham so: {total_params:,}")
    print(f"So tham so se train: {trainable_params:,} "
          f"({trainable_params / total_params * 100:.1f}%)")

    # Test forward voi batch gia
    dummy_image = torch.randn(2, 3, 224, 224)
    dummy_text = torch.randn(2, TFIDF_DIM)
    output = model(dummy_image, dummy_text)
    print(f"Output shape: {output.shape}")  # ky vong: (2, 2)