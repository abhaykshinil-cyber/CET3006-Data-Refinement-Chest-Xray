"""
Chest X-Ray Pneumonia Classifier — Model Architecture
======================================================
Base model  : ResNet18 (ImageNet pretrained)
Head        : Dropout(p=0.5) → Linear(512 → 2)
Loss        : CrossEntropyLoss
Device      : CUDA if available, else CPU

Architecture Modifications
---------------------------
ResNet18's default final layer is:
    (fc): Linear(in_features=512, out_features=1000, bias=True)

We replace it with a Sequential block:
    (fc): Sequential(
        Dropout(p=0.5)          ← regularisation to reduce overfitting
        Linear(512 → 2)         ← binary classification (NORMAL vs PNEUMONIA)
    )

The rest of the network (all convolutional + batch-norm layers) retains its
ImageNet pretrained weights, giving us strong feature representations from day 1.
All parameters are set trainable (full fine-tuning); freeze the backbone early
layers yourself if you want faster/cheaper training on a small dataset.
"""

import torch
import torch.nn as nn
from torchvision import models

# ─────────────────────────────────────────────
# Label reference
# ─────────────────────────────────────────────
NUM_CLASSES = 2          # 0 = NORMAL, 1 = PNEUMONIA
DROPOUT_P   = 0.5


# ─────────────────────────────────────────────
# 1. Model definition
# ─────────────────────────────────────────────
class ChestXRayClassifier(nn.Module):
    """
    Binary chest X-ray classifier built on pretrained ResNet18.

    Parameters
    ----------
    num_classes : int   — number of output classes (default 2)
    dropout_p   : float — dropout probability applied before the FC layer
    pretrained  : bool  — load ImageNet weights for the backbone
    """

    def __init__(
        self,
        num_classes: int   = NUM_CLASSES,
        dropout_p:   float = DROPOUT_P,
        pretrained:  bool  = True,
    ):
        super().__init__()

        # ── Load backbone ─────────────────────────────────────────────────
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # in_features of the original fc layer = 512
        in_features = backbone.fc.in_features   # 512

        # ── Replace classification head ───────────────────────────────────
        #   Original : Linear(512 → 1000)
        #   New      : Dropout(0.5) → Linear(512 → 2)
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

        self.model = backbone

        # ── Loss function ─────────────────────────────────────────────────
        # CrossEntropyLoss combines LogSoftmax + NLLLoss internally.
        # Works directly with raw logits — do NOT apply softmax before it.
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor  shape (B, 3, 224, 224)

        Returns
        -------
        logits : torch.Tensor  shape (B, 2)
        """
        return self.model(x)

    def compute_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Convenience wrapper around CrossEntropyLoss.

        Parameters
        ----------
        logits  : (B, 2)  — raw model outputs
        targets : (B,)    — integer class labels {0, 1}

        Returns
        -------
        loss : scalar tensor
        """
        return self.criterion(logits, targets)


# ─────────────────────────────────────────────
# 2. Device helper
# ─────────────────────────────────────────────
def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"  GPU detected : {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("  No GPU found — running on CPU.")
    return device


# ─────────────────────────────────────────────
# 3. Model factory (recommended entry point)
# ─────────────────────────────────────────────
def build_model(
    num_classes: int   = NUM_CLASSES,
    dropout_p:   float = DROPOUT_P,
    pretrained:  bool  = True,
) -> tuple[ChestXRayClassifier, torch.device]:
    """
    Instantiate the classifier, move it to the correct device, and return both.

    Usage
    -----
        model, device = build_model()
        logits = model(images.to(device))
        loss   = model.compute_loss(logits, labels.to(device))
    """
    device = get_device()
    model  = ChestXRayClassifier(
        num_classes=num_classes,
        dropout_p=dropout_p,
        pretrained=pretrained,
    ).to(device)
    return model, device


# ─────────────────────────────────────────────
# 4. Architecture summary helper
# ─────────────────────────────────────────────
def print_architecture(model: ChestXRayClassifier):
    """Print a human-readable summary of the model."""

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n" + "=" * 54)
    print("  ChestXRayClassifier — Architecture Summary")
    print("=" * 54)
    print(f"\n  Backbone       : ResNet18 (ImageNet pretrained)")
    print(f"  Input shape    : (B, 3, 224, 224)")
    print(f"  Output shape   : (B, {NUM_CLASSES})")
    print(f"\n  Modified head  :")
    print(f"    ├─ AdaptiveAvgPool2d  → (B, 512, 1, 1)")
    print(f"    ├─ Flatten            → (B, 512)")
    print(f"    ├─ Dropout(p=0.5)")
    print(f"    └─ Linear(512 → {NUM_CLASSES})   ← replaces original Linear(512 → 1000)")
    print(f"\n  Loss function  : CrossEntropyLoss")
    print(f"\n  Parameters     :")
    print(f"    Total     : {total_params:>12,}")
    print(f"    Trainable : {trainable_params:>12,}")
    print("=" * 54)

    print("\n  Final FC layer detail:")
    print(f"    {model.model.fc}")
    print()


# ─────────────────────────────────────────────
# 5. Quick smoke-test
# ─────────────────────────────────────────────
def smoke_test(model: ChestXRayClassifier, device: torch.device):
    """
    Forward pass a random batch to confirm shapes and loss computation.
    """
    print("\n[Smoke Test] Running forward pass with a random batch …")
    model.eval()
    with torch.no_grad():
        dummy_images  = torch.randn(4, 3, 224, 224).to(device)  # B=4
        dummy_labels  = torch.randint(0, NUM_CLASSES, (4,)).to(device)

        logits = model(dummy_images)
        loss   = model.compute_loss(logits, dummy_labels)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)

    print(f"  Input  shape : {tuple(dummy_images.shape)}")
    print(f"  Logits shape : {tuple(logits.shape)}")
    print(f"  Probs  shape : {tuple(probs.shape)}")
    print(f"  Preds        : {preds.tolist()}")
    print(f"  Labels       : {dummy_labels.tolist()}")
    print(f"  Loss         : {loss.item():.4f}")
    print("\n✓ Smoke test passed — model is ready for training.")


# ─────────────────────────────────────────────
# 6. Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("Building ChestXRayClassifier …\n")

    model, device = build_model(pretrained=True)

    print_architecture(model)
    smoke_test(model, device)

    print("\n  To use in your training loop:")
    print("  ─────────────────────────────────────────────")
    print("  from chest_xray_model import build_model")
    print()
    print("  model, device = build_model()")
    print("  optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)")
    print()
    print("  # Training step:")
    print("  logits = model(images.to(device))")
    print("  loss   = model.compute_loss(logits, labels.to(device))")
    print("  loss.backward()")
    print("  optimizer.step()")
