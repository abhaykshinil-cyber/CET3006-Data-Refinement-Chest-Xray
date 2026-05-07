"""
model.py - Backbone-agnostic binary classifier for Chest X-Ray pneumonia detection.

Public API
----------
    ChestXRayClassifier               -> nn.Module
    build_model(pretrained, backbone) -> (model, device)
    load_checkpoint(path, device, backbone) -> model with best weights loaded
"""

import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from config import BACKBONE, NUM_CLASSES, DROPOUT_P


class ChestXRayClassifier(nn.Module):
    """
    Backbone-agnostic binary chest X-ray classifier.

    Supported backbones
    -------------------
    "resnet18"        : Linear(512  -> 2),  ~11M params
    "efficientnet_b0" : Linear(1280 -> 2),  ~5.3M params -- lighter, faster

    Loss
    ----
    compute_loss()          : standard mean CrossEntropyLoss
    compute_loss_weighted() : per-sample weighted CrossEntropyLoss (soft cleaning)
    """

    def __init__(
        self,
        num_classes: int   = NUM_CLASSES,
        dropout_p:   float = DROPOUT_P,
        pretrained:  bool  = True,
        backbone:    str   = BACKBONE,
    ):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

        if backbone == "efficientnet_b0":
            weights        = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net            = models.efficientnet_b0(weights=weights)
            in_features    = net.classifier[1].in_features      # 1280
            net.classifier = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes),
            )
            self.model = net
        else:
            # default: resnet18
            weights     = models.ResNet18_Weights.DEFAULT if pretrained else None
            net         = models.resnet18(weights=weights)
            in_features = net.fc.in_features                    # 512
            net.fc      = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes),
            )
            self.model = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, 224, 224)  ->  logits: (B, 2)"""
        return self.model(x)

    def compute_loss(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Standard mean CrossEntropyLoss."""
        return self.criterion(logits, targets)

    def compute_loss_weighted(
        self,
        logits:  torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-sample weighted CrossEntropyLoss.

        Instead of removing suspicious samples we keep them all but downweight
        their gradient contribution proportionally to detected uncertainty.
        This retains dataset size while reducing the influence of likely
        mislabelled examples -- a soft alternative to hard removal.

        weights : (B,) tensor of per-sample weights in [weight_floor, 1.0]
        """
        per_sample = F.cross_entropy(logits, targets, reduction="none")
        return (per_sample * weights).mean()


def get_device() -> torch.device:
    """Return CUDA if available, else CPU."""
    if torch.cuda.is_available():
        print(f"  GPU : {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("  No GPU found -- using CPU.")
    return torch.device("cpu")


def build_model(
    pretrained:  bool  = True,
    num_classes: int   = NUM_CLASSES,
    dropout_p:   float = DROPOUT_P,
    backbone:    str   = BACKBONE,
) -> tuple:
    """
    Instantiate classifier and move to device.

    Parameters
    ----------
    backbone : "resnet18" | "efficientnet_b0"

    Returns
    -------
    (model, device)
    """
    device = get_device()
    model  = ChestXRayClassifier(num_classes, dropout_p, pretrained, backbone).to(device)
    print(f"  Backbone: {backbone}")
    return model, device


def load_checkpoint(
    path:     pathlib.Path,
    device:   torch.device,
    backbone: str = BACKBONE,
) -> "ChestXRayClassifier":
    """
    Load saved weights from `path` into a fresh (architecture-only) model.

    Parameters
    ----------
    path     : .pth checkpoint file written by EarlyStopping
    device   : where to map the tensors
    backbone : must match the backbone used during training

    Returns
    -------
    model in eval mode with best weights loaded
    """
    model, _ = build_model(pretrained=False, backbone=backbone)
    state    = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  Weights loaded from {path.name}")
    return model
