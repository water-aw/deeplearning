import torch
import torch.nn as nn

def dice_loss(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits) if logits.shape[1] == 1 else torch.softmax(logits, dim=1)
    targets = targets.float()
    inter = (probs * targets).sum(dim=(0,2,3))
    union = probs.sum(dim=(0,2,3)) + targets.sum(dim=(0,2,3))
    dice = (2*inter + eps) / (union + eps)
    return 1 - dice.mean()

def build_loss(name, **kwargs):
    name = name.lower()
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "dice":
        return dice_loss
    if name == "bce+dice":
        bce = nn.BCEWithLogitsLoss()
        def loss_fn(logits, targets):
            return bce(logits, targets.float()) + dice_loss(logits, targets)
        return loss_fn
    raise ValueError(f"Unknown loss {name}")
