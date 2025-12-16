import torch

@torch.no_grad()
def dice_coef(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits) if logits.shape[1]==1 else torch.softmax(logits, dim=1)
    targets = targets.float()
    inter = (probs * targets).sum(dim=(0,2,3))
    union = probs.sum(dim=(0,2,3)) + targets.sum(dim=(0,2,3))
    return ((2*inter + eps) / (union + eps)).mean()

@torch.no_grad()
def iou(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits) if logits.shape[1]==1 else torch.softmax(logits, dim=1)
    targets = targets.float()
    inter = (probs * targets).sum(dim=(0,2,3))
    union = probs.sum(dim=(0,2,3)) + targets.sum(dim=(0,2,3)) - inter
    return ((inter + eps) / (union + eps)).mean()
