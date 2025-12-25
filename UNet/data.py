from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import InterpolationMode
from PIL import Image
from pathlib import Path
import os

class SegDataset(Dataset):
    def __init__(self, root, split="train", is_multiclass=False, num_classes=1, augment=False):
        self.img_dir = Path(root) / split / "images"
        self.mask_dir = Path(root) / split / "masks"
        self.ids = sorted(os.listdir(self.img_dir))
        self.is_multiclass = is_multiclass
        self.num_classes = num_classes
        self.size = (1918, 1280)  # (H, W)
        self.tfm = transforms.Compose([
            transforms.ToTensor(),
            # 可加随机增强
        ])

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = Image.open(self.img_dir / name).convert("RGB")
        mask_path = self._find_mask(name)
        mask = Image.open(mask_path)
        img = F.resize(img, self.size)
        mask = F.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)
        img = self.tfm(img)
        mask = transforms.ToTensor()(mask)
        if self.is_multiclass:
            mask = mask.long().squeeze(0)
        return img, mask

    def _find_mask(self, name: str) -> Path:
        """Find mask matching image name, supporting _mask suffix and common extensions."""
        stem = Path(name).stem
        for s in (stem, f"{stem}_mask"):
            for ext in (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"):
                cand = Path(self.mask_dir) / f"{s}{ext}"
                if cand.exists():
                    return cand
        found = list(Path(self.mask_dir).glob(f"{stem}*.*"))
        if found:
            return found[0]
        raise FileNotFoundError(f"mask not found for {name}")

def build_datasets(cfg):
    train_ds = SegDataset(cfg.data_root, "train", cfg.num_classes>1, cfg.num_classes)
    val_ds = SegDataset(cfg.data_root, "val", cfg.num_classes>1, cfg.num_classes)
    return train_ds, val_ds
