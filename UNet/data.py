from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class SegDataset(Dataset):
    def __init__(self, root, split="train", is_multiclass=False, num_classes=1, augment=False):
        self.img_dir = os.path.join(root, split, "images")
        self.mask_dir = os.path.join(root, split, "masks")
        self.ids = sorted(os.listdir(self.img_dir))
        self.is_multiclass = is_multiclass
        self.num_classes = num_classes
        self.tfm = transforms.Compose([
            transforms.ToTensor(),
            # 可加随机增强
        ])

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))
        img = self.tfm(img)
        mask = transforms.ToTensor()(mask)
        if self.is_multiclass:
            mask = mask.long().squeeze(0)
        return img, mask

def build_datasets(cfg):
    train_ds = SegDataset(cfg.data_root, "train", cfg.num_classes>1, cfg.num_classes)
    val_ds = SegDataset(cfg.data_root, "val", cfg.num_classes>1, cfg.num_classes)
    return train_ds, val_ds
