# 把train中20%的数据转移到val中，比例可以自定义
# 本脚本用于carvana数据集，当然也有一定的泛用性，可以针对不同任务做简单修改

import random
from pathlib import Path
import shutil

# Paths
root = Path("UNet/data")
train_images = root / "train" / "images"
train_masks = root / "train" / "masks"
val_images = root / "val" / "images"
val_masks = root / "val" / "masks"

# Settings
val_ratio = 0.2      # portion of train moved to val
seed = 42            # deterministic split
move_files = True    # True: move; False: copy


def find_mask_for_image(img_path: Path) -> Path:
    """Find a mask with the same stem, optional _mask suffix, any common extension."""
    stem = img_path.stem
    stem_candidates = [stem, f"{stem}_mask"]
    exts = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"]

    for st in stem_candidates:
        for ext in exts:
            cand = train_masks / f"{st}{ext}"
            if cand.exists():
                return cand

    # Fallback: any file starting with stem or stem_mask
    found = list(train_masks.glob(f"{stem}*.*"))
    if found:
        return found[0]

    raise RuntimeError(
        f"Missing mask. Tried stems {stem_candidates} with extensions png/jpg/jpeg/bmp/gif/tif/tiff."
    )


def main():
    random.seed(seed)
    val_images.mkdir(parents=True, exist_ok=True)
    val_masks.mkdir(parents=True, exist_ok=True)

    img_files = sorted([p for p in train_images.iterdir() if p.is_file()])
    if not img_files:
        raise RuntimeError("train/images has no files")

    k = max(1, int(len(img_files) * val_ratio))
    picked = set(random.sample(img_files, k))

    for img_path in img_files:
        if img_path not in picked:
            continue
        mask_path = find_mask_for_image(img_path)
        target_img = val_images / img_path.name
        target_mask = val_masks / mask_path.name
        if move_files:
            shutil.move(str(img_path), str(target_img))
            shutil.move(str(mask_path), str(target_mask))
        else:
            shutil.copy2(str(img_path), str(target_img))
            shutil.copy2(str(mask_path), str(target_mask))

    print(f"Done: moved {k} of {len(img_files)} samples to val.")


if __name__ == "__main__":
    main()
